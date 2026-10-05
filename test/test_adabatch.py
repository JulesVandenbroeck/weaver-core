import math
import copy
import unittest
from types import SimpleNamespace

import torch

from weaver.utils.nn.adabatch import adabatch_accum_steps, adabatch_setup, _rescale_lr
from weaver.utils.nn.tools import train_classification, train_regression


def _make_args(**kwargs):
    args = dict(
        adabatch_factor=2,
        adabatch_every=10,
        adabatch_start_epoch=None,
        adabatch_max_accum=8,
        adabatch_lr_exp=0.5,
        batch_size=16,
        steps_per_epoch=None,
    )
    args.update(kwargs)
    return SimpleNamespace(**args)


def _warmup_cos(total_steps=100, warmup_steps=10):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    return lr_lambda


class AccumStepsTest(unittest.TestCase):
    def test_disabled(self):
        for epoch in range(50):
            self.assertEqual(adabatch_accum_steps(epoch, factor=1, every=10), 1)

    def test_growth_schedule(self):
        steps = [adabatch_accum_steps(epoch, factor=2, every=10, max_accum=8) for epoch in range(50)]
        self.assertEqual(steps, [1] * 10 + [2] * 10 + [4] * 10 + [8] * 20)

    def test_start_epoch_and_cap(self):
        steps = [adabatch_accum_steps(epoch, factor=3, every=2, start=1, max_accum=10) for epoch in range(6)]
        self.assertEqual(steps, [1, 3, 3, 9, 9, 10])


class RescaleLrTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(4, 2)

    def _opt(self, lr_mult=False):
        if lr_mult:
            groups = [{"params": [self.model.weight]}, {"params": [self.model.bias], "lr": 0.2}]
        else:
            groups = self.model.parameters()
        return torch.optim.SGD(groups, lr=0.1)

    def test_lambda_lr(self):
        opt = self._opt(lr_mult=True)
        lr_lambda = _warmup_cos()
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        for _ in range(30):
            scheduler.step()
        _rescale_lr(opt, scheduler, 4)
        for base_lr, group in zip([0.1, 0.2], opt.param_groups):
            self.assertAlmostEqual(group["lr"], 4 * base_lr * lr_lambda(30))
        scheduler.step()
        for base_lr, group in zip([0.1, 0.2], opt.param_groups):
            self.assertAlmostEqual(group["lr"], 4 * base_lr * lr_lambda(31))

    def test_chainable_scheduler(self):
        opt = self._opt()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2], gamma=0.1)
        _rescale_lr(opt, scheduler, 2)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.2)
        _rescale_lr(opt, scheduler, 4)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.4)
        scheduler.step()
        scheduler.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.04)

    def test_resume(self):
        # rescaling after reloading the optimizer state gives the same lr as an uninterrupted run
        for make_scheduler in (
            lambda opt, last_epoch: torch.optim.lr_scheduler.LambdaLR(opt, _warmup_cos(), last_epoch=last_epoch),
            lambda opt, last_epoch: torch.optim.lr_scheduler.MultiStepLR(opt, [5], gamma=0.1, last_epoch=last_epoch),
        ):
            opt = self._opt()
            scheduler = make_scheduler(opt, -1)
            for scale in [1, 2, 2]:
                _rescale_lr(opt, scheduler, scale)
                for _ in range(3):
                    scheduler.step()
            opt_state = copy.deepcopy(opt.state_dict())
            _rescale_lr(opt, scheduler, 4)
            expected = opt.param_groups[0]["lr"]

            resumed_opt = self._opt()
            resumed_opt.load_state_dict(opt_state)
            resumed_scheduler = make_scheduler(resumed_opt, 8)
            _rescale_lr(resumed_opt, resumed_scheduler, 4)
            self.assertAlmostEqual(resumed_opt.param_groups[0]["lr"], expected)

    def test_one_cycle_not_supported(self):
        opt = self._opt()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=10)
        with self.assertRaises(NotImplementedError):
            _rescale_lr(opt, scheduler, 2)

    def test_setup(self):
        opt = self._opt()
        args = _make_args(adabatch_every=1)
        self.assertEqual(adabatch_setup(args, opt, None, epoch=0), {"accum_steps": 1})
        self.assertEqual(adabatch_setup(args, opt, None, epoch=2), {"accum_steps": 4})
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.1 * 4**0.5)


class _FakeLoader:
    def __init__(self, batches, input_names, label_names):
        self.batches = batches
        self.dataset = SimpleNamespace(config=SimpleNamespace(input_names=input_names, label_names=label_names))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class GradAccumulationTest(unittest.TestCase):
    extra_args = {"args": SimpleNamespace(use_amp=False, amp_dtype="bf16")}

    def _run(self, train_fn, loss_func, labels, num_outputs, accum_steps):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, num_outputs))
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        x = torch.randn(32, 4)
        batch_size = len(x) // accum_steps
        batches = [
            ({"x": x[i : i + batch_size]}, {"y": labels[i : i + batch_size]}, {}) for i in range(0, len(x), batch_size)
        ]
        loader = _FakeLoader(batches * 2, ["x"], ["y"])
        train_fn(
            model,
            loss_func,
            opt,
            None,
            loader,
            torch.device("cpu"),
            0,
            extra_args=self.extra_args,
            accum_steps=accum_steps,
        )
        return [p.detach() for p in model.parameters()]

    def _check(self, train_fn, loss_func, labels, num_outputs):
        # 2 optimizer steps with batches of 32, or with 2 x 16 / 4 x 8 accumulated batches, give the same model
        reference = self._run(train_fn, loss_func, labels, num_outputs, accum_steps=1)
        for accum_steps in (2, 4):
            params = self._run(train_fn, loss_func, labels, num_outputs, accum_steps=accum_steps)
            for p, p_ref in zip(params, reference):
                torch.testing.assert_close(p, p_ref)

    def test_classification(self):
        labels = torch.arange(32) % 3
        self._check(train_classification, torch.nn.CrossEntropyLoss(), labels, num_outputs=3)

    def test_regression(self):
        labels = torch.linspace(-1, 1, 32)
        self._check(train_regression, torch.nn.MSELoss(), labels, num_outputs=1)


if __name__ == "__main__":
    unittest.main()
