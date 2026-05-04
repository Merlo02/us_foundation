"""
Cosine LR scheduler with linear warm-up.

Direct port of the identical wrappers found in:
  - BioFoundation  schedulers/cosine.py
  - TimeFM         schedulers/cosine.py

Both repos wrap ``timm.scheduler.cosine_lr.CosineLRScheduler``, which
operates in *step* space rather than epoch space (``t_in_epochs=False``).

PyTorch Lightning integration
------------------------------
In ``configure_optimizers`` return::

    {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": CosineLRSchedulerWrapper(...),
            "interval": "step",
            "frequency": 1,
        },
    }

and override ``lr_scheduler_step``::

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)

The ``step_update`` call is timm's API — it is NOT the standard
``scheduler.step()`` from torch.optim.lr_scheduler.
"""

from __future__ import annotations

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler


class CosineLRSchedulerWrapper(CosineLRScheduler):
    """Cosine annealing with linear warm-up, built on timm's scheduler.

    Args:
        optimizer: The PyTorch optimizer.
        total_training_opt_steps: Total number of optimiser steps over the
            whole training run (= ``trainer.estimated_stepping_batches``).
        trainer: The ``pl.Trainer`` instance.  Used only to read
            ``trainer.max_epochs`` in order to derive steps-per-epoch.
        warmup_epochs: Number of warm-up epochs (converted to steps
            internally).
        min_lr: Minimum LR at the end of cosine decay.
        warmup_lr_init: Starting LR during warm-up (typically 1e-6).
        t_in_epochs: If ``True``, interpret ``t_initial`` / ``warmup_t``
            as epoch counts.  Defaults to ``False`` (step-based).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_training_opt_steps: int,
        trainer,
        warmup_epochs: int,
        min_lr: float,
        warmup_lr_init: float,
        t_in_epochs: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        # Use trainer locally only to derive steps_per_epoch — never store it
        # as an instance attribute: PL serialises scheduler.__dict__ when saving
        # checkpoints, and the Trainer holds DataLoader workers with
        # non-picklable _thread.lock objects.
        steps_per_epoch = max(
            total_training_opt_steps // trainer.max_epochs, 1
        )
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_training_opt_steps

        super().__init__(
            optimizer=self.optimizer,
            t_initial=self.total_steps,
            lr_min=self.min_lr,
            warmup_lr_init=self.warmup_lr_init,
            warmup_t=self.warmup_steps,
            t_in_epochs=self.t_in_epochs,
        )
