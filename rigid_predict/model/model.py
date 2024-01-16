from typing import *

from transformers.optimization import get_linear_schedule_with_warmup

import torch
import pytorch_lightning as pl

from rigid_predict.model.module import RigidPacking
import time
from rigid_predict.model import losses

class RigidPacking_Lighting(RigidPacking, pl.LightningModule):

    def __init__(
            self,
            lr: float = 5e-5,
            l2: float = 0.0,
            l1: float = 0.0,
            circle_reg: float = 0.0,
            epochs: int = 10,
            steps_per_epoch: int = 250,  # Dummy value
            # diffusion_fraction: float = 0.7,
            lr_scheduler: str = None,
            **kwargs,

    ):
        """Feed args to BertForDiffusionBase and then feed the rest into"""
        RigidPacking.__init__(self, **kwargs)

        self.learning_rate = lr
        self.l1_lambda = l1
        self.l2_lambda = l2
        self.circle_lambda = circle_reg
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr_scheduler = lr_scheduler


    def training_step(self, batch, batch_idx):
        """
        Training step, runs once per batch
        """
        outputs = self.forward(batch)
        avg_loss = losses.fape_loss(outputs, batch)
        
        self.log("train_loss", avg_loss, on_epoch=True, batch_size=batch.batch_size, rank_zero_only=True)

        return avg_loss

    def training_epoch_end(self, outputs) -> None:
        #  tracemalloc.start()
        """Log the average training loss over the epoch"""
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end:({t_delta:.2f} seconds)"
        )
        # Increment counter and timers
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Validation step
        """
        with torch.no_grad():
            out = self.forward(batch)
            avg_loss = losses.fape_loss(out, batch)

            self.log("val_loss", avg_loss, on_epoch=True, batch_size=batch.batch_size, rank_zero_only=True)
        return avg_loss

    """
    def on_after_backward(self) -> None:
        print("on_after_backward enter")
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)
        print("on_after_backward exit")
    """
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                # https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
                # Transformers typically do well with linear warmup
                warmup_steps = int(self.epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",  # Call after 1 epoch
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval
