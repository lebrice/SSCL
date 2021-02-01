""" Example/Template of a Model to be used as part of a Method.

You can use this as a base class when creating your own models, or you can
start from scratch, whatever you like best.
"""
from dataclasses import dataclass
from typing import *

import gym
import numpy as np
import pytorch_lightning as pl
import torch
from gym import Space, spaces
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from simple_parsing import Serializable, choice, mutable_field
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from sequoia.settings import Setting, ActiveSetting, PassiveSetting, Environment
from sequoia.common.config import Config
from sequoia.common.loss import Loss
from sequoia.methods.aux_tasks.auxiliary_task import AuxiliaryTask
from sequoia.methods.models.output_heads import (ActorCriticHead,
                                                 ClassificationHead,
                                                 OutputHead, PolicyHead,
                                                 RegressionHead)
from sequoia.settings import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__file__)


# WIP (@lebrice): Playing around with this idea, to try and maybe use the idea
# of creating typed objects for the 'Observation', the 'Action' and the 'Reward'
# for each kind of model.
from sequoia.settings import Actions, Observations, Rewards
from sequoia.settings.assumptions.incremental import IncrementalSetting
SettingType = TypeVar("SettingType", bound=IncrementalSetting)

from .base_model import ForwardPass
from .class_incremental_model import ClassIncrementalModel
from .self_supervised_model import SelfSupervisedModel
from .semi_supervised_model import SemiSupervisedModel


class BaselineModel(SemiSupervisedModel,
                    ClassIncrementalModel,
                    SelfSupervisedModel,
                    Generic[SettingType]):
    """ Base model LightningModule (nn.Module extended by pytorch-lightning)
    
    This model splits the learning task into a representation-learning problem
    and a downstream task (output head) applied on top of it.   

    The most important method to understand is the `get_loss` method, which
    is used by the [train/val/test]_step methods which are called by
    pytorch-lightning.
    """

    @dataclass
    class HParams(SemiSupervisedModel.HParams,
                  SelfSupervisedModel.HParams,
                  ClassIncrementalModel.HParams):
        """ HParams of the Model. """
        # NOTE: The different hyper-parameters can be found as fields in the
        # base classes, but most of them are defined in base_hparams.py.
        pass

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__(setting=setting, hparams=hparams, config=config)
        
        self.save_hyperparameters({
            "hparams": self.hp.to_dict(),
            "config": self.config.to_dict(),
        })
        
        logger.debug(f"setting of type {type(self.setting)}")
        logger.debug(f"Observation space: {self.observation_space}")
        logger.debug(f"Action/Output space: {self.action_space}")
        logger.debug(f"Reward/Label space: {self.reward_space}")

        if self.config.debug and self.config.verbose:
            logger.debug("Config:")
            logger.debug(self.config.dumps(indent="\t"))
            logger.debug("Hparams:")
            logger.debug(self.hp.dumps(indent="\t"))

        if self.trainer:
            OutputHead.base_model_optimizer = self.optimizers()

        # Upgrade the type of hparams for the output head, based on the setting.
        self.hp.output_head = self.hp.output_head.upgrade(target_type=self.output_head_type(setting).HParams)
        
        self.output_head: OutputHead = self.create_output_head(setting)

        self.tasks: Dict[str, AuxiliaryTask] = self.create_auxiliary_tasks()

        for task_name, task in self.tasks.items():
            logger.debug("Auxiliary tasks:")
            assert isinstance(task, AuxiliaryTask), f"Task {task} should be a subclass of {AuxiliaryTask}."
            if task.coefficient != 0:
                logger.debug(f"\t {task_name}: {task.coefficient}")
                logger.info(f"enabling the '{task_name}' auxiliary task (coefficient of {task.coefficient})")
                task.enable()

    # @auto_move_data
    def forward(self, observations: IncrementalSetting.Observations) -> ForwardPass:  # type: ignore
        # NOTE: Implementation is mostly in `base_model.py`.
        return super().forward(observations)

    def create_output_head(self, setting: Setting, add_to_optimizer: bool = None) -> OutputHead:
        """Create an output head for the current setting.
        
        NOTE: This assumes that the input, action and reward spaces don't change
        between tasks.
        
        Parameters
        ----------
        add_to_optimizer : bool, optional
            Wether to add the parameters of the new output head to the optimizer
            of the model. Defaults to None, in which case we add the output head
            parameters as long as it doesn't have an `optimizer` attribute of
            its own.

        Returns
        -------
        OutputHead
            The new output head.
        """
        # NOTE: Implementation is in `base_model.py`.
        return super().create_output_head(setting, add_to_optimizer=add_to_optimizer)

    def output_head_type(self, setting: SettingType) -> Type[OutputHead]:
        """ Return the type of output head we should use in a given setting.
        """
        # NOTE: Implementation is in `base_model.py`.
        return super().output_head_type(setting)

    def training_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      batch_idx: int,
                      *args, **kwargs):
        step_result = self.shared_step(
            batch,
            batch_idx,
            environment=self.setting.train_env,
            loss_name="train",
            *args,
            **kwargs
        )
        loss: Tensor = step_result["loss"]
        loss_object: Loss = step_result["loss_object"]
        
        if not isinstance(loss, Tensor) or not loss.requires_grad:
            # NOTE: There might be no loss at some steps, because for instance
            # we haven't reached the end of an episode in an RL setting.
            return None

        # NOTE In RL, we can only update the model's weights on steps where the output
        # head has as loss, because the output head has buffers of tensors whose grads
        # would become invalidated if we performed the optimizer step.
        if loss.requires_grad and not self.trainer.train_loop.automatic_optimization:
            output_head_loss = loss_object.losses.get(self.output_head.name)            
            update_model = output_head_loss is not None and output_head_loss.requires_grad
            optimizer = self.optimizers()
            self.manual_backward(loss, optimizer, retain_graph=not update_model)
            if update_model:
                optimizer.step()
                optimizer.zero_grad()
        return step_result

    def validation_step(self,
                        batch: Tuple[Observations, Optional[Rewards]],
                        batch_idx: int,
                        *args,
                        **kwargs):
        return self.shared_step(
            batch,
            batch_idx,
            environment=self.setting.val_env,
            loss_name="val",
            *args,
            **kwargs,
        )

    def test_step(self,
                  batch: Tuple[Observations, Optional[Rewards]],
                  batch_idx: int,
                  *args,
                  **kwargs):
        return self.shared_step(
            batch,
            batch_idx,
            *args,
            environment=self.setting.test_env,
            loss_name="test",
            **kwargs,
        )

    def shared_step(self,
                    batch: Tuple[Observations, Optional[Rewards]],
                    batch_idx: int,
                    environment: Environment,
                    loss_name: str,
                    dataloader_idx: int = None,
                    optimizer_idx: int = None) -> Dict:
        results = super().shared_step(batch, batch_idx, environment, loss_name, dataloader_idx=dataloader_idx, optimizer_idx=optimizer_idx)
        loss_tensor = results["loss"]
        loss = results["loss_object"]

        if loss_tensor != 0.:
            for key, value in loss.to_pbar_message().items():
                assert not isinstance(value, (dict, str)), "shouldn't be nested at this point!"
                self.log(key, value, prog_bar=True)
                logger.debug(f"{key}: {value}")
            
            for key, value in loss.to_log_dict(verbose=self.config.verbose).items():
                assert not isinstance(value, (dict, str)), "shouldn't be nested at this point!"
                self.log(key, value, logger=True)
        return results
