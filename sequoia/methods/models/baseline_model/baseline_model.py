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
from sequoia.common.hparams import uniform, log_uniform, categorical
from sequoia.settings import Actions, Observations, Rewards
from sequoia.settings.assumptions.incremental import IncrementalSetting
SettingType = TypeVar("SettingType", bound=IncrementalSetting)
from sequoia.methods.models.simple_convnet import SimpleConvNet

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
        # NOTE: All the fields below were just copied from the BaseHParams class, just
        # to improve visibility a bit.

        # Class variables that hold the available optimizers and encoders.
        # NOTE: These don't get parsed from the command-line.
        available_optimizers: ClassVar[Dict[str, Type[Optimizer]]] = {
            "sgd": optim.SGD,
            "adam": optim.Adam,
            "rmsprop": optim.RMSprop,
        }
        
        # Which optimizer to use.
        optimizer: Type[Optimizer] = categorical(available_optimizers, default=optim.Adam)
        
        available_encoders: ClassVar[Dict[str, Type[nn.Module]]] = {
            "vgg16": tv_models.vgg16,
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
            "resnet101": tv_models.resnet101,
            "resnet152": tv_models.resnet152,
            "alexnet": tv_models.alexnet,
            "densenet": tv_models.densenet161,
            # TODO: Add the self-supervised pl modules here!
            "simple_convnet": SimpleConvNet,
        }
        # Which encoder to use.
        encoder: Type[nn.Module] = choice(
            available_encoders,
            default=SimpleConvNet,
            # # TODO: Only considering these two for now when performing an HPO sweep.
            # probabilities={"resnet18": 0., "simple_convnet": 1.0},
        )

        # Learning rate of the optimizer.
        learning_rate: float = log_uniform(1e-6, 1e-2, default=1e-3)
        # L2 regularization term for the model weights.
        weight_decay: float = log_uniform(1e-12, 1e-3, default=1e-6)
        
        # Batch size to use during training and evaluation.
        batch_size: Optional[int] = None

        # Number of hidden units (before the output head).
        # When left to None (default), the hidden size from the pretrained
        # encoder model will be used. When set to an integer value, an
        # additional Linear layer will be placed between the outputs of the
        # encoder in order to map from the pretrained encoder's output size H_e
        # to this new hidden size `new_hidden_size`.
        new_hidden_size: Optional[int] = None
        # Retrain the encoder from scratch.
        train_from_scratch: bool = False
        # Wether we should keep the weights of the pretrained encoder frozen.
        freeze_pretrained_encoder_weights: bool = False

        # Hyper-parameters of the output head.
        output_head: OutputHead.HParams = mutable_field(OutputHead.HParams)

        # Wether the output head should be detached from the representations.
        # In other words, if the gradients from the downstream task should be
        # allowed to affect the representations.
        detach_output_head: bool = False

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

        # Upgrade the type of hparams for the output head, based on the setting.
        output_head_type = self.output_head_type(setting)
        if not isinstance(self.hp.output_head, output_head_type.HParams):
            self.hp.output_head = self.hp.output_head.upgrade(target_type=output_head_type.HParams)
        
        self.output_head: OutputHead = self.create_output_head(setting, task_id=None)

        # Dictionary of auxiliary tasks.
        self.tasks: Dict[str, AuxiliaryTask] = self.create_auxiliary_tasks()

        for task_name, task in self.tasks.items():
            logger.debug("Auxiliary tasks:")
            assert isinstance(task, AuxiliaryTask), f"Task {task} should be a subclass of {AuxiliaryTask}."
            if task.coefficient != 0:
                logger.debug(f"\t {task_name}: {task.coefficient}")
                logger.info(f"enabling the '{task_name}' auxiliary task (coefficient of {task.coefficient})")
                task.enable()

    @auto_move_data
    def forward(self, observations: Setting.Observations) -> ForwardPass:  # type: ignore
        """Forward pass of the model.
        
        For the given observations, creates a `ForwardPass`, a dict-like object which
        will hold the observations, the representations and the output head predictions.
        
        Parameters
        ----------
        observations : Setting.Observations
            Observations from one of the environments of a Setting.

        Returns
        -------
        ForwardPass
            A dict-like object which holds the observations, representations, and output
            head predictions (actions). See the `ForwardPass` class for more info.
        """
        # NOTE: Implementation is mostly in `base_model.py`.
        return super().forward(observations)

    def create_output_head(self, setting: Setting, task_id: Optional[int]) -> OutputHead:
        """Create an output head for the current action and reward spaces.
        
        NOTE: This assumes that the input, action and reward spaces don't change
        between tasks.

        Parameters
        ----------
        setting : Setting
            Current Setting. This is the same as `self.setting`, but provided because at
            some point the idea was to use a singledispatchmethod to choose which kind
            of output head to create based on the type of Setting.
        task_id : Optional[int]
            ID of the task associated with this new output head. Can be `None`, which is
            interpreted as saying that either that task labels aren't available, or that
            this output head will be used for all tasks. 

        Returns
        -------
        OutputHead
            The new output head for the given task.
        """
        # NOTE: Actual implementation is in `base_model.py`. This is added here just for
        # convenience when extending the baseline model.
        return super().create_output_head(setting, task_id=task_id)

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
