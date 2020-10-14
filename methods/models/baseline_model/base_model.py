"""Base class for the Model to be used as part of a Method.

This is meant

TODO: There is a bunch of work to be done here.
"""
import gym
import dataclasses
import itertools
import functools
from abc import ABC
from dataclasses import dataclass
from collections import abc as collections_abc
from typing import (Any, ClassVar, Dict, Generic, List, NamedTuple, Optional,
                    Sequence, Tuple, Type, TypeVar, Union)

import torch
from gym import spaces
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.core.lightning import ModelSummary, log
from simple_parsing import list_field
from torch import nn, Tensor

from common.config import Config
from common.loss import Loss
from common.transforms import SplitBatch, Transforms
from common.batch import Batch
from settings.base.setting import Setting, SettingType, Observations, Actions, Rewards
from utils.logging_utils import get_logger
from settings import Environment, Observations, Actions, Rewards

from .base_hparams import BaseHParams
from ..output_heads import OutputHead, RegressionHead, ClassificationHead
from ..forward_pass import ForwardPass


logger = get_logger(__file__)

class BaseModel(LightningModule, Generic[SettingType]):
    """ Base model LightningModule (nn.Module extended by pytorch-lightning)
    
    WIP: (@lebrice): Trying to tidy up the hierarchy of the different kinds of
    models a little bit. 
    
    This model splits the learning task into a representation-learning problem
    and a downstream task (output head) applied on top of it.   

    The most important method to understand is the `get_loss` method, which
    is used by the [train/val/test]_step methods which are called by
    pytorch-lightning.
    """
    @dataclass
    class HParams(BaseHParams):
        """ HParams of the Model. """

    def __init__(self, setting: SettingType, hparams: HParams, config: Config):
        super().__init__()
        self.setting: SettingType = setting
        self.hp = hparams
        
        self.Observations: Type[Observations] = setting.Observations
        self.Actions: Type[Actions] = setting.Actions
        self.Rewards: Type[Rewards] = setting.Rewards
        
        self.observation_space: gym.Space = setting.observation_space
        self.action_space: gym.Space = setting.action_space
        self.reward_space: gym.Space = setting.reward_space
        
        self.input_shape  = self.observation_space[0].shape
        self.reward_shape = self.reward_space.shape
        
        self.split_batch_transform = SplitBatch(observation_type=self.Observations,
                                                reward_type=self.Rewards)
        self.config: Config = config
        # TODO: Decided to Not set this property, so the trainer doesn't
        # fallback to using it instead of the passed datamodules/dataloaders.
        # self.datamodule: LightningDataModule = setting

        # TODO: Debug this, seemed to be causing issues because it was trying to
        # save the 'setting' argument, which contained some things that weren't
        # pickleable.
        all_params_dict = {
            "hparams": hparams.to_dict(),
            "config": config.to_dict(),
        }
        self.save_hyperparameters(all_params_dict)

        # (Testing) Setting this attribute is supposed to help with ddp/etc
        # training in pytorch-lightning. Not 100% sure.
        # self.example_input_array = torch.rand(self.batch_size, *self.input_shape)
        
        # Create the encoder and the output head. 
        self.encoder, self.hidden_size = self.hp.make_encoder()
        self.output_head = self.create_output_head()

    @auto_move_data
    def forward(self, input_batch: Any) -> ForwardPass:
        """ Forward pass of the Model.
        
        Returns a ForwardPass object (or a dict)
        """
        observations: Observations = self.Observations.from_inputs(input_batch)
        
        # Encode the observation to get representations.
        representations = self.encode(observations)
        # Pass the observations and representations to the output head to get
        # the 'action' (prediction).
        actions = self.get_actions(observations, representations)

        forward_pass = ForwardPass(
            observations=observations,
            representations=representations,
            actions=actions,
        )
        return forward_pass

    def encode(self, observations: Observations) -> Tensor:
        """Encodes a batch of samples `x` into a hidden vector.

        Args:
            observations (Union[Tensor, Observation]): Tensor of Observation
            containing a batch of samples (before preprocess_observations).

        Returns:
            Tensor: The hidden vector / embedding for that sample, with size
                [<batch_size>, `self.hp.hidden_size`].
        """
        assert isinstance(observations, self.Observations)
        # If there's any additional 'input preprocessing' to do, do it here.
        # NOTE (@lebrice): This is currently done this way so that we don't have
        # to pass transforms to the settings from the method side.
        """
        TODOS:
        - Mark the transforms fields on the Setting as ClassVars.
        - Add those fields on the Method/models' HParam!                        
        """
        preprocessed_observations = self.preprocess_observations(observations)
        # Here in this base model the encoder only takes the 'x' from the observations.
        h_x = self.encoder(preprocessed_observations.x)
        if isinstance(h_x, list) and len(h_x) == 1:
            # Some pretrained encoders sometimes give back a list with one tensor. (?)
            h_x = h_x[0]
        return h_x

    def get_actions(self, observations: Observations, representations: Tensor) -> Actions:
        """ Pass the required inputs to the output head and get predictions.
        
        NOTE: This method is basically just here so we can customize what we
        pass to the output head, or what we take from it, similar to the
        `encode` method.
        """
        if self.hp.detach_output_head:
            representations = representations.detach()

        actions = self.output_head(
            observations=observations,
            representations=representations
        )
        assert isinstance(actions, self.Actions)
        return actions

    def create_output_head(self) -> OutputHead:
        """ Create an output head for the current action space. """

        if isinstance(self.action_space, spaces.Discrete):
            self.output_shape = (self.action_space.n,)
            return ClassificationHead(
                input_size=self.hidden_size,
                action_space=self.action_space,
                reward_space=self.reward_space,
            )

        if isinstance(self.action_space, spaces.Box):
            # Regression problem
            self.output_shape = self.action_space.shape
            return RegressionHead(
                input_size=self.hidden_size,
                action_space=self.action_space,
                reward_space=self.reward_space,
            )

        raise NotImplementedError(
            f"No output head available for action space {self.action_space}"
        )

    def training_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      *args,
                      **kwargs):
        return self.shared_step(
            batch,
            *args,
            environment=self.setting.train_env,
            loss_name="train",
            **kwargs
        )

    def validation_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      *args,
                      **kwargs):
        return self.shared_step(
            batch,
            *args,
            environment=self.setting.val_env,
            loss_name="val",
            **kwargs
        )

    def test_step(self,
                      batch: Tuple[Observations, Optional[Rewards]],
                      *args,
                      **kwargs):
        return self.shared_step(
            batch,
            *args,
            environment=self.setting.test_env,
            loss_name="test",
            **kwargs
        )

    def shared_step(self,
                    batch: Tuple[Observations, Rewards],
                    batch_idx: int,
                    environment: Environment,
                    loss_name: str,
                    dataloader_idx: int = None,
                    optimizer_idx: int = None) -> Dict:
        """
        This is the shared step for this 'example' LightningModule. 
        Feel free to customize/change it if you want!
        """
        if dataloader_idx is not None:
            assert isinstance(dataloader_idx, int)
            loss_name += f"/{dataloader_idx}"

        observations, rewards = self.split_batch(batch)

        # Get the forward pass results, containing:
        # - "observation": the augmented/transformed/processed observation.
        # - "representations": the representations for the observations.
        # - "actions": The actions (predictions)
        forward_pass: ForwardPass = self(observations)
        
        # get the actions from the forward pass:
        actions = forward_pass.actions
        
        if rewards is None:
            assert False, "not using this API atm."
            # Get the reward from the environment (the dataloader).
            rewards = environment.send(actions)
            assert rewards is not None

        loss: Loss = self.get_loss(forward_pass, rewards, loss_name=loss_name)
        return {
            "loss": loss.loss,
            "log": loss.to_log_dict(),
            "progress_bar": loss.to_pbar_message(),
            "loss_object": loss,
        }
        # The above can also easily be retrieved like this:
        result = loss.to_pl_dict()
        return result
    
    @auto_move_data
    def split_batch(self, batch: Any) -> Tuple[Observations, Rewards]:
        """ Splits the batch into the observations and the rewards. 
        
        Uses the types defined on the setting that this model is being applied
        on (which were copied to `self.Observations` and `self.Actions`).
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            observations, rewards = batch
            if (isinstance(observations, self.Observations) and
                isinstance(rewards, self.Rewards)):
                return observations, rewards
        return self.split_batch_transform(batch)

    def get_loss(self, forward_pass: Dict[str, Batch], rewards: Rewards = None, loss_name: str = "") -> Loss:
        """Returns a Loss object containing the total loss and metrics. 

        Args:
            x (Tensor): The input examples.
            y (Tensor, optional): The associated labels. Defaults to None.
            name (str, optional): Name to give to the resulting loss object. Defaults to "".

        Returns:
            Loss: An object containing everything needed for logging/progressbar/metrics/etc.
        """
        assert loss_name
        # Create an 'empty' Loss object with the given name, so that we always
        # return a Loss object, even when `y` is None and we can't the loss from
        # the output_head.
        total_loss = Loss(name=loss_name)
        if rewards:
            assert rewards.y is not None
            # So far we only use 'y' from the rewards in the output head.
            supervised_loss = self.output_head.get_loss(forward_pass, y=rewards.y)
            total_loss += supervised_loss
        return total_loss

    def preprocess_observations(self, observations: Observations) -> Observations:
        return observations
    
    def preprocess_rewards(self, reward: Rewards) -> Rewards:
        return reward

    def configure_optimizers(self):
        return self.hp.make_optimizer(self.parameters())

    @property
    def batch_size(self) -> int:
        return self.hp.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.hp.batch_size = value 
    
    @property
    def learning_rate(self) -> float:
        return self.hp.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.hp.learning_rate = value

    def on_task_switch(self, task_id: int, training: bool = False) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int): the Id of the task.
            training (bool): Wether we are currently training or valid/testing.
        """

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        model_summary = ModelSummary(self, mode=mode)
        log.debug('\n' + str(model_summary))
        return model_summary
