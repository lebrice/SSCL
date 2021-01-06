from dataclasses import dataclass
from typing import Dict, List, Union

import gym
import torch
from gym import spaces
from torch import Tensor, nn, LongTensor
from simple_parsing import list_field

from sequoia.common import Batch, ClassificationMetrics, Loss
from sequoia.common.layers import Flatten
from sequoia.settings import Observations, Actions, Rewards

from .output_head import OutputHead
from ..forward_pass import ForwardPass

@dataclass(frozen=True)
class ClassificationOutput(Actions):
    """ Typed dict-like class that represents the 'forward pass'/output of a
    classification head, which correspond to the 'actions' to be sent to the
    environment, in the general formulation.
    """
    y_pred: Union[LongTensor, Tensor]
    logits: Tensor

    @property
    def action(self) -> LongTensor:
        return self.y_pred
    
    @property
    def y_pred_log_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.logits[:, self.y_pred]

    @property
    def y_pred_prob(self) -> Tensor:
        """ returns the log probabilities for the chosen actions/predictions. """
        return self.probabilities[self.y_pred]

    @property
    def probabilities(self) -> Tensor:
        """ Returns the normalized probabilies for each class, i.e. the
        softmax-ed version of `self.logits`.
        """
        return self.logits.softmax(-1)


class ClassificationHead(OutputHead):

    @dataclass
    class HParams(OutputHead.HParams):
        hidden_layers: int = 1
        hidden_neurons: List[int] = list_field(64)

    def __init__(self,
                 input_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space = None,
                 hparams: "OutputHead.HParams" = None,
                 name: str = "classification"):
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        assert isinstance(action_space, spaces.Discrete)
        output_size = action_space.n
        self.dense = self.make_dense_network(
            in_features=self.input_size,
            hidden_neurons=self.hparams.hidden_neurons,
            out_features=output_size,
            activation=self.hparams.activation,
        )
        # if output_size == 2:
        #     # TODO: Should we be using this loss instead?
        #     self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, observations: Observations, representations: Tensor) -> ClassificationOutput:
        # TODO: This should probably take in a dict and return a dict, or something like that?
        # TODO: We should maybe convert this to also return a dict instead
        # of a Tensor, just to be consistent with everything else. This could
        # also maybe help with having multiple different output heads, each
        # having a different name and giving back a dictionary of their own
        # forward pass tensors (if needed) and predictions?
        logits = self.dense(representations)
        y_pred = logits.argmax(dim=-1)
        return ClassificationOutput(
            logits=logits,
            y_pred=y_pred,
        )

    def get_loss(self, forward_pass: ForwardPass, actions: ClassificationOutput, rewards: Rewards) -> Loss:
        logits: Tensor = actions.logits
        y_pred: Tensor = actions.y_pred
        y: Tensor = rewards.y

        n_classes = logits.shape[-1]
        # Could remove these: just used for debugging.
        assert len(y.shape) == 1, y.shape
        assert not torch.is_floating_point(y), y.dtype
        assert 0 <= y.min(), y
        assert y.max() < n_classes, y

        loss = self.loss_fn(logits, y)
        
        assert loss.shape == ()
        metrics = ClassificationMetrics(y_pred=logits, y=y)
        
        assert self.name, "Output Heads should have a name!"
        loss_object = Loss(
            name=self.name,
            loss=loss,
            # NOTE: we're passing the tensors to the Loss object because we let
            # it create the Metrics for us automatically.
            metrics={self.name: metrics},
        )
        return loss_object
