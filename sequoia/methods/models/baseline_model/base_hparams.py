from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, ClassVar

import torchvision.models as tv_models
from simple_parsing import mutable_field, choice
from torch import nn, optim
from torch.optim.optimizer import Optimizer  # type: ignore

from sequoia.methods.models.output_heads import OutputHead
from sequoia.utils import Parseable, Serializable
from sequoia.utils.pretrained_utils import get_pretrained_encoder
from sequoia.common.hparams import HyperParameters, uniform, log_uniform, categorical

from ..simple_convnet import SimpleConvNet

available_optimizers: Dict[str, Type[Optimizer]] = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
}
available_encoders: Dict[str, Type[nn.Module]] = {
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


@dataclass
class BaseHParams(HyperParameters):
    """ Set of 'base' Hyperparameters for the 'base' LightningModule. """
    # Class variable versions of the above dicts, for easier subclassing.
    # NOTE: These don't get parsed from the command-line.
    available_optimizers: ClassVar[Dict[str, Type[Optimizer]]] = available_optimizers.copy()
    available_encoders: ClassVar[Dict[str, Type[nn.Module]]] = available_encoders.copy()

    # Learning rate of the optimizer.
    learning_rate: float = log_uniform(1e-6, 1e-2, default=1e-3)
    # L2 regularization term for the model weights.
    weight_decay: float = log_uniform(1e-12, 1e-3, default=1e-6)
    # Which optimizer to use.
    optimizer: Type[Optimizer] = categorical(available_optimizers, default=optim.Adam)
    # Use an encoder architecture from the torchvision.models package.
    encoder: Type[nn.Module] = categorical(
        available_encoders,
        default=tv_models.resnet18,
        # TODO: Only using these two by default when performing a sweep.
        probabilities={"resnet18": 0.5, "simple_convnet": 0.5},
    )
    
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

    # Settings for the output head.
    # TODO: This could be overwritten in a subclass to do classification or
    # regression or RL, etc.
    output_head: OutputHead.HParams = mutable_field(OutputHead.HParams)

    # Wether the output head should be detached from the representations.
    # In other words, if the gradients from the downstream task should be
    # allowed to affect the representations.
    detach_output_head: bool = False

    def __post_init__(self):
        """Use this to initialize (or fix) any fields parsed from the
        command-line.
        """
        super().__post_init__()

    def make_optimizer(self, *args, **kwargs) -> Optimizer:
        """ Creates the Optimizer object from the options. """
        optimizer_class = self.optimizer
        options = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        options.update(kwargs)
        return optimizer_class(*args, **options)

    @property
    def encoder_model(self) -> Type[nn.Module]:
        return self.encoder
    
    def make_encoder(self, encoder_name: str = None) -> Tuple[nn.Module, int]:
        """Creates an Encoder model and returns the resulting hidden size.

        Returns:
            Tuple[nn.Module, int]: the encoder and the hidden size.
        """
        if encoder_name and encoder_name not in self.available_encoders:
            raise KeyError(
                f"No encoder with name {encoder_name} found! "
                f"(available encoders: {list(self.available_encoders.keys())}.")
            encoder_model = self.available_encoders[encoder_name]
        else:
            encoder_model = self.encoder
        encoder, hidden_size = get_pretrained_encoder(
            encoder_model=encoder_model,
            pretrained=not self.train_from_scratch,
            freeze_pretrained_weights=self.freeze_pretrained_encoder_weights,
            new_hidden_size=self.new_hidden_size,
        )
        return encoder, hidden_size

