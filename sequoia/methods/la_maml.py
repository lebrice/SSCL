import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type
import gym
import torch
from gym import spaces
from torch import Tensor, nn
from simple_parsing import ArgumentParser
sys.path.extend([".", ".."])
from sequoia.settings import Method, Setting
from sequoia.settings.passive.cl import ClassIncrementalSetting, DomainIncrementalSetting, TaskIncrementalSetting
from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    PassiveEnvironment,
    Results,
    Rewards,
)
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Type, Any
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tqdm
from torch import Tensor
from torchvision.models import ResNet
from wandb.wandb_run import Run
from sequoia.methods import register_method
from sequoia.utils import get_logger
import math
import os
import traceback
import ipdb
import random
from random import shuffle
import numpy as np
import ipdb
from torch.autograd import Variable
from scipy.stats import pearsonr
import datetime

class Learner(nn.Module):
    def __init__(self, config, args = None):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config
        self.tf_counter = 0
        self.args = args

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        self.names = []

        for i, (name, param, extra_name) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                if(args and self.args.xav_init):
                    w = nn.Parameter(torch.ones(*param[:4]))
                    b = nn.Parameter(torch.zeros(param[0]))
                    torch.nn.init.xavier_normal_(w.data)
                    b.data.normal_(0, math.sqrt(2)/math.sqrt(1+9*b.data.shape[0]))
                    self.vars.append(w)
                    self.vars.append(b)
                else:
                    w = nn.Parameter(torch.ones(*param[:4]))
                    # gain=1 according to cbfin's implementation
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    # [ch_out]
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # layer += 1
                if(args and self.args.xav_init):
                    w = nn.Parameter(torch.ones(*param))
                    # b = nn.Parameter(torch.zeros(param[0]))
                    torch.nn.init.xavier_normal_(w.data)
                    # b.data.normal_(0, math.sqrt(2)/math.sqrt(1+9*b.data.shape[0]))
                    self.vars.append(w)
                    # self.vars.append(b)
                else:
                    # [ch_out, ch_in]
                    w = nn.Parameter(torch.ones(*param))
                    # gain=1 according to cbfinn's implementation
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'cat':
                pass
            elif name is 'cat_start':
                pass
            elif name is "rep":
                pass
            elif name in ["residual3", "residual5", "in"]:
                pass
            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):

        info = ''

        for name, param, extra_name in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name is 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name is 'rep':
                tmp = 'rep'
                info += tmp + "\n"


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=False, feature=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        cat_var = False
        cat_list = []

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        try:

            for (name, param, extra_name) in self.config:
                # assert(name == "conv2d")
                if name == 'conv2d':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2

                    # print(name, param, '\tout:', x.shape)
                elif name == 'convt2d':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2


                elif name == 'linear':

                    # ipdb.set_trace()
                    if extra_name == 'cosine':
                        w = F.normalize(vars[idx])
                        x = F.normalize(x)
                        x = F.linear(x, w)
                        idx += 1
                    else:
                        w, b = vars[idx], vars[idx + 1]
                        x = F.linear(x, w, b)
                        idx += 2

                    if cat_var:
                        cat_list.append(x)

                elif name == 'rep':
                    # print('rep')
                    # print(x.shape)
                    if feature:
                        return x

                elif name == "cat_start":
                    cat_var = True
                    cat_list = []

                elif name == "cat":
                    cat_var = False
                    x = torch.cat(cat_list, dim=1)

                elif name == 'bn':
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    idx += 2
                    bn_idx += 2
                elif name == 'flatten':
                    # print('flatten')
                    # print(x.shape)

                    x = x.view(x.size(0), -1)

                elif name == 'reshape':
                    # [b, 8] => [b, 2, 2, 2]
                    x = x.view(x.size(0), *param)
                elif name == 'relu':
                    x = F.relu(x, inplace=param[0])
                elif name == 'leakyrelu':
                    x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    x = F.tanh(x)
                elif name == 'sigmoid':
                    x = torch.sigmoid(x)
                elif name == 'upsample':
                    x = F.upsample_nearest(x, scale_factor=param[0])
                elif name == 'max_pool2d':
                    x = F.max_pool2d(x, param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    x = F.avg_pool2d(x, param[0], param[1], param[2])

                else:
                    print(name)
                    raise NotImplementedError

        except:
            traceback.print_exc(file=sys.stdout)
            ipdb.set_trace()

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x


    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def define_task_lr_params(self, alpha_init=1e-3):
        # Setup learning parameters
        self.alpha_lr = nn.ParameterList([])

        self.lr_name = []
        for n, p in self.named_parameters():
            self.lr_name.append(n)

        for p in self.parameters():
            self.alpha_lr.append(nn.Parameter(alpha_init * torch.ones(p.shape, requires_grad=True)))

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

class BaseNet(torch.nn.Module):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 hparams):
        super(BaseNet, self).__init__()

        image_shape = observation_space[0].shape
        assert image_shape == (3, 28, 28)
        assert isinstance(action_space, spaces.Discrete)
        assert action_space == reward_space
        n_classes = action_space.n
        image_channels = image_shape[0]

        # define the model
        input_channels=1
        n_classes=action_space.n

        config = [
                    ('conv2d', [6, image_channels, 5, 5, 1, 0], ''),
                    ('relu', [True], ''),
                    ('max_pool2d',[2,None,0],''),


                    ('conv2d', [16, 6, 5, 5, 1, 0], ''),
                    ('relu', [True], ''),
                    ('max_pool2d',[2,None,0],''),

                    ('flatten', [], ''),
                    ('linear', [120,256], ''),
                    ('relu', [True], ''),
                    ('linear', [84,120], ''),
                    ('relu', [True], ''),
                    ('linear', [n_classes,84], '')

                ]

        self.net = Learner(config)

        # define the lr params


        self.net.define_task_lr_params(alpha_init = hparams.alpha_init)

        self.opt_wt = torch.optim.SGD(list(self.net.parameters()), lr=hparams.weight_outer_lr)
        self.opt_lr = torch.optim.SGD(list(self.net.alpha_lr.parameters()), lr=hparams.alpha_lr_outer_lr)

        self.epoch = 0

        # setup losses
        self.loss = torch.nn.CrossEntropyLoss()

        self.cuda = hparams.cuda
        if self.cuda:
            self.net = self.net.cuda()

    def zero_grads(self):
        self.opt_lr.zero_grad()
        self.opt_wt.zero_grad()
        self.net.zero_grad()
        self.net.alpha_lr.zero_grad()

class Net(BaseNet):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 reward_space: gym.Space,
                 hparams):
        super(Net, self).__init__(
                 observation_space,
                 action_space,
                 reward_space,
                 hparams)


    def forward(self, x, t=0):
        output = self.net.forward(x)
        return output

    def meta_loss(self, x, fast_weights, y, t):
        """
        differentiate the loss through the network updates wrt alpha
        """
        logits = self.net.forward(x, fast_weights)
        loss_q = self.loss(logits.squeeze(1), y)
        return loss_q, logits

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        logits = self.net.forward(x, fast_weights)
        loss = self.loss(logits, y)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = False #Hyperparameter
        grads = torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required)

        '''
        for i in range(len(grads)):
            torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)
        '''
        #add hyperparameter for gradient clipping!
        fast_weights = list(
                map(lambda p: p[1][0] - p[0] * nn.functional.relu(p[1][1]), zip(grads, zip(fast_weights, self.net.alpha_lr))))
        return fast_weights
    def shared_step(
        self, batch: Tuple[Observations, Optional[Rewards]], environment: Environment
    ) -> Tuple[Tensor, Dict]:
        """Shared step used for both training and validation.

        Parameters
        ----------
        batch : Tuple[Observations, Optional[Rewards]]
            Batch containing Observations, and optional Rewards. When the Rewards are
            None, it means that we'll need to provide the Environment with actions
            before we can get the Rewards (e.g. image labels) back.

            This happens for example when being applied in a Setting which cares about
            sample efficiency or training performance, for example.

        environment : Environment
            The environment we're currently interacting with. Used to provide the
            rewards when they aren't already part of the batch (as mentioned above).

        Returns
        -------
        Tuple[Tensor, Dict]
            The Loss tensor, and a dict of metrics to be logged.
        """
        # Since we're training on a Passive environment, we will get both observations
        # and rewards, unless we're being evaluated based on our training performance,
        # in which case we will need to send actions to the environments before we can
        # get the corresponding rewards (image labels).
        observations: Observations = batch[0]
        rewards: Optional[Rewards] = batch[1]
        # Get the predictions:
        logits = self.net(observations.x)
        y_pred = logits.argmax(-1)

        if rewards is None:
            # If the rewards in the batch is None, it means we're expected to give
            # actions before we can get rewards back from the environment.
            rewards = environment.send(Actions(y_pred))

        assert rewards is not None
        image_labels = rewards.y

        loss = self.loss(logits, image_labels)

        accuracy = (y_pred == image_labels).sum().float() / len(image_labels)
        metrics_dict = {"accuracy": accuracy.item()}
        return loss, metrics_dict

class Buffer(nn.Module):
    def __init__(
        self,
        capacity: int,
        input_shape: Tuple[int, ...],
        extra_buffers: Dict[str, Type[torch.Tensor]] = None,
        rng: np.random.RandomState = None,
    ):
        super().__init__()
        self.rng = rng or np.random.RandomState()

        bx = torch.zeros([capacity, *input_shape], dtype=torch.float)
        by = torch.zeros([capacity], dtype=torch.long)

        self.register_buffer("bx", bx)
        self.register_buffer("by", by)
        self.buffers = ["bx", "by"]

        extra_buffers = extra_buffers or {}
        for name, dtype in extra_buffers.items():
            tmp = dtype(capacity).fill_(0)
            self.register_buffer(f"b{name}", tmp)
            self.buffers += [f"b{name}"]

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full = 0
        # (@lebrice) args isn't defined here:
        # self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x: torch.arange(x.size(0)).to(x.device)
        self.shuffle = lambda x: x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[: self.current_index]

    @property
    def y(self):
        raise NotImplementedError("Can't make y one-hot, dont have n_classes.")
        return self.to_one_hot(self.by[: self.current_index])

    def add_reservoir(self, batch: Dict[str, Tensor]) -> None:
        n_elem = batch["x"].size(0)

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)

        if place_left:
            offset = min(place_left, n_elem)

            for name, data in batch.items():
                buffer = getattr(self, f"b{name}")
                if isinstance(data, Iterable):
                    buffer[self.current_index : self.current_index + offset].data.copy_(
                        data[:offset]
                    )
                else:
                    buffer[self.current_index : self.current_index + offset].fill_(data)

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == batch["x"].size(0):
                return

        x = batch["x"]
        self.place_left = False
        indices = (
            torch.FloatTensor(x.size(0) - place_left)
            .to(x.device)
            .uniform_(0, self.n_seen_so_far)
            .long()
        )
        valid_indices: Tensor = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero(as_tuple=False).squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        # perform overwrite op

        for name, data in batch.items():
            buffer = getattr(self, f"b{name}")
            if isinstance(data, Iterable):
                data = data[place_left:]
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data

    def sample(self, n_samples: int, exclude_task: int = None) -> Dict[str, Tensor]:
        buffers = {}
        if exclude_task is not None:
            assert hasattr(self, "bt")
            valid_indices = (self.bt != exclude_task).nonzero().squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[: self.current_index]

        bx = buffers["bx"]
        if bx.size(0) < n_samples:
            return buffers
        else:
            indices_np = self.rng.choice(bx.size(0), n_samples, replace=False)
            indices = torch.from_numpy(indices_np).to(self.bx.device)
            return {k[1:]: v[indices] for (k, v) in buffers.items()}

class LA_MAML(Method, target_setting=ClassIncrementalSetting):

    @dataclass
    class HParams:
        """ Hyper-parameters of the demo model. """
        # Learning rate of the optimizer.
        alpha_init: float=0.001
        weight_outer_lr: float=0.1
        alpha_lr_outer_lr: float=0.1
        cuda: int= 0



    def __init__(self, hparams: HParams):
        self.hparams: LA_MAML.HParams = hparams
        self.max_epochs: int = 1
        self.early_stop_patience: int = 2

        self.buffer_capacity: int = 200
        self.seed : int = None


        self.buffer: Optional[Buffer] = None
        self.task: int = 0
        self.rng = np.random.RandomState(self.seed)
        self.device = torch.device("cuda" if (hparams.cuda and torch.cuda.is_available()) else "cpu")
        #device for buffer should be on cpu!! the rest is cuda


    def configure(self, setting: ClassIncrementalSetting):
        """ Called before the method is applied on a setting (before training).

        You can use this to instantiate your model, for instance, since this is
        where you get access to the observation & action spaces.
        """
        self.model = Net(
            observation_space=setting.observation_space,
            action_space=setting.action_space,
            reward_space=setting.reward_space,
            hparams=self.hparams
        ).to(self.device)

        image_space: spaces.Box = setting.observation_space[0]
        # Create the buffer.
        if self.buffer_capacity:
            self.buffer = Buffer(
                capacity=self.buffer_capacity,
                input_shape=image_space.shape,
                extra_buffers={"t": torch.LongTensor},
                rng=self.rng,
            ).to(device=torch.device("cpu"))

    def fit(self, train_env: PassiveEnvironment, valid_env: PassiveEnvironment):
        # configure() will have been called by the setting before we get here.
        import tqdm
        from numpy import inf
        best_val_loss = inf
        best_epoch = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            # Training loop:
            with tqdm.tqdm(train_env) as train_pbar:
                train_pbar.set_description(f"Training Epoch {epoch}")
                for i, batch in enumerate(train_pbar):
                    #buffer
                    observations: Observations = batch[0]
                    rewards: Optional[Rewards] = batch[1]

                    #sample from previous tasks (implement to modulate size)
                    if self.buffer.n_seen_so_far!=0:
                        b_samples = self.buffer.sample(observations.x.size(0))
                        b_x=b_samples["x"]
                        b_y=b_samples["y"]
                        bm_x=torch.cat((b_x,observations.x),dim=0).to(self.device)
                        bm_y=torch.cat((b_y,rewards.y),dim=0).to(self.device)
                    else:
                        #nothing samples in buffer
                        bm_x=observations.x.to(self.device)
                        bm_y=rewards.y.to(self.device)
                    #initialize meta losses
                    meta_losses=[0 for _ in range(observations.x.size(0))]
                    #Inner loop
                    fast_weights=None
                    for k in range(observations.x.size(0)):
                        input_train=observations.x[k].unsqueeze(dim=0).to(self.device)
                        label_train=rewards.y[k].unsqueeze(dim=0).to(self.device)

                        fast_weights=self.model.inner_update(input_train,fast_weights,label_train,0)

                        meta_loss,logits=self.model.meta_loss(bm_x,fast_weights,bm_y,0)


                        meta_losses[k]+=meta_loss

                    self.model.zero_grads()
                    meta_loss=sum(meta_losses)/len(meta_losses)
                    meta_loss.backward()

                    #option to clip gradient
                    self.model.opt_lr.step()
                    self.model.opt_wt.step()

                    self.model.net.zero_grad()
                    self.model.net.alpha_lr.zero_grad()



                    self.buffer.add_reservoir({"x": observations.x.cpu(), "y": rewards.y.cpu(), "t": 0})


            # Validation loop:
            self.model.eval()
            torch.set_grad_enabled(False)
            with tqdm.tqdm(valid_env) as val_pbar:
                val_pbar.set_description(f"Validation Epoch {epoch}")
                epoch_val_loss = 0.

                for i, batch in enumerate(val_pbar):
                    batch_val_loss, metrics_dict = self.model.shared_step(batch, environment=valid_env)
                    epoch_val_loss += batch_val_loss
                    val_pbar.set_postfix(**metrics_dict, val_loss=epoch_val_loss)
            torch.set_grad_enabled(True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = valid_env
                best_epoch = epoch
            if epoch - best_epoch > self.early_stop_patience:
                print(f"Early stopping at epoch {i}.")
                break

    def get_actions(self, observations: Observations, action_space: gym.Space) -> Actions:
        """ Get a batch of predictions (aka actions) for these observations. """
        with torch.no_grad():
            logits = self.model(observations.x)
        # Get the predicted classes
        y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = ""):
        """Adds command-line arguments for this Method to an argument parser."""
        parser.add_arguments(cls.HParams, "hparams")

    @classmethod
    def from_argparse_args(cls, args, dest: str = ""):
        """Creates an instance of this Method from the parsed arguments."""
        hparams: cls.HParams = args.hparams
        return cls(hparams=hparams)

def demo():
    method = LA_MAML(hparams=LA_MAML.HParams())
    setting = ClassIncrementalSetting(dataset="fashionmnist")
    results = setting.apply(method)
    return results

if __name__ == "__main__":
    results=demo()
    print(results.summary())
