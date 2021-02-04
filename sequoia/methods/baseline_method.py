""" Defines a Method, which is a "solution" for a given "problem" (a Setting).

The Method could be whatever you want, really. For the 'baselines' we have here,
we use pytorch-lightning, and a few little utility classes such as `Metrics` and
`Loss`, which are basically just like dicts/objects, with some cool other
methods.
"""
import json
import operator
import warnings
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from simple_parsing import mutable_field

from sequoia.common import Config, TrainerConfig
from sequoia.common.config import WandbLoggerConfig
from sequoia.settings import ActiveSetting, PassiveSetting
from sequoia.settings.active.continual import ContinualRLSetting
from sequoia.settings.assumptions.incremental import IncrementalSetting
from sequoia.settings.base import Method
from sequoia.settings.base.environment import Environment
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.settings.base.results import Results
from sequoia.settings.base.setting import Setting, SettingType
from sequoia.utils import Parseable, Serializable, compute_identity, get_logger
from sequoia.methods import register_method

from .models import BaselineModel, ForwardPass

logger = get_logger(__file__)


@register_method
@dataclass
class BaselineMethod(Method, Serializable, Parseable, target_setting=Setting):
    """ Versatile Baseline method which targets all settings.

    Uses pytorch-lightning's Trainer for training and LightningModule as model.

    Uses a [BaselineModel](methods/models/baseline_model/baseline_model.py), which
    can be used for:
    - Self-Supervised training with modular auxiliary tasks;
    - Semi-Supervised training on partially labeled batches;
    - Multi-Head prediction (e.g. in task-incremental scenario);
    """

    # NOTE: these two fields are also used to create the command-line arguments.
    # HyperParameters of the method.
    hparams: BaselineModel.HParams = mutable_field(BaselineModel.HParams)
    # Configuration options.
    config: Config = mutable_field(Config)
    # Options for the Trainer object.
    trainer_options: TrainerConfig = mutable_field(TrainerConfig)

    def __init__(
        self,
        hparams: BaselineModel.HParams = None,
        config: Config = None,
        trainer_options: TrainerConfig = None,
        **kwargs,
    ):
        """ Creates a new BaselineMethod, using the provided configuration options.

        Parameters
        ----------
        hparams : BaselineModel.HParams, optional
            Hyper-parameters of the BaselineModel used by this Method. Defaults to None.

        config : Config, optional
            Configuration dataclass with options like log_dir, device, etc. Defaults to
            None.

        trainer_options : TrainerConfig, optional
            Dataclass which holds all the options for creating the `pl.Trainer` which
            will be used for training. Defaults to None.

        **kwargs :
            If any of the above arguments are left as `None`, then they will be created
            using any appropriate value from `kwargs`, if present.

        ## Examples:
        ```
        method = BaselineMethod(hparams=BaselineModel.HParams(learning_rate=0.01))
        method = BaselineMethod(learning_rate=0.01) # Same as above

        method = BaselineMethod(config=Config(debug=True))
        method = BaselineMethod(debug=True) # Same as above

        method = BaselineMethod(hparams=BaselineModel.HParams(learning_rate=0.01),
                                config=Config(debug=True))
        method = BaselineMethod(learning_rate=0.01, debug=True) # Same as above
        ```
        """
        # TODO: When creating a Method from a script, like `BaselineMethod()`,
        # should we expect the hparams to be passed? Should we create them from
        # the **kwargs? Should we parse them from the command-line?

        # Option 2: Try to use the keyword arguments to create the hparams,
        # config and trainer options.
        if kwargs:
            logger.info(
                f"using keyword arguments {kwargs} to populate the corresponding "
                f"values in the hparams, config and trainer_options."
            )
            self.hparams = hparams or BaselineModel.HParams.from_dict(
                kwargs, drop_extra_fields=True
            )
            self.config = config or Config.from_dict(kwargs, drop_extra_fields=True)
            self.trainer_options = trainer_options or TrainerConfig.from_dict(
                kwargs, drop_extra_fields=True
            )

        elif self._argv:
            # Since the method was parsed from the command-line, parse those as
            # well from the argv that were used to create the Method.
            # Option 3: Parse them from the command-line.
            # assert not kwargs, "Don't pass any extra kwargs to the constructor!"
            self.hparams = hparams or BaselineModel.HParams.from_args(
                self._argv, strict=False
            )
            self.config = config or Config.from_args(self._argv, strict=False)
            self.trainer_options = trainer_options or TrainerConfig.from_args(
                self._argv, strict=False
            )

        else:
            # Option 1: Use the default values:
            self.hparams = hparams or BaselineModel.HParams()
            self.config = config or Config()
            self.trainer_options = trainer_options or TrainerConfig()
        assert self.hparams
        assert self.config
        assert self.trainer_options

        if self.config.debug:
            # Disable wandb logging if debug is True.
            self.trainer_options.no_wandb = True

        # The model and Trainer objects will be created in `self.configure`.
        # NOTE: This right here doesn't create the fields, it just gives some
        # type information for static type checking.
        self.trainer: Trainer
        self.model: BaselineModel

        self.additional_train_wrappers: List[Callable] = []
        self.additional_valid_wrappers: List[Callable] = []

    def configure(self, setting: SettingType) -> None:
        """Configures the method for the given Setting.

        Concretely, this creates the model and Trainer objects which will be
        used to train and test a model for the given `setting`.

        Args:
            setting (SettingType): The setting the method will be evaluated on.

        TODO: For the Challenge, this should be some kind of read-only proxy to the
        actual Setting.
        """
        # Note: this here is temporary, just tinkering with wandb atm.
        method_name: str = self.get_name()

        # Set the default batch size to use, depending on the kind of Setting.
        if self.hparams.batch_size is None:
            if isinstance(setting, ActiveSetting):
                # Default batch size of 1 in RL
                self.hparams.batch_size = 1
            elif isinstance(setting, PassiveSetting):
                self.hparams.batch_size = 32
            else:
                warnings.warn(
                    UserWarning(
                        f"Dont know what batch size to use by default for setting "
                        f"{setting}, will try 16."
                    )
                )
                self.hparams.batch_size = 16
        # Set the batch size on the setting.
        setting.batch_size = self.hparams.batch_size

        # TODO: Should we set the 'config' on the setting from here?
        if setting.config and setting.config == self.config:
            pass
        elif self.config != Config():
            assert (
                setting.config is None or setting.config == Config()
            ), "method.config has been modified, and so has setting.config!"
            setting.config = self.config
        elif setting.config:
            assert (
                setting.config != Config()
            ), "Weird, both configs have default values.."
            self.config = setting.config

        setting_name: str = setting.get_name()
        dataset: str = setting.dataset
        wandb_options: WandbLoggerConfig = self.trainer_options.wandb
        if wandb_options.run_name is None:
            wandb_options.run_name = f"{method_name}-{setting_name}" + (
                f"-{dataset}" if dataset else ""
            )

        if isinstance(setting, IncrementalSetting):
            if self.hparams.multihead is None:
                # Use a multi-head model by default if the task labels are
                # available at both train and test time.
                if setting.task_labels_at_test_time:
                    assert setting.task_labels_at_train_time
                self.hparams.multihead = setting.task_labels_at_test_time

        if isinstance(setting, ContinualRLSetting):
            setting.add_done_to_observations = True

            if not setting.observe_state_directly:
                if self.hparams.encoder is None:
                    self.hparams.encoder = "simple_convnet"
                # TODO: Add 'proper' transforms for cartpole, specifically?
                from sequoia.common.transforms import Transforms

                setting.train_transforms.append(Transforms.resize_64x64)
                setting.val_transforms.append(Transforms.resize_64x64)
                setting.test_transforms.append(Transforms.resize_64x64)

            # Configure the baseline specifically for an RL setting.
            # TODO: Select which output head to use from the command-line?
            # Limit the number of epochs so we never iterate on a closed env.
            # TODO: Would multiple "epochs" be possible?
            if setting.max_steps is not None:
                self.trainer_options.max_epochs = 1
                self.trainer_options.limit_train_batches = setting.max_steps // (
                    setting.batch_size or 1
                )
                self.trainer_options.limit_val_batches = min(
                    setting.max_steps // (setting.batch_size or 1), 1000
                )
                # TODO: Test batch size is limited to 1 for now.
                # NOTE: This isn't used, since we don't call `trainer.test()`.
                self.trainer_options.limit_test_batches = setting.max_steps

        self.model = self.create_model(setting)
        assert self.hparams is self.model.hp

        # The PolicyHead actually does its own backward pass, so we disable
        # automatic optimization when using it.
        from .models.output_heads import PolicyHead

        if isinstance(self.model.output_head, PolicyHead):
            # Doing the backward pass manually, since there might not be a loss
            # at each step.
            self.trainer_options.automatic_optimization = False

        self.trainer = self.create_trainer(setting)

    def fit(
        self,
        train_env: Environment[Observations, Actions, Rewards],
        valid_env: Environment[Observations, Actions, Rewards],
    ):
        """Called by the Setting to train the method.
        Could be called more than once before training is 'over', for instance
        when training on a series of tasks.
        Overwrite this to customize training.
        """
        assert self.model is not None, (
            "Setting should have been called method.configure(setting=self) "
            "before calling `fit`!"
        )
        return self.trainer.fit(
            model=self.model, train_dataloader=train_env, val_dataloaders=valid_env,
        )

    def get_actions(
        self, observations: Observations, action_space: gym.Space
    ) -> Actions:
        """ Get a batch of predictions (actions) for a batch of observations.

        This gets called by the Setting during the test loop.

        TODO: There is a mismatch here between the type of the output of this
        method (`Actions`) and the type of `action_space`: we should either have
        a `Discrete` action space, and this method should return ints, or this
        method should return `Actions`, and the `action_space` should be a
        `NamedTupleSpace` or something similar.
        Either way, `get_actions(obs, action_space) in action_space` should
        always be `True`.
        """
        self.model.eval()

        # Check if the observation is batched or not. If it isn't, add a
        # batch dimension to the inputs, and later remove any batch
        # dimension from the produced actions before they get sent back to
        # the Setting.
        single_obs_space = self.model.observation_space

        model_inputs = observations

        # Check if the observations aren't batched.
        not_batched = observations[0].shape == single_obs_space[0].shape
        if not_batched:
            model_inputs = observations.with_batch_dimension()

        with torch.no_grad():
            forward_pass = self.model(model_inputs)
        # Simplified this for now, but we could add more flexibility later.
        assert isinstance(forward_pass, ForwardPass)

        # If the original observations didn't have a batch dimension,
        # Remove the batch dimension from the results.
        if not_batched:
            forward_pass = forward_pass.remove_batch_dimension()

        actions: Actions = forward_pass.actions
        action_numpy = actions.actions_np
        assert action_numpy in action_space, (action_numpy, action_space)
        return actions

    def create_model(self, setting: SettingType) -> BaselineModel[SettingType]:
        """Creates the BaselineModel (a LightningModule) for the given Setting.

        You could extend this to customize which model is used depending on the
        setting.

        TODO: As @oleksost pointed out, this might allow the creation of weird
        'frankenstein' methods that are super-specific to each setting, without
        really having anything in common.

        Args:
            setting (SettingType): An experimental setting.

        Returns:
            BaselineModel[SettingType]: The BaselineModel that is to be applied
            to that setting.
        """
        # Create the model, passing the setting, hparams and config.
        return BaselineModel(setting=setting, hparams=self.hparams, config=self.config)

    def create_trainer(self, setting: SettingType) -> Trainer:
        """Creates a Trainer object from pytorch-lightning for the given setting.

        NOTE: At the moment, uses the KNN and VAE callbacks.
        To use different callbacks, overwrite this method.

        Args:

        Returns:
            Trainer: the Trainer object.
        """
        # We use this here to create loggers!
        callbacks = self.create_callbacks(setting)
        trainer = self.trainer_options.make_trainer(
            config=self.config, callbacks=callbacks,
        )
        return trainer

    def get_experiment_name(self, setting: Setting, experiment_id: str = None) -> str:
        """Gets a unique name for the experiment where `self` is applied to `setting`.

        This experiment name will be passed to `orion` when performing a run of
        Hyper-Parameter Optimization.

        Parameters
        ----------
        - setting : Setting

            The `Setting` onto which this method will be applied. This method will be used when

        - experiment_id: str, optional

            A custom hash to append to the experiment name. When `None` (default), a
            unique hash will be created based on the values of the Setting's fields.

        Returns
        -------
        str
            The name for the experiment.
        """
        if not experiment_id:
            setting_dict = setting.to_dict()
            # BUG: Some settings have non-string keys/value or something?
            experiment_id = compute_identity(size=5, **setting_dict)
        assert isinstance(
            setting.dataset, str
        ), "assuming that dataset is a str for now."
        return (
            f"{self.get_name()}-{setting.get_name()}_{setting.dataset}_{experiment_id}"
        )

    def get_search_space(self, setting: Setting) -> Mapping[str, Union[str, Dict]]:
        """Returns the search space to use for HPO in the given Setting.

        Parameters
        ----------
        setting : Setting
            The Setting on which the run of HPO will take place.

        Returns
        -------
        Mapping[str, Union[str, Dict]]
            An orion-formatted search space dictionary, mapping from hyper-parameter
            names (str) to their priors (str), or to nested dicts of the same form.
        """
        return self.hparams.get_orion_space()

    def hparam_sweep(
        self,
        setting: Setting,
        search_space: Dict[str, Union[str, Dict]] = None,
        experiment_id: str = None,
        database_path: Union[str, Path] = None,
        max_runs: int = None,
    ) -> Tuple[BaselineModel.HParams, float]:
        """ Performs a Hyper-Parameter Optimization sweep using orion.

        Changes the values in `self.hparams` iteratively, returning the best hparams
        found so far.

        Parameters
        ----------
        setting : Setting
            Setting to run the sweep on.

        search_space : Dict[str, Union[str, Dict]], optional
            Search space of the hyper-parameter optimization algorithm. Defaults to
            `None`, in which case the result of the `get_search_space` method is used.

        experiment_id : str, optional
            Unique Id to use when creating the experiment in Orion. Defaults to `None`,
            in which case a hash of the `setting`'s fields is used.

        database_path : Union[str, Path], optional
            Path to a pickle file to be used by Orion to store the hyper-parameters and
            their corresponding values. Default to `None`, in which case the database is
            created at path `./orion_db.pkl`.

        max_runs : int, optional
            Maximum number of runs to perform. Defaults to `None`, in which case the run
            lasts until the search space is exhausted.

        Returns
        -------
        Tuple[BaselineModel.HParams, float]
            Best HParams, and the corresponding performance.
        """

        # TODO: Maybe make this more general than just the BaselineMethod, if there's a
        # demand for that, so that any other method can use this by just implementing
        # some simple method like an `adapt_to_new_hparams` or something.
        from orion.client import build_experiment
        from orion.core.worker.trial import Trial

        # Setting max epochs to 1, just to keep runs somewhat short.
        self.trainer_options.max_epochs = 1

        # Call 'configure', so that we create `self.model` at least once, which will
        # update the hparams.output_head field to be of the right type. This is
        # necessary in order for the `get_orion_space` to retrieve all the hparams
        # of the output head.
        self.configure(setting)
        search_space = search_space or self.get_search_space(setting)
        logger.info("HPO Search space:\n" + json.dumps(search_space, indent="\t"))

        database_path: Path = Path(database_path or "./orion_db.pkl")
        experiment_name = self.get_experiment_name(setting, experiment_id=experiment_id)

        experiment = build_experiment(
            name=experiment_name,
            space=search_space,
            debug=self.config.debug,
            algorithms="BayesianOptimizer",
            max_trials=max_runs,
            storage={
                "type": "legacy",
                "database": {"type": "pickleddb", "host": str(database_path),},
            },
        )

        previous_trials: List[Trial] = experiment.fetch_trials_by_status("completed")
        previous_hparams: List[BaselineModel.HParams] = [
            type(self.hparams).from_dict(trial.params) for trial in previous_trials
        ]
        # Since Orion works in a 'lower is better' fashion, so if the `objective` of the
        # Results class for the given Setting have "higher is better", we negate the
        # objectives when extracting them and again before submitting them to Orion.
        lower_is_better = setting.Results.lower_is_better
        sign = 1 if lower_is_better else -1
        previous_objectives: List[float] = [
            sign * trial.objective.value for trial in previous_trials
        ]
        if previous_objectives:
            logger.info(
                f"Using existing Experiment {experiment} which has "
                f"{len(previous_trials)} existing trials."
            )
            best_index = (np.argmin if lower_is_better else np.argmax)(
                previous_objectives
            )
            best_hparams = previous_hparams[best_index]
            best_objective = previous_objectives[best_index]
        else:
            logger.info(f"Created new experiment with name {experiment_name}")
            best_hparams = self.hparams
            best_objective = None
        logger.info(f"Best result encountered so far: {best_objective}")
        logger.info(f"Best hparams so far: {best_hparams}")

        while not experiment.is_done:
            # Get a new suggestion of hparams to try:
            trial: Trial = experiment.suggest()

            ## Re-create the Model with the new suggested Hparams values.

            new_params: Dict = trial.params
            # Inner function, just used to make the code below a bit simpler.
            # TODO: We should probably also change some values in the Config (e.g.
            # log_dir, checkpoint_dir, etc) between runs.
            logger.info(
                "Suggested values for this run:\n" + json.dumps(new_params, indent="\t")
            )
            # Here we overwrite the corresponding attributes with the new suggested values
            # leaving other fields unchanged.
            new_hparams = self.hparams.replace(**new_params)
            # Change the hyper-parameters, then reconfigure (which recreates the model).
            self.hparams = new_hparams
            self.configure(setting)

            ## Evaluate the method again on the setting:

            result: Results = setting.apply(self, config=self.config)
            experiment.observe(
                trial,
                [
                    dict(
                        name=result.objective_name,
                        type="objective",
                        value=sign * result.objective,
                    )
                ],
            )

            ## Receive the new results.
            better = operator.lt if lower_is_better else operator.gt
            if best_objective is None:
                # First run:
                best_hparams = self.hparams
                best_objective = result.objective
            elif better(result.objective, best_objective):
                # New best result.
                best_hparams = self.hparams
                best_objective = result.objective

            # Receive the results, maybe log to wandb, whatever you wanna do.
            self.receive_results(setting, result)
        return best_hparams, best_objective

    def receive_results(self, setting: Setting, results: Results):
        """ Receives the results of an experiment, where `self` was applied to Setting
        `setting`, which produced results `results`.
        """
        method_name: str = self.get_name()
        setting_name: str = setting.get_name()
        dataset = setting.dataset
        if wandb.run:
            wandb.summary["method"] = method_name
            wandb.summary["setting"] = setting_name
            if dataset and isinstance(dataset, str):
                wandb.summary["dataset"] = dataset
            wandb.log(results.to_log_dict())
            wandb.log(results.make_plots())
            wandb.run.finish()
        # Reset the run name so we create a new one next time we're applied on a
        # Setting.
        self.trainer_options.wandb.run_name = None

    def create_callbacks(self, setting: SettingType) -> List[Callback]:
        """Create the PytorchLightning Callbacks for this Setting.

        These callbacks will get added to the Trainer in `create_trainer`.

        Parameters
        ----------
        setting : SettingType
            The `Setting` on which this Method is going to be applied.

        Returns
        -------
        List[Callback]
            A List of `Callaback` objects to use during training.
        """
        # TODO: Move this to something like a `configure_callbacks` method in the model,
        # once PL adds it.
        # from sequoia.common.callbacks.vae_callback import SaveVaeSamplesCallback
        return [
            # self.hparams.knn_callback,
            # SaveVaeSamplesCallback(),
        ]

    def apply_all(
        self, argv: Union[str, List[str]] = None
    ) -> Dict[Type[Setting], Results]:
        """(WIP): Runs this Method on all its applicable settings.

        Returns
        -------

            Dict mapping from setting type to the Results produced by this method.
        """
        applicable_settings = self.get_applicable_settings()

        all_results: Dict[Type[Setting], Results] = {}
        for setting_type in applicable_settings:
            setting = setting_type.from_args(argv)
            results = setting.apply(self)
            all_results[setting_type] = results
        print(f"All results for method of type {type(self)}:")
        print(
            {
                method.get_name(): (results.get_metric() if results else "crashed")
                for method, results in all_results.items()
            }
        )
        return all_results

    def __init_subclass__(
        cls, target_setting: Type[SettingType] = Setting, **kwargs
    ) -> None:
        """Called when creating a new subclass of Method.

        Args:
            target_setting (Type[Setting], optional): The target setting.
                Defaults to None, in which case the method will inherit the
                target setting of it's parent class.
        """
        if not is_dataclass(cls):
            logger.critical(
                UserWarning(
                    f"The BaselineMethod subclass {cls} should be decorated with "
                    f"@dataclass!\n"
                    f"While this isn't strictly necessary for things to work, it is"
                    f"highly recommended, as any dataclass-style class attributes "
                    f"won't have the corresponding command-line arguments "
                    f"generated, which can cause a lot of subtle bugs."
                )
            )
        super().__init_subclass__(target_setting=target_setting, **kwargs)

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """Called when switching between tasks.
        
        Args:
            task_id (int, optional): the id of the new task. When None, we are
            basically being informed that there is a task boundary, but without
            knowing what task we're switching to.
        """
        self.model.on_task_switch(task_id)
