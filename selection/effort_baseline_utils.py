from transformers import Trainer
import math
import os
import sys

import warnings

from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from torch.autograd import grad
# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    hp_params,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils

import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
import numpy as np

from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

# from transformers.trainer_callback import (
#     DefaultFlowCallback,
#     ProgressCallback,
#     TrainerState,
# )
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ExportableState,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
    get_model_param_count,
)
from transformers.trainer_utils import (
    HPSearchBackend,
    # ShardedDDPOption,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    seed_worker,
    set_seed,
)

from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

import datasets
from peft import PeftModel

import ipdb 

logger = logging.get_logger(__name__)

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"



import time
import pickle

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

    if version.parse(accelerate_version) > version.parse("0.20.3"):
        from accelerate.utils import (
            load_fsdp_model,
            load_fsdp_optimizer,
            save_fsdp_model,
            save_fsdp_optimizer,
        )

def save_time_cost(time_cost,base_dir,task_name,key=None):
    time_path=os.path.join(base_dir,f'time_cost_{task_name}.pkl')
    if not os.path.exists(time_path):
        record_dict={}
    else:
        with open(time_path, 'rb') as f:
            record_dict = pickle.load(f)
    if isinstance(time_cost,dict):
        for key in time_cost.keys():
            record_dict[key]={}
            record_dict[key]['record_time']=time_cost[key]
        with open(time_path, 'wb') as f:
            pickle.dump(record_dict, f)
    else:
        for method_name in record_dict.keys():
            if key.startswith(method_name):
                num=key.split('-')[-1]
                record_dict[method_name][num]=time_cost+record_dict[method_name]['record_time']
        with open(time_path, 'wb') as f:
            pickle.dump(record_dict, f)

def save_feature(score_dict,base_dir,task_name):
    feature_path=os.path.join(base_dir,f'feature_{task_name}.pt')
    target_path=os.path.join(base_dir,f'target_{task_name}.pt')
    count=0
    while os.path.exists(feature_path):
        count+=1
        feature_path=os.path.join(base_dir,f'feature_{task_name}-{count}.pt')
        target_path=os.path.join(base_dir,f'target_{task_name}-{count}.pt')
    torch.save(score_dict['feature'],feature_path)
    torch.save(score_dict['target'],target_path)

def load_feature(base_dir,task_name):
    feature_path=os.path.join(base_dir,f'feature_{task_name}.pt')
    target_path=os.path.join(base_dir,f'target_{task_name}.pt')
    feature=torch.load(feature_path)
    target=torch.load(target_path)

    

    count=1
    feature_path=os.path.join(base_dir,f'feature_{task_name}-{count}.pt')
    target_path=os.path.join(base_dir,f'target_{task_name}-{count}.pt')
    while os.path.exists(feature_path):
        tmp_feature=torch.load(feature_path)
        feature.extend(tmp_feature)
        tmp_target=torch.load(target_path)
        target.extend(tmp_target)
        count+=1
        feature_path=os.path.join(base_dir,f'feature_{task_name}-{count}.pt')
        target_path=os.path.join(base_dir,f'target_{task_name}-{count}.pt')
    return feature,target


class Effort_Trainer(Trainer):

    def set_attribte(self):
        self.get_sample_grad = False
        self.get_batch_grad = False

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return SequentialSampler(self.train_dataset)

    def get_train_shuffle_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def _get_train_shuffle_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return RandomSampler(self.train_dataset)
        
    def get_sample_loss(self, inputs):
        
        # model = self._wrap_model(self.model_wrapped)
        # with self.accelerator.accumulate(model):
        loss = self.loss_step(self.model, inputs)  
        return loss

    def get_sample_loss_debug(self, inputs):
        
        model = self.model
        model.train()
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs) 
        loss = outputs['loss']
        return loss
    
    def loss_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        return loss

    def get_grad(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        method=None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args
        args.method=method

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if (
            resume_from_checkpoint is not None
            and not is_sagemaker_mp_enabled()
            and not self.is_deepspeed_enabled
            and not self.is_fsdp_enabled
        ):
            self._load_from_checkpoint(resume_from_checkpoint)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled#xy

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # self.state = TrainerState()
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )#xy
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size#xy

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)


        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        # start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            gradients = [None for _ in range(len(epoch_iterator))]
            print("*** len of gradients list:", len(gradients))

            self.TIME_FORWARD = 0
            self.TIME_BACKWARD = 0
            self.TIME_APPEND = 0
            self.TIME_FETCH_GRAD = 0
            self.TIME_ALL = 0 

            # if args.method=='EL2N':
            score_dict={}
            score_dict['score']={}
            # feature_list=[]
            # target_list=[]

            # START = time.time()
            step = -1
            for step, inputs in enumerate(tqdm(epoch_iterator)):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if 'D2' in args.method or 'Moderate' in args.method:
                    start_time=time.time()
                    if 'feature' not in score_dict.keys():
                        score_dict['feature']=[]
                        score_dict['target']=[]
                    elif len(score_dict['feature'])>=20000:
                        save_feature(score_dict,self.base_dir,self.task_name)
                        score_dict['feature']=[]
                        score_dict['target']=[]
                    # only save feature in this step
                    mask = inputs.data['labels'] != -100
                    target = inputs.data['labels'][mask].cpu().numpy().tolist()
                    # mask_logits = mask.unsqueeze(-1).expand(-1, -1, outputs.logits.size(-1)).cpu()
                    # feature = outputs.logits[mask_logits].view(outputs.logits.size(0), -1, outputs.logits.size(-1)).squeeze(0).cpu()
                    feature = model(**inputs,output_hidden_states=True).hidden_states[-1][mask].cpu().detach()
                    score_dict['feature'].append(feature)
                    score_dict['target'].append(target)
                    del mask,target,feature,inputs
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    _time_cost=time.time()-start_time
                    try:
                        self.time_cost['D2']+=_time_cost
                        self.time_cost['Moderate']+=_time_cost
                    except:
                        pass
                else:
                    # if args.method=='EL2N':
                    with self.accelerator.accumulate(model):
                        gradient,raw_score_dict = self.training_step(model, inputs, method=args.method)
                        # loss_list.append(float(loss.cpu().detach().numpy()))
                    # if args.method=='EL2N':
                    #     loss_list.append(loss)
                    # else:
                    #     with self.accelerator.accumulate(model):
                    #         gradient = self.training_step(model, inputs)

                    gradients[step] = torch.norm(gradient,dim=1).cpu().detach()
                    for key in raw_score_dict.keys():
                        if key not in score_dict['score'].keys():
                            score_dict['score'][key]=[]
                        score_dict['score'][key].append(raw_score_dict[key])

                model.zero_grad()


            if 'EL2N' in score_dict['score'].keys():
                score_dict['score']['EL2N']=torch.cat([torch.Tensor(t).unsqueeze(0) for t in score_dict['score']['EL2N']])
            if 'Entropy' in score_dict['score'].keys():
                score_dict['score']['Entropy']=torch.cat([torch.sum(t, dim=1) for t in score_dict['score']['Entropy']])
                score_dict['score']['Loss_Entropy']=torch.cat([t.unsqueeze(0) for t in score_dict['score']['Loss_Entropy']])
            if 'Effort' in args.method:
                score_dict['score']['Effort']=torch.cat([_[0].cpu().unsqueeze(0) for _ in gradients], dim=0)
                # score_dict['score']['Effort']=torch.norm(all_gradients, dim=1)
            # if 'D2' in args.method or 'Moderate' in args.method:
            #     score_dict['feature']=feature_list
            #     score_dict['target']=target_list
            # if args.method=='EL2N': return torch.cat([torch.Tensor(t).unsqueeze(0) for t in loss_list])
            return score_dict
            # if self.get_sample_grad:
            #     return gradients
            # elif self.get_batch_grad:
            #     return torch.mean(torch.cat([_[0].unsqueeze(0) for _ in gradients], dim=0), dim=0)
            

    def training_step(self,
                      model: nn.Module,
                      inputs: Dict[str, Union[torch.Tensor, Any]],
                      method=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        
        model.train()
        if self.model_type=='llama27':
            target_layer=31
        elif self.model_type=='llama213':
            target_layer=39
        elif self.model_type=='gemma':
            target_layer=17
        elif self.model_type=='gemma7':
            target_layer=27
        elif (self.model_type=='codegen1' or self.model_type=='codegen37'):
            target_layer=15
        # elif self.model_type=='codegen37':
        #     target_layer=27
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)


        with self.compute_loss_context_manager():
            loss, score_dict = self.compute_loss(model, inputs,method)
        # self.TIME_FORWARD += time.time()-start
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        #xy
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(torch.mean(loss), retain_graph=True)

        start_time=time.time()
        if self.get_batch_grad:
            try:
                gradient = model.base_model.model.base_model.model.model.layers[target_layer].self_attn.v_proj.lora_B.weight.grad.reshape(-1).cpu()
            except:
                gradient = model.base_model.model.model.layers[target_layer].self_attn.v_proj.lora_B.weight.grad.reshape(-1).cpu()
        
        elif self.get_sample_grad:
            try:
                gradient = [None for _ in range(len(loss))]
                for s_id, s_loss in enumerate(loss):
                    if s_id == len(loss)-1:
                        # self.accelerator.backward(s_loss)
                        self.accelerator.backward(torch.mean(s_loss))
                    else:
                        # self.accelerator.backward(s_loss, retain_graph=True)
                        self.accelerator.backward(torch.mean(s_loss),retain_graph=True)
                    try:
                        gradient[s_id] = model.base_model.model.base_model.model.model.layers[target_layer].self_attn.v_proj.lora_B.default.weight.grad.reshape(1,-1).cpu()
                    except:
                        gradient[s_id] = model.base_model.model.model.layers[target_layer].self_attn.v_proj.lora_B.default.weight.grad.reshape(1,-1).cpu()
            except:
                gradient = [None]

                # self.accelerator.backward(s_loss)
                self.accelerator.backward(loss)
                if 'codegen' in self.model_type:
                    try:
                        gradient[0] = model.base_model.model.base_model.model.transformer.h[target_layer].attn.qkv_proj.lora_B.default.weight.grad.reshape(1,-1).cpu()
                    except:
                        gradient[0] = model.base_model.model.transformer.h[target_layer].attn.qkv_proj.lora_B.default.weight.grad.reshape(1,-1).cpu()
                else:
                    try:
                        gradient[0] = model.base_model.model.base_model.model.model.layers[target_layer].self_attn.v_proj.lora_B.default.weight.grad.reshape(1,-1).cpu()
                    except:
                        gradient[0] = model.base_model.model.model.layers[target_layer].self_attn.v_proj.lora_B.default.weight.grad.reshape(1,-1).cpu()
        _time_cost=time.time()-start_time
        if 'Effort' in method:
            self.time_cost['Effort']+=_time_cost
        return torch.cat(gradient, dim=0).cpu(),score_dict


    def compute_loss(self, model, inputs, method=[],return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.get_sample_grad:
            if self.label_smoother is not None:
                self.label_smoother = Sample_LabelSmoother(epsilon=self.args.label_smoothing_factor)
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        raw_score_dict={}
        if 'EL2N' in method or 'CCS-EL2N' in method:
            start_time=time.time()
            mse_loss=torch.nn.MSELoss()
            mask = inputs.data['labels'] != -100
            filtered_label=inputs.data['labels'][mask]
            mask_logits = mask.unsqueeze(-1).expand(-1, -1, outputs.logits.size(-1))
            filtered_logits = outputs.logits[mask_logits].view(outputs.logits.size(0), -1, outputs.logits.size(-1)).squeeze(0)

            label_one_hot = F.one_hot(filtered_label, num_classes=outputs.logits.size()[-1])#.to(outputs.logits.device)#.cpu()#
            loss = torch.sqrt(mse_loss(filtered_logits,label_one_hot)).cpu().detach().numpy()
            raw_score_dict['EL2N']=loss
            del filtered_label,label_one_hot, filtered_logits,mask_logits,mask,mse_loss
            torch.cuda.empty_cache()
            _time_cost=time.time()-start_time
            self.time_cost['EL2N']+=_time_cost
            self.time_cost['CCS-EL2N']+=_time_cost


        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                try:
                    loss=outputs.loss['logits']
                except Exception as e:
                    print(e)
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # ipdb.set_trace()
        # return (loss, outputs) if return_outputs else loss
        del outputs
        torch.cuda.empty_cache()
        return loss,raw_score_dict
    
    def compute_mse_loss(self, model, inputs, return_outputs=False, misc=[]):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        score_dict={}

        mse_loss=torch.nn.MSELoss()
        outputs = model(**inputs)
            
        mask = inputs.data['labels'] != -100
        filtered_label=inputs.data['labels'][mask]

        mask_logits = mask.unsqueeze(-1).expand(-1, -1, outputs.logits.size(-1))
        filtered_logits = outputs.logits[mask_logits].view(outputs.logits.size(0), -1, outputs.logits.size(-1)).squeeze(0)

        label_one_hot = F.one_hot(filtered_label, num_classes=outputs.logits.size()[-1])#.to(outputs.logits.device)#.cpu()#
        loss = torch.sqrt(mse_loss(filtered_logits,label_one_hot)).cpu().detach().numpy()
        score_dict['EL2N']=loss

        # Entropy selection
        if 'entropy' in misc:
            prob = nn.Softmax(dim=1)(outputs.logits)

            entropy = -1 * prob * torch.log(prob + 1e-10)
            entropy = torch.sum(entropy, dim=1).detach().cpu()

            loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()

            score_dict['entropy'] = entropy
            score_dict['loss'] = outputs.loss

        del filtered_label,label_one_hot, filtered_logits,mask_logits,mask,outputs,mse_loss
        torch.cuda.empty_cache()
        return score_dict
        # return (loss, outputs) if return_outputs else loss
    
    def freeze_layers(self,model_type):
        self.model_type=model_type
        for name, params in self.model.named_parameters():
            if self.model_type=='llama27' and ("layers.31.self_attn.v_proj.lora_B.default.weight" not in name) and params.requires_grad:
                params.requires_grad = False
            elif self.model_type=='llama213' and ("layers.39.self_attn.v_proj.lora_B.default.weight" not in name) and params.requires_grad:
                params.requires_grad = False
            elif self.model_type=='gemma' and ("layers.17.self_attn.v_proj.lora_B.default.weight" not in name) and params.requires_grad:
                params.requires_grad = False
            elif self.model_type=='gemma7' and ("layers.27.self_attn.v_proj.lora_B.default.weight" not in name) and params.requires_grad:
                params.requires_grad = False
            elif (self.model_type=='codegen1' or self.model_type=='codegen37') and ("h.15.attn.qkv_proj.lora_B.default.weight" not in name) and params.requires_grad:
                params.requires_grad = False
            # elif self.model_type=='codegen37' and ("layers.27.self_attn.v_proj.lora_B.default.weight" not in name) and params.requires_grad:
            #     params.requires_grad = False
            else:
                # params.requires_grad = True
                if params.requires_grad:
                    print(name)

    def enable_trainable_layers(self):
        for name, params in self.model.named_parameters():
            if self.model_type=='llama27' and ("layers.31.self_attn.v_proj.lora_B.default.weight" in name) or ("layers.31.mlp" in name): # or ("model.norm" in name) or ("model.lm_head" in name)
                try:
                    params.requires_grad = True
                except:
                    ipdb.set_trace()
                    pass
            elif self.model_type=='llama213' and ("layers.39.self_attn.v_proj.lora_B.default.weight" in name) or ("layers.39.mlp" in name): # or ("model.norm" in name) or ("model.lm_head" in name)
                try:
                    params.requires_grad = True
                except:
                    ipdb.set_trace()
                    pass
            elif self.model_type=='gemma' and ("layers.17.self_attn.v_proj.lora_B.default.weight" in name) or ("layers.17.mlp" in name): # or ("model.norm" in name) or ("model.lm_head" in name)
                try:
                    params.requires_grad = True
                except:
                    ipdb.set_trace()
                    pass
            elif self.model_type=='gemma7' and ("layers.27.self_attn.v_proj.lora_B.default.weight" in name) or ("layers.27.mlp" in name): # or ("model.norm" in name) or ("model.lm_head" in name)
                try:
                    params.requires_grad = True
                except:
                    ipdb.set_trace()
                    pass
            elif (self.model_type=='codegen1' or self.model_type=='codegen37') and ("h.15.attn.qkv_proj.lora_B.default.weight" in name) or ("h.15.mlp" in name): # or ("model.norm" in name) or ("model.lm_head" in name)
                try:
                    params.requires_grad = True
                except:
                    ipdb.set_trace()
                    pass

class Sample_LabelSmoother():
    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
