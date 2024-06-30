import json
import os
import shutil
import warnings

import transformers
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

try:
    # transformers v4.41.0 https://github.com/huggingface/transformers/commit/ad697f18016cf5eb4be48f9552c69c2421aa5581
    from transformers.trainer_callback import ExportableState

    has_exportable = True
except ImportError:
    has_exportable = False


# 保存模型时的回调函数
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.local_rank == 0 or args.local_rank == -1:
            # 保存效果最好的模型
            best_checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best"
            )
            # 因为只保存最新5个检查点，所以要确保不是之前的检查点
            if os.path.exists(state.best_model_checkpoint):
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)
                shutil.copytree(state.best_model_checkpoint, best_checkpoint_folder)
            print(f"效果最好的检查点为：{state.best_model_checkpoint}，评估结果为：{state.best_metric}")
        return control


def skip_first_batches(dataloader, num_batches=0):

    return dataloader


def patch_skip_batches():
    """Accelerater can't correctly skip lhotse dataloader."""

    transformers.trainer.skip_first_batches = skip_first_batches


if has_exportable:
    # 保存lhotse数据状态
    class LhotseSamplerMixin(ExportableState):
        # 指向lhotse的CutSampler的指针
        cut_sampler = None

        def __init__(self):
            # resume所需的sampler state来源
            self.init_sampler_state = None

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):

            if self.cut_sampler:
                if self.init_sampler_state:
                    try:

                        self.cut_sampler.load_state_dict(self.init_sampler_state)
                    except Exception as exc:
                        msg = f"{exc}\n[extra info] Faield to restore the state of the sampler that is described in a state_dict:{self.init_sampler_state}.\nFallback to original sampler."
                        warnings.warn(f"{msg}")
                else:
                    self.init_sampler_state = self.sampler_state

        def state(self) -> dict:
            return {
                "args": {},
                "attributes": {
                    "sampler_state": self.sampler_state,
                },
            }

        @property
        def sampler_state(self) -> dict:
            return self.cut_sampler.state_dict() if self.cut_sampler else {}

        @sampler_state.setter
        def sampler_state(self, state: dict):
            self.init_sampler_state = state

else:

    class LhotseSamplerMixin(object):
        def __init__(*args, **kwargs):
            pass


# transformers v4.41.0 引入stateful callback，实现resume训练。
#   https://github.com/huggingface/transformers/commit/ad697f18016cf5eb4be48f9552c69c2421aa5581
# lhotse数据epoch设定
class LhotseCallback1(LhotseSamplerMixin, TrainerCallback):
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.cut_sampler = kwargs["train_dataloader"].sampler
        super().on_train_begin(args, state, control, **kwargs)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        sampler = self.cut_sampler
        sampler.set_epoch(sampler.epoch + 1)


class LhotseCallback(TrainerCallback):

    cut_sampler = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.cut_sampler = kwargs["train_dataloader"].sampler
        output_dir = None
        if isinstance(args.resume_from_checkpoint, bool):
            if args.resume_from_checkpoint:
                output_dir = get_last_checkpoint(args.output_dir)
        elif args.resume_from_checkpoint is not None:
            output_dir = args.resume_from_checkpoint
        if output_dir is not None:

            data_state_pth = os.path.join(output_dir, "lhotset_sampler.json")
            if os.path.exists(data_state_pth):
                with open(data_state_pth, "r") as f:
                    try:

                        self.cut_sampler.load_state_dict(self.init_sampler_state)
                    except Exception as exc:
                        msg = f"{exc}\n[extra info] Faield to restore the state of the sampler that is described in {data_state_pth}.\nFallback to original sampler."
                        warnings.warn(f"{msg}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        output_dir = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        if args.local_rank <= 0:
            data_state_pth = os.path.join(output_dir, "lhotset_sampler.json")

            with open(data_state_pth, "w") as f:
                f.write(
                    json.dumps(self.cut_sampler.state_dict(), indent=2, sort_keys=True)
                    + "\n"
                )

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        sampler = self.cut_sampler
        sampler.set_epoch(sampler.epoch + 1)
