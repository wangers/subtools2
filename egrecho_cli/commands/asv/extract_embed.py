# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

import egrecho.models.groups.asv_group  # register model
from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.data.datasets.audio import ASVSamples
from egrecho.data.dew import split_raw_file
from egrecho.pipeline.speaker_embedding import SpeakerEmbedding
from egrecho.utils.cuda_utils import device2gpu_id, parse_gpus_opt, release_memory
from egrecho.utils.data_utils import try_length
from egrecho.utils.dist import TorchMPLauncher
from egrecho.utils.io import (
    KaldiVectorWriter,
    buf_count_newlines,
    check_input_dataformat,
    get_filename,
    is_remote_url,
)
from egrecho.utils.logging import _infer_rank, get_logger
from egrecho_cli.register import register_command

logger = get_logger()

gpus_argdoc = r"""
What GPUs should be used.

Case 0: comma-seperated list, e.g., `"1,"` or `"0,1"` means specified id(s).
Case 1: a single int(str) negative number (-1) means all visiable devices `[0, N-1]`.
Case 2: `''` or `None` or 0 returns None means no gpu.
Case 3: a single int(str) number equals 1 means auto choose a spare gpu.
Case 4: a single int(str) number n greater than 1 returns `[0, n-1]`.
"""

DESCRIPTION = """Extract xvector."""


@register_command(name="extract-embed", help=DESCRIPTION)
class ExtractEmbedding(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_cfg_flag()
        parser.add_argument(
            "wav_file",
            type=str,
            help="wav file path, must be jsonl/json/csv format.",
        )
        parser.add_argument(
            "dirpath",
            type=str,
            help="dirpath where contains config file, for local checkpoint structure can be "
            "['dir/last.ckpt', 'dir/checkpoints/last.ckpt', 'dir/version_*/checkpoingts/last.ckpt'].",
        )
        parser.add_argument(
            "--outdir",
            type=str,
            default=None,
            help="if not set, auto logic -> Case 1: ckpt relative to pipeline.dirpath, the outdir will be "
            " in the same level as ckpt path with name of ckpt. Case 2: Can not relove ckpt realtive to dirpath "
            "outdir is pipeline.dirpath with name of ckpt.",
        )

        parser.add_argument(
            "--sub_outdir",
            type=str,
            default="",
            help="subdir of outdir, placeholder of embeddings, can set to indicate some speicial config.",
        )
        parser.add_argument(
            "--gpus",
            default="1",
            type=Union[str, int],
            help=gpus_argdoc,
        )
        parser.add_argument(
            "--number_processes",
            "-nj",
            default=1,
            type=int,
            help="Number of processes, will be averaged number on each gpu.",
        )

        parser.add_method_arguments(
            ASVSamples,
            "build_source_dataset",
            "data",
            fail_untyped=False,
            skip={
                "path_or_paths",
                "decode_fn",
                "sampler_kwargs",
            },  # filt unsused kwargs avoid crowded help
        )

        # parse device outside
        parser.add_method_arguments(
            SpeakerEmbedding,
            "from_pretrained",
            "pipeline",
            fail_untyped=False,
            skip={"resolve_mode", "device", "batch_size", "tokenizer"},
        )
        parser.link_arguments("dirpath", "pipeline.dirpath")

        parser.add_argument(
            "--summary_depth",
            "-sd",
            type=int,
            default=2,
            help="Model summary depth display.",
        )

        parser.set_defaults(
            {
                "pipeline.num_workers": None,
                "data.shuffle": False,
                "data.partition": False,
            }
        )
        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        check_input_dataformat(args.wav_file)

        executor = ExtractEmbedding(args)
        executor.run()

    def __init__(
        self,
        args,
    ):
        # check arguments and get ckpt path, this won't initiate class
        resolve_opt, _ = SpeakerEmbedding.from_pretrained(
            resolve_mode=True, **(args.clone().pipeline)
        )
        ckpt_path = resolve_opt.checkpoint
        self.ckpt_name = get_filename(ckpt_path)

        outdir = args.outdir
        if not outdir:
            dirpath = Path(args.dirpath).resolve()
            if not is_remote_url(ckpt_path):
                try:
                    ckpt_rel_dir = (Path(ckpt_path).resolve()).relative_to(dirpath)
                    outdir_rel_dir = ckpt_rel_dir.parent
                except ValueError:
                    outdir_rel_dir = ""

            outdir = (
                Path(args.dirpath) / outdir_rel_dir / self.ckpt_name.replace(".", "_")
            )
        self.outdir = Path(outdir)
        self.embd_dir = self.outdir / args.sub_outdir

        device_list = parse_gpus_opt(args.gpus)
        device_num = len(device_list) if device_list else 0
        self.use_gpu = device_num > 0

        num_procs = args.number_processes if args.number_processes > 1 else 1
        datafile = args.wav_file
        self.proc_datafiles = split_raw_file(
            datafile, split_num=num_procs, even_chunksize=True, sub_dir=f"{num_procs}"
        )
        self.num_procs = min(
            num_procs, len(self.proc_datafiles)
        )  # edge case: nj may greater than datafile length.
        self.embd_split_dir = self.embd_dir / f"{self.num_procs}"
        self.proc_devices = (
            [device_list[rank % device_num] for rank in range(self.num_procs)]
            if device_num > 0
            else ["cpu"] * self.num_procs
        )
        self.device_list = device_list

        self.args = args

    def run(self):
        if self.num_procs > 1:
            # TODO: can torch.multiprocessing save mem for model when forking ?
            launcher = TorchMPLauncher(
                num_processes=self.num_procs, start_method="spawn"
            )
            launcher.launch(self.run_pipeline)  # 'LOCAL_RANK' will be added in os.env
        else:
            self.run_pipeline()

        # concat splits
        if (_infer_rank() or 0) == 0:
            total_embed_file = self.embd_split_dir / "xvector.scp"

            embed_spilts = (
                self.embd_split_dir / f"xvector.{rank}.scp"
                for rank in range(self.num_procs)
            )

            with open(total_embed_file, "w") as wf:
                for split_file in embed_spilts:
                    with open(split_file) as rf:
                        wf.write(rf.read())
            result_file = shutil.copy(total_embed_file, self.embd_dir)

            # records ckpt-related embed location last running.
            with open(Path(self.args.dirpath) / "extract_outdir.last", "w") as f:
                f.write(str(self.outdir))

            release_memory()  # gc
            logger.info(
                f"Concate extracted embedding done, saved {buf_count_newlines(result_file)} number xv to "
                f"{result_file}",
            )

    def run_pipeline(
        self,
    ):
        infer_rank = _infer_rank()
        rank = infer_rank or 0
        device = self.proc_devices[rank]
        pipeline = SpeakerEmbedding.from_pretrained(device=device, **self.args.pipeline)
        samples = ASVSamples.build_source_dataset(
            self.proc_datafiles[rank], **self.args.data
        )
        if rank == 0:
            try:
                from lightning.pytorch.utilities.model_summary import ModelSummary

                model_summary = ModelSummary(
                    pipeline.model, max_depth=self.args.summary_depth
                )
                print(model_summary)
                del model_summary
            except:  # noqa
                warnings.warn("Failed calling lightning model summary.")
                print(pipeline.model)

        release_memory()

        kaldi_ark_name = f"xvector.{rank}"
        if rank == 0:
            self.log_process()
            logger.info(
                f"Start extracting xv rank [{rank}] -> {self.embd_split_dir/(kaldi_ark_name+'.scp')}.",
                ranks=0,
            )

        tot_len = try_length(samples)
        if not tot_len:
            try:
                tot_len = samples.src_length()  # egrecho iterable rough length
            except Exception:
                pass

        desc_suffix = f" rank {infer_rank}" if infer_rank is not None else ""
        cnt = 0

        with KaldiVectorWriter(self.embd_split_dir, kaldi_ark_name) as w, tqdm(
            total=tot_len,
            unit=" utts",
            ascii=False,
            disable=(rank != 0),
            desc=f"Extracting{desc_suffix}",
            delay=5,  # delay to wait for first log info.
        ) as pbar:
            for output in pipeline(samples):
                if cnt == 0:
                    if not all(
                        require_key in output for require_key in ("xvector", "id")
                    ):
                        raise ValueError(
                            "Required 'xvector', 'id' keys in pipeline's output, "
                            f"but got output type: ({type(output)}) value: {output}"
                        )
                id_, xvector = output["id"], output["xvector"]
                if cnt == 0 and rank == 0:
                    logger.info(f"First id={id_}, xvector shape={xvector.shape})")
                w.write(id_, (xvector.squeeze()).numpy())
                cnt += 1
                pbar.update()

        extra_msg = (
            f", rank {infer_rank} is waiting for other ranks"
            if infer_rank is not None
            else ""
        )
        logger.info(f"Extracting done, saved {cnt} number xv{extra_msg}.", ranks=0)

    def log_process(self):
        if not self.use_gpu:
            warnings.warn(
                f"Execution with number ({self.num_procs}) proc(s) will take place on the CPU, "
                "which may result in inefficiency."
            )
        else:
            device2id = {
                device2gpu_id(device_id): device_id for device_id in self.device_list
            }

            logger.info(
                f"Execution with number ({self.num_procs}) proc(s) will take place on "
                f"{len(self.device_list)} GPU(s):\n#### Device id: {', '.join(map(str, device2id.keys()))} "
                f"\n#### Real cuda id: {', '.join(map(str, device2id.values()))}"
            )

    @property
    def outdir(self):
        return self._out_dir

    @outdir.setter
    def outdir(self, out_dir):
        self._out_dir = out_dir

    @property
    def ckpt_name(self):
        return self._ckpt_name

    @ckpt_name.setter
    def ckpt_name(self, ckpt_name):
        self._ckpt_name = ckpt_name


if __name__ == "__main__":
    parser = ExtractEmbedding.get_dummy_parser()
    parser = ExtractEmbedding.setup_parser(parser)
    args = parser.parse_args()
    ExtractEmbedding.run_from_args(args, parser)
