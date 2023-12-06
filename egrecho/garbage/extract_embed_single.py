# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import os
import warnings
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.data.datasets.audio import ASVSamples
from egrecho.pipeline.speaker_embedding import SpeakerEmbedding
from egrecho.utils.data_utils import try_length
from egrecho.utils.io import (
    KaldiVectorWriter,
    get_filename,
    is_remote_url,
    check_input_dataformat,
)
from egrecho.utils.logging import get_logger
from egrecho_cli.register import register_command

logger = get_logger()

DESCRIPTION = """Extract xvector single process."""


@register_command(name="extract_embed_single", help=DESCRIPTION)
class ExtractEmbeddingSingle(BaseCommand):
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
            "modeldir",
            type=str,
            help="dirpath where contains config file, for local checkpoint structure can be "
            "['dir/last.ckpt', 'dir/checkpoints/last.ckpt', 'dir/version_*/checkpoingts/last.ckpt'].",
        )
        parser.add_method_arguments(
            ASVSamples,
            "build_source_dataset",
            "data",
            fail_untyped=False,
            skip={
                "path_or_paths",
                "decode_fn",
                "dew_cls",
                "easy_check",
                "sampler_kwargs",
                "join_str",
            },
        )

        parser.add_method_arguments(
            SpeakerEmbedding,
            "from_pretrained",
            "pipeline",
            fail_untyped=False,
            skip={"batch_size", "tokenizer"},
        )
        parser.link_arguments("modeldir", "pipeline.dirpath")
        parser.add_argument(
            "--outdir",
            type=str,
            default=None,
            help="if not set, auto logic -> Case 1: ckpt relative to pipeline.dirpath, the outdir will be "
            " in the same level as ckpt path with name of ckpt. Case 2: indicates that ckpt is remote or manually "
            "set, outdir is pipeline.dirpath.",
        )

        parser.add_argument(
            "--sub_outdir",
            type=str,
            default="",
            help="subdir of outdir, placeholder of embeddings, can set to indicate some speicial config.",
        )
        parser.add_argument(
            "--kaldi_ark_name",
            type=str,
            default="xvector",
            help="xvector fname, will generate ``'xvector.scp'&'xvector.ark'``, "
            "specify it in multi processes case, (e.g., xvector.1).",
        )
        parser.add_argument(
            "--summary_depth",
            "-sd",
            type=int,
            default=2,
            help="Model summary depth display.",
        )
        parser.add_argument(
            "--pbar_intervel",
            type=int,
            default=None,
            help="pbar update intervel, set a propal number for multi subprocess to avoid crowd log.",
        )
        parser.set_defaults(
            {
                "pipeline.device": "auto",
                "pipeline.num_workers": None,
                "data.shuffle": False,
            }
        )
        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        check_input_dataformat(args.wav_file)

        rank = _infer_rank()
        is_main_rank = (rank or 0) == 0

        embd_pipeline = SpeakerEmbedding.from_pretrained(**args.pipeline)
        if is_main_rank:
            try:
                from lightning.pytorch.utilities.model_summary import ModelSummary

                model_summary = ModelSummary(
                    embd_pipeline.model, max_depth=args.summary_depth
                )
                print(model_summary)
                del model_summary
            except:  # noqa
                warnings.warn("Failed calling lightning model summary.")
                print(embd_pipeline.model)

        outdir = args.outdir
        ckpt_path = embd_pipeline.resolved_opt.checkpoint
        if not outdir:
            dirpath = Path(args.workdir).resolve()
            if not is_remote_url(ckpt_path):
                try:
                    ckpt_rel_dir = (Path(ckpt_path).resolve()).relative_to(dirpath)
                    outdir_rel_dir = ckpt_rel_dir.parent
                except ValueError:
                    outdir_rel_dir = ""

            outdir = dirpath / outdir_rel_dir

        ckpt_name = get_filename(ckpt_path)
        embd_dir = Path(outdir) / args.sub_outdir / ckpt_name.replace(".", "_")

        embd_dir.mkdir(exist_ok=True, parents=True)
        samples = ASVSamples.build_source_dataset(args.wav_file, **args.data)
        tot_len = try_length(samples)

        if not tot_len:
            try:
                tot_len = samples.src_length()  # egrecho iterable rough length
            except Exception:
                pass

        # timer = Timer()
        logger.info(
            f"Start extracting xv -> {embd_dir/(args.kaldi_ark_name+'.scp')}.",
            ranks=0,
        )
        cnt = 0
        args.pbar_intervel = args.pbar_intervel or 1
        desc_suffix = f"rank {rank}" if rank is not None else ""

        with KaldiVectorWriter(embd_dir, args.kaldi_ark_name) as w, tqdm(
            total=tot_len,
            unit=" utts",
            ascii=False,
            disable=not is_main_rank,
            miniters=args.pbar_intervel,
            dynamic_ncols=True,
            desc=f"Extracting {desc_suffix}",
            delay=5,  # delay to wait for first log info.
        ) as pbar:
            for output in embd_pipeline(samples):
                if cnt == 0:
                    if not all(
                        require_key in output for require_key in ("xvector", "id")
                    ):
                        raise ValueError(
                            f"Required 'xvector', 'id' keys in pipeline's output, but got output type: ({type(output)}) value: {output}"
                        )
                id_, xvector = output["id"], output["xvector"]
                if cnt == 0 and is_main_rank:
                    logger.info(f"First id={id_}, xvector shape={xvector.shape})")
                w.write(id_, (xvector.squeeze()).numpy())
                cnt += 1
                if cnt % args.pbar_intervel == 0:
                    pbar.update(args.pbar_intervel)
        logger.info(f"Extracting done, saved {cnt} number xv.", ranks=0)
        return embd_dir / (args.kaldi_ark_name + ".scp")

    # @classmethod
    # def extract_embedding(pipeline: SpeakerEmbedding, samples: ASVSamples, )


def _infer_rank() -> Optional[int]:
    cand_rank_env = ("RANK", "LOCAL_RANK")
    for key in cand_rank_env:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


if __name__ == "__main__":
    parser = ExtractEmbeddingSingle.get_dummy_parser()
    parser = ExtractEmbeddingSingle.setup_parser(parser)
    args = parser.parse_args()
    ExtractEmbeddingSingle.run_from_args(args, parser)
