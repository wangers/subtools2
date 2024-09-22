# -*- coding:utf-8 -*-
# (Author: Leo 202408)

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
from feature_extractor import E2TTSExtractor
from lightning.pytorch.utilities.model_summary import ModelSummary
from pipeline_e2 import E2TTSPipeLine
from tokenizer_e2tts import E2TTSTokenizer
from tqdm.auto import tqdm
from utils import plot_spectrogram, save_array

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.data.dew import DewSamples
from egrecho.models.e2_tts.model import E2TTS, E2TTSConfig
from egrecho.utils.constants import DEFAULT_MODEL_FILENAME
from egrecho.utils.cuda_utils import release_memory
from egrecho.utils.data_utils import try_length
from egrecho.utils.io import resolve_ckpt
from egrecho.utils.logging import get_logger
from egrecho.utils.seeder import set_all_seed
from egrecho.utils.torch_utils import single_torch_nj
from egrecho_cli import register_command

logger = get_logger(__name__)

DESCRIPTION = "TTS e2tts."
DEFAUT_GEN_ARGS = {
    "steps": 32,
    "cfg_strength": 1.0,
    "max_duration": 4096,
    "odeint_kwargs": dict(atol=1e-5, rtol=1e-5, method="midpoint"),
}


@dataclass
class DemoArgs:
    """TTS a demo

    Args:
        text: string content
        prompt_audio: path to prompt audio
        prompt_text: text of prompt_audio
        outfile: out wavfile, if None, get a file in Path('./')
        gen_dur_in_s: need target duration if duration model missing
        gen_args: generate_kwargs
    """

    text: str
    prompt_audio: str
    prompt_text_path: Optional[str] = None
    prompt_text: Optional[str] = None
    outfile: Optional[str] = None
    gen_dur_sec: float = 3.0
    gen_args: dict = field(default_factory=lambda: DEFAUT_GEN_ARGS)
    post_args: dict = field(default_factory=lambda: {"with_cond": False})
    vocoder: Optional[dict] = None

    def __post_init__(self):
        if not self.outfile:
            fname = "e2tts_demo_" + str(Path(self.prompt_audio).name)
            self.outfile = Path("./") / fname
        if self.prompt_text is not None:
            pass
        elif self.prompt_text_path is None:
            raise ValueError(
                f"Either prompt_text_path/prompt_text should be provided, but both are None"
            )
        else:
            self.prompt_text_path = Path(self.prompt_text_path)
            with open(self.prompt_text_path, encoding="utf-8") as f:
                self.prompt_text = f.read().strip()


# @dataclass
# class InferArgs:
#     """Infers audios via a json file, must has fields: text, prompt_audio, prompt_text

#     Args:
#         eg_file: test egfile
#         outdir: output storage
#         gen_args: generate_kwargs

#     """

#     eg_file: str
#     outdir: str
#     overwrite: bool = True
#     gen_args: dict = field(default_factory=lambda: DEFAUT_GEN_ARGS)


@dataclass
class ExportArgs:
    """
    Args:
        outdir: outdir of model, should be empty or not exitst.
    """

    outdir: str

    def __post_init__(self):
        self.outdir = Path(self.outdir)
        if self.outdir.is_dir() and list(self.outdir.glob("*")):
            raise FileExistsError(
                f"The output folder exists and is not empty: {str(self.outdir)}."
                " Please delete it first or choose a different name."
            )


@register_command(name="tts-e2")
class TTSE2(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser) -> CommonParser:
        parser.add_cfg_flag()

        parser.add_function_arguments(resolve_ckpt, "model")
        parser.add_argument(
            "--gpu_id", type=Union[int, str], default="auto", help="|auto|-1|0|"
        )
        parser.set_defaults(
            {
                "model.dirpath": "exp/e2tts",
            }
        )

        subcommands = parser.add_subcommands(title=None)

        demo_parser = CommonParser(description="TTS a demo.")
        subcommands.add_subcommand("demo", demo_parser)
        demo_parser.add_class_args(DemoArgs)
        # infer_parser = CommonParser(description="Infer valle.")
        # subcommands.add_subcommand("infer", infer_parser)
        # infer_parser.add_class_args(InferArgs)
        export_parser = CommonParser(description="Exports to a new directory.")
        subcommands.add_subcommand("export", export_parser)
        export_parser.add_class_args(ExportArgs)
        return parser

    @staticmethod
    def run_from_args(args, parser: CommonParser):

        # args = parser.instantiate_classes(args)

        eng = TTSE2(args, parser)
        eng.run()

    def __init__(self, args, parser):

        self.args = args
        self.subcommand_name = self.args["subcommand"]
        self.sub_args = self.args[self.subcommand_name]
        set_all_seed(seed=42)

        model_ckpt = resolve_ckpt(**args.model)
        cfgdir = Path(model_ckpt).parent / "config"
        self.tokenizer = E2TTSTokenizer.fetch_from(cfgdir)
        self.extractor = E2TTSExtractor.fetch_from(cfgdir)
        self.model = E2TTS.fetch_from(
            Path(model_ckpt).parent, init_weight=Path(model_ckpt).name
        )

        # invalid it cauze we are going to meth rather generate than forward
        self.model.example_input_array = None
        model_summary = ModelSummary(self.model, max_depth=3)
        print(model_summary)
        release_memory(model_summary)
        logger.info(f"Get model done!")

    def request_pipline(
        self, generate_kwargs=None, postprocess_params=None, vocoder=None
    ):
        logger.info(f"Request an e2tts pipeline.")
        pipeline = E2TTSPipeLine(
            feature_extractor=self.extractor,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.args.gpu_id,
            vocoder=vocoder,
            generate_kwargs=generate_kwargs,
            postprocess_params=postprocess_params,
        )
        return pipeline

    def run(self):
        if self.subcommand_name == "demo":
            self.run_demo()
        # if self.subcommand_name == "infer":
        #     self.run_infer()
        if self.subcommand_name == "export":
            self.run_export()

    def run_demo(self):
        demo_args = DemoArgs(**self.sub_args)
        pipline = self.request_pipline(
            generate_kwargs=demo_args.gen_args,
            postprocess_params=demo_args.post_args,
            vocoder=demo_args.vocoder,
        )
        result = pipline(
            inputs=demo_args.text,
            prompt=(demo_args.prompt_audio, demo_args.prompt_text),
            gen_dur_sec=demo_args.gen_dur_sec,
        )[0]
        sampling_rate = result["sampling_rate"]
        meta = result["meta"]
        outfile = demo_args.outfile
        if "samples" in result:
            samples = result["samples"].cpu().detach()
            torchaudio.save(str(outfile), samples, sampling_rate)
        if "mel" in result:
            mel = result["mel"].cpu().detach()
            mel_txt = Path(outfile).with_suffix(".txt")
            mel_fig = Path(outfile).with_suffix(".png")
            save_array(mel, mel_txt)
            plot_spectrogram(mel, save_to=mel_fig)
        rtf = round(meta["norm_gen_time"] / meta["gen_dur"], 3)
        logger.info(
            f"TTS demo done, outfname {str(Path(outfile).with_suffix(''))}, meta={meta}, rtf={rtf}."
        )
        return result

    # def run_infer(self):

    #     infer_args = InferArgs(**self.sub_args)
    #     gen_kwargs = infer_args.gen_args
    #     overwrite = infer_args.overwrite
    #     out_dir = Path(infer_args.outdir)
    #     if out_dir.is_dir():
    #         if overwrite:
    #             logger.warning(f"Force clearing ({out_dir}) ...")
    #             shutil.rmtree(out_dir)
    #         else:
    #             try:
    #                 next(out_dir.iterdir())
    #             except StopIteration:
    #                 pass
    #             else:
    #                 raise ValueError(f"{out_dir} already exists, cleaning it or set overwrite=True.")
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     inputs = DewSamples.from_files(infer_args.eg_file)
    #     guess_length = try_length(inputs)
    #     pipline = self.request_pipline(generate_kwargs=gen_kwargs)
    #     manifest_path = out_dir / 'egs_infer_tts.jsonl'
    #     dew_writer = DewSamples.open_writer(manifest_path)
    #     save_wav_dir = out_dir / 'audio'
    #     save_wav_dir.mkdir(parents=True, exist_ok=True)

    #     def _save_worker(
    #         batch_result: list,
    #         idxs: list,
    #     ):
    #         for i, result in enumerate(batch_result):

    #             if ref_audio := result.get('ref_audio'):
    #                 candidate_id = f"{idxs[i]:04d}-{Path(ref_audio).stem}"
    #             else:
    #                 candidate_id = f"{idxs[i]:04d}"
    #             cut_id = result['meta'].get('id', candidate_id)
    #             samples = result['samples']
    #             sampling_rate = result['sampling_rate']
    #             savepath = str(Path(save_wav_dir) / f"{cut_id}_tts.wav")

    #             torchaudio.save(
    #                 savepath,
    #                 samples,
    #                 sample_rate=sampling_rate,
    #             )
    #             dew = {}
    #             dew['id'] = cut_id
    #             dew['hyp_audio'] = savepath
    #             dew['meta'] = result['meta']
    #             dew_writer.write(dew, flush=True)

    #     futures = []
    #     # just creates a background save worker
    #     with ThreadPoolExecutor(max_workers=1) as executor, dew_writer, tqdm(
    #         desc="Pipeline tts audios",
    #         total=guess_length,
    #         dynamic_ncols=True,
    #         leave=True,
    #     ) as pbar:
    #         cnt = 0
    #         for batch_result in pipline(iter(inputs)):
    #             bsz = len(batch_result)

    #             pbar.update(bsz)
    #             futures.append(
    #                 executor.submit(
    #                     _save_worker,
    #                     batch_result,
    #                     idxs=list(range(cnt, cnt + bsz)),
    #                 )
    #             )
    #             cnt += bsz
    #         # block
    #         for f in futures:
    #             f.result()

    def run_export(self):
        export_args = ExportArgs(**self.sub_args)
        repo: Path = export_args.outdir
        repo.mkdir(parents=True, exist_ok=True)
        save_load_helper = self.model.save_load_helper
        save_load_helper.save_to(repo, self.model, (self.tokenizer, self.extractor))
        logger.info(f"Save e2tts done, outdir: {repo}")


single_torch_nj()
if __name__ == "__main__":
    parser = TTSE2.get_dummy_parser()
    parser = TTSE2.setup_parser(parser)
    args = parser.parse_args()

    logger.info(f"Got parsed args: \n{parser.dump(args.clone())}", ranks=[0])

    TTSE2.run_from_args(args, parser)
