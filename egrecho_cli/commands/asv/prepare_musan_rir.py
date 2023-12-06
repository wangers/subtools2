# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-9)

from pathlib import Path
from typing import Optional

from egrecho.core.parser import BaseCommand, CommonParser
from egrecho.data.datasets.audio.augments.base import NoiseSet
from egrecho.utils.logging import get_logger
from egrecho_cli.register import register_command

logger = get_logger()

DESCRIPTION = "Prepare the openrir and musan dataset for speech augment."


@register_command(name="mkmr", aliases=["make_musan_rir"], help=DESCRIPTION)
class MakeMusanRIR(BaseCommand):
    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_cfg_flag()
        parser.add_argument(
            "output_folder",
            type=str,
            help="Save noise database to this folder\n"
            "contains: rir, music, speech, noise.",
        )

        parser.add_argument(
            "--db-type",
            type=str,
            default="lmdb",
            help="Noise database type, supports hdf5 and lmdb now, lmdb is much faster.",
        )
        parser.add_argument(
            "--openrir-folder",
            type=str,
            default="./data",
            help="where has openslr rir.",
        )
        parser.add_argument(
            "--musan-folder",
            type=str,
            default="./data",
            help="where has openslr musan.",
        )
        parser.add_argument(
            "--resample", type=int, default=16000, help="resample in save stage."
        )
        parser.add_argument("--nj", type=int, default=4, help="num of jobs.")

        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        exec = MakeMusanRIR(args)
        exec.run()

    def __init__(self, args):
        rir_dir = Path(args.openrir_folder) / "RIRS_NOISES"
        musan_dir = Path(args.musan_folder) / "musan"
        self.output_folder = Path(args.output_folder)

        assert rir_dir.is_dir(), f"{rir_dir} is not exist, please download it."
        assert musan_dir.is_dir(), f"{musan_dir} is not exist, please download it."

        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.rir_dir = rir_dir
        self.musan_dir = musan_dir
        self.output_folder = Path(args.output_folder)
        self.resample = args.resample
        self.nj = args.nj
        self.db_type = args.db_type

    def run(self):
        db_cls = NoiseSet.get_register_class(self.db_type)

        musan_parts = ["speech", "music", "noise"]
        for part in musan_parts:
            wave_items = []
            for file in (self.musan_dir / part).rglob("*.wav"):
                wave_items.append((file.stem, file))

            max_length = 20.0 if part == "music" else None
            db_cls.create_db(
                self.output_folder / f"musan_{part}",
                wave_items,
                max_length=max_length,
                resample=self.resample,
                nj=self.nj,
            )

        rir_parts = ["real_rirs_isotropic_noises", "simulated_rirs"]
        wave_items = []
        for part in rir_parts:
            if part == "real_rirs_isotropic_noises":
                for file in (self.rir_dir / part).rglob("*.wav"):
                    wave_items.append((file.stem, file))
            else:
                for room in ["large", "small", "medium"]:
                    for file in (self.rir_dir / part / f"{room}room").rglob("*.wav"):
                        wave_items.append((f"{room}-{file.stem}", file))
        db_cls.create_db(
            self.output_folder / "rir",
            wave_items,
            resample=self.resample,
            nj=self.nj,
        )
        logger.info(f"Make musan and rir to {self.output_folder} done")


if __name__ == "__main__":
    pass
