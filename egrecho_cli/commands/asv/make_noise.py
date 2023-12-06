# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-9)

from pathlib import Path
from typing import Optional

from egrecho.core.parser import ActionConfigFile, BaseCommand, CommonParser
from egrecho.data.datasets.audio.augments.base import NoiseSet
from egrecho.utils.logging import get_logger
from egrecho_cli.register import register_command

logger = get_logger()

DESCRIPTION = "Prepare the noise set for speech augment."


@register_command(name="mknoise", aliases=["make_noise"], help=DESCRIPTION)
class MakeNoiseSet(BaseCommand):
    """
    Make general noise data base.
    """

    @classmethod
    def get_dummy_parser(cls) -> CommonParser:
        return CommonParser(description=DESCRIPTION)

    @classmethod
    def setup_parser(cls, parser: CommonParser):
        parser.add_argument(
            "-c",
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file yaml format.",
        )
        parser.add_argument(
            "wave_scp",
            type=str,
            help="file contains a list of tuples formed as (utt_key, utt_path) or (utt_path).",
        )
        parser.add_argument("output_db", type=str, help="Save noise database.")

        parser.add_argument(
            "--db-type",
            type=str,
            default="lmdb",
            help="Noise database type, supports hdf5 and lmdb now, lmdb is much faster.",
        )
        parser.add_argument("--mode", type=str, default="w", help="file handler mode.")
        parser.add_argument(
            "--max-length",
            type=float,
            help="The maximum length in seconds. Waveforms longer "
            "than this will be cut into pieces.\n"
            "Default is None means save the whole utts.",
        )
        parser.add_argument(
            "--resample", type=int, default=16000, help="resample in save stage."
        )
        parser.add_argument("--nj", type=int, default=1, help="num of jobs.")
        parser.add_argument(
            "--record-csv",
            type=str,
            help="If not None, given a path to record the added keys of this operation.",
        )

        return parser

    @staticmethod
    def run_from_args(args, parser: Optional[CommonParser] = None):
        exec = MakeNoiseSet(args)
        exec.run()

    def __init__(self, args):
        self.max_length = args.max_length
        if self.max_length is not None:
            assert self.max_length > 0.2, "Cut utts to too short clips is not pleased."

        self.wave_scp = Path(args.wave_scp)
        self.output_db = Path(args.output_db)
        (self.output_db.parent).mkdir(parents=True, exist_ok=True)

        self.record_csv = args.record_csv
        self.resample = args.resample
        self.nj = args.nj
        self.db_type = args.db_type
        self.mode = args.mode

    def run(self):
        db_cls = NoiseSet.get_register_class(self.db_type)

        items = []
        with self.wave_scp.open(encoding="utf8") as f:
            for line in f:
                attr = line.strip().split()
                assert len(attr) == 2 or len(attr) == 1
                items.append(line.strip().split())

        db_cls.create_db(
            self.output_db,
            items,
            mode=self.mode,
            max_length=self.max_length,
            resample=self.resample,
            nj=self.nj,
            record_csv=self.record_csv,
        )
        logger.info(f"Make noises from {self.wave_scp} to {self.output_db} done")


if __name__ == "__main__":
    pass
