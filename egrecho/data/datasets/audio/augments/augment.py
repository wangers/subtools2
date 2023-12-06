# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-02-11)

import copy
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import torch

from egrecho.data.datasets.audio.augments.base import (
    BaseSpeechAugmentConfig,
    SignalPerturb,
)
from egrecho.utils.common import alt_none
from egrecho.utils.io import ConfigFileMixin, resolve_file


@dataclass
class ASVSpeechAgugmentConfig(BaseSpeechAugmentConfig):
    """
    This config aims to build a dict to initiate a ``SpeechAugment`` pipeline,
    it is designed to ASV task, supports batch-wise forward.

    As `wave_drop` with additive noise perturb in a same chain performs not well in sv task,
    we build a separate chain for `wave_drop`. The total pipline has various augment.
    include: `tempo`, `rir`, `sim_rir`, `back_ground noise`, `point noise` and `babble`.

    The signal data flow is showed as below::

                                    +-------------+
    #                               |  waveforms  |
    #                               +------+------+
    #                                      |<------------ p
    #           +-----------------+--------+-----------------------+
    #           |                 |                                |
    #           |                 |                                |
    #           |                 |                                |
    #           |                 |                                |
    #    +------v------+   +------v------+                  +------v------+
    #    |    tempo    |   |     rir     |                  |     rir     |---> p=0.5
    #    +------+------+   +------+------+                  +------+------+
    #           |                 |                                |
    #           |                 |               +----------------|---------------+         +--------------+
    #  ---------+-----------------+---------------|----------------+---------------|-------->|random_weight |
    #           |                 |               |                |               |         +--------------+
    #           |                 |        +------v------+  +------v-----+  +------v-----+
    #           |                 |        |   bg_noise  |  |    noise   |  |    babble  |
    #           |                 |        +-------------+  +------+-----+  +------------+
    #           |                 |                                |
    #           |                 |                                |
    #           |                 |                                |
    #    +------v------+   +------v------+                  +------v------+
    #    |  wave_drop  |   |    tempo    |                  |    tempo    |
    #    +-------------+   +-------------+                  +-------------+


    `random_weight` in each chain control the prob of that pipeline, in desired mode, random weights
    distributed evenly, the right parts sums to `3.0` and the left part is `1 + 1`

    Args:
        init_p (float, defaults to `0.5`):
            the total entry prob.
        db_dir (str):
            outer noise dbs' parent dir.
        rir_fname (str, defaults to `"rir.lmdb"`):
            real rir signal fname.
        sim_rir_prob (float, defaults to `0`):
            reverb combines sim and real perturb, in other words, `1-sim_rir_prob` is the real signal perturb.
            so if reverb is needed but `rir_fname` is None, this param must set to 1.
        sim_rir_max_D (float, defaults to `36`):
            the maximum range of sample distance between the sound sources and the receiver.
        sim_rir_max_R (float, defaults to `1.6`):
            the maximum range of ratio between the volume and the total surface area of the room:

                - 1.2, means a room with length, width and height of 12 m, 12 m and 4 m.
                - 1.5, length, width and height: 24 m, 24 m and 4 m
                - 1.6, length, width and height: 36 m, 36 m and 4 m
                - 2.0, length, width and height: 48 m, 36 m and 5 m

        rir_weight (float, defaults to `1.0`):
            random weight rel to other chain perturbs (e.g., the figure above has three chains.)
        rir_ahead_noise_prob (float, defaults to `0.5`):
            rir prob ahead additive noise perturbs (the right part of above figure).
        noise_fname:
            point noise source file.
        noise_weight (float, defaults to `1.0`):
            will added to the right part of above figure.
        bg_noise_fname:
            back ground noise with a higher snr, maybe is continues music.
        bg_noise_weight:
            same as other weigths.
        babble_fname:
            mix multi speech to simulate babble noise.
        babble_weight:
            same as other weights.
        babble_mix_num (tuple, defaults to `(3, 7)`):
            mix num
        tempo_factors (tuple, defaults to `(0.95, 1.0, 1.05)`):
            tempo-based speed perturb, keep pitch unchanged.
        wave_drop (bool, defaults to `True`):
            whether applys signal level drop out, this is much more time consuming.
        drop_time_count (tuple, defaults to `(0, 0)`):
            Time-level drop times, each time drops a random seconds (s) in range of `[0.065, 0.125]`.
            and needs to set (e.g., `(0, 4)`), to open it.

            Inside freq-level drop (defaults to `(0, 3)`) is not expose.
        drop_weight:
            same as others.
        db_type (str, defaults to `None`):
            database type of source real audios (noise or rir). if None, will auto detect according to
            provided suffix of path (e.g., file name with `"xxx.lmdb"` will match lmdb).
        sample_rate (int, defaults to `16000`):
            Some perturbs need configure sample_rate (e.g., `wav_drop` need it to).
        extra_perturb: Optional[Dict]
            config dict for custom random perturb, will be added as a new chain in above figure.
        extra_weight: Optional[float]
    """

    name: ClassVar[str] = "asv_speechaug"

    db_dir: Optional[Path] = None
    # global prob.
    init_p: float = 0.5

    rir_fname: str = "rir.lmdb"
    sim_rir_prob: float = 0.0
    sim_rir_max_D: float = 36
    sim_rir_max_R: float = 1.6
    rir_weight: Optional[float] = None

    rir_ahead_noise_prob: Optional[float] = None

    noise_fname: str = "musan_noise.lmdb"
    noise_weight: Optional[float] = None

    bg_noise_fname: str = "musan_music.lmdb"
    bg_noise_weight: Optional[float] = None

    babble_fname: str = "musan_speech.lmdb"
    babble_weight: Optional[float] = None
    babble_mix_num: Optional[Tuple[int, int]] = None

    tempo_factors: Optional[Tuple[float, ...]] = field(
        default_factory=lambda: (0.95, 1.0, 1.05)
    )

    wave_drop: bool = False
    drop_time_count: Optional[Tuple[int, int]] = None
    drop_weight: Optional[float] = None

    db_type: Optional[str] = None
    sample_rate: int = 16_000

    extra_perturb: Optional[Dict] = None
    extra_weight: Optional[float] = None

    def __post_init__(self):
        self.random_lst = []
        self.random_weight = []

        # each chian will contains a tempo perturb.
        tempo = dict(
            name="speed",
            method="tempo",
            sample_rate=self.sample_rate,
            resize_shape=True,
            factors=self.tempo_factors,
        )

        # wave_drop and additive noise should not be in a chain, seperate it
        if self.wave_drop:
            wave_drop = {
                "name": "wave_drop",
                "sample_rate": self.sample_rate,
                "drop_time_count": alt_none(self.drop_time_count, (0, 0)),
            }
            drop_chain = dict(name="chain_perturb", perturbs=[tempo, wave_drop])
            self.random_lst.append(drop_chain)
            self.random_weight.append(alt_none(self.drop_weight, 1.0))

        has_rir = False
        sim_prob = alt_none(self.sim_rir_prob, 0.0)
        if self.rir_fname or sim_prob > 0:
            if self.rir_fname is not None:
                rir_file = resolve_file(self.rir_fname, base_path=self.db_dir)
            rir = dict(
                name="reverb",
                noise_db=rir_file,
                noise_db_type=self.db_type,
                sim_prob=sim_prob,
                sim_rir_conf=dict(
                    sample_rate=self.sample_rate,
                    max_D=alt_none(self.sim_rir_max_D, 36),
                    max_R=alt_none(self.sim_rir_max_R, 1.6),
                ),
            )
            rir_chain = dict(name="chain_perturb", perturbs=[rir, tempo])
            self.random_lst.append(rir_chain)
            self.random_weight.append(alt_none(self.rir_weight, 1.0))
            has_rir = True

        # build additive noises, compose in one RandomPerturb
        add_noises = []
        add_noises_weight = []
        if self.noise_fname:
            noise_path = resolve_file(self.noise_fname, base_path=self.db_dir)
            noise = dict(
                name="mix",
                noise_db_type=self.db_type,
                noise_db=noise_path,
                snr=(0, 15),
                noise_max_len=15,
            )
            add_noises.append(noise)
            add_noises_weight.append(alt_none(self.noise_weight, 1.0))

        if self.bg_noise_fname:
            bg_noise_fname = resolve_file(self.bg_noise_fname, base_path=self.db_dir)
            bg_noise = dict(
                name="mix",
                noise_db=bg_noise_fname,
                noise_db_type=self.db_type,
                snr=(5, 15),
                noise_max_len=15,
            )
            add_noises.append(bg_noise)
            add_noises_weight.append(alt_none(self.bg_noise_weight, 1.0))

        if self.babble_fname:
            babble_fname = resolve_file(self.babble_fname, base_path=self.db_dir)
            babble = dict(
                name="mix",
                noise_db=babble_fname,
                noise_db_type=self.db_type,
                mix_num=alt_none(self.babble_mix_num, (3, 7)),
                snr=(13, 20),
                noise_max_len=15,
            )

            add_noises.append(babble)
            add_noises_weight.append(alt_none(self.babble_weight, 1.0))

        # link rir, additive noises and tempo, compose one envrionmental perturb chain.
        if len(add_noises) > 0:
            env_perturbs_chain = []
            add_noises = dict(
                name="random_perturb",
                perturbs=add_noises,
                random_weight=add_noises_weight,
            )
            if has_rir:
                ahead_rir = copy.deepcopy(rir)
                ahead_rir.update(init_p=alt_none(self.rir_ahead_noise_prob, 0.5))
                env_perturbs_chain.append(ahead_rir)

            env_perturbs_chain += [add_noises, tempo]
            env_chain = dict(name="chain_perturb", perturbs=env_perturbs_chain)

            env_perturb_weight = sum(add_noises_weight)
            self.random_lst.append(env_chain)
            self.random_weight.append(env_perturb_weight)

        if self.extra_perturb:
            self.random_lst.append(self.extra_perturb)
            self.random_weight.append(alt_none(self.extra_weight, 1.0))

        # Just one tempo perturb.
        if len(self.random_lst) == 0:
            self.random_lst.append(tempo)
            self.random_weight.append(1.0)

    @property
    def perturbs(self) -> Dict[str, Any]:
        return dict(
            init_p=self.init_p,
            name="random_perturb",
            perturbs=self.random_lst,
            random_weight=self.random_weight,
        )


class SpeechAgugment(torch.nn.Module, ConfigFileMixin):
    def __init__(
        self,
        conf: Optional[Union[Dict, BaseSpeechAugmentConfig, SignalPerturb]],
    ):
        super().__init__()
        self.conf = conf
        if isinstance(conf, dict):
            try:
                perturb_dict = BaseSpeechAugmentConfig.from_dict(conf)

                self.augment = SignalPerturb.from_dict(perturb_dict.perturbs)
            except Exception as e:
                warnings.warn(
                    f"{e}\nFailed to get configuration from BaseSpeechAugmentConfig, "
                    f"try to treat the dict as Signalperturb."
                )
                self.augment = SignalPerturb.from_dict(conf)
                self.conf = self.augment.to_dict()
        else:
            if isinstance(conf, BaseSpeechAugmentConfig):
                self.augment = SignalPerturb.from_dict(conf.perturbs)
            elif isinstance(conf, SignalPerturb):
                self.augment = conf
            else:
                raise ValueError(
                    f"When conf ({conf}) is not dict, it should be instance of either `BaseSpeechAugmentConfig` or `SignalPerturb`."
                )
            self.conf = conf.to_dict()

    def forward(
        self,
        samples: torch.Tensor = None,
        lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        output_dict = self.augment(
            samples=samples,
            lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
            sample_rate=sample_rate,
            **kwargs,
        )

        return output_dict

    def to_dict(self):
        return self.conf

    @classmethod
    def from_dict(cls, dict):
        return cls(dict)
