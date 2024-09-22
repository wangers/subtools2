#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.

See the function `remove_short_and_long_utt()` in transducer_stateless/train.py
for usage.
"""

# (Author: Leo 2024-06-04)

from lhotse import load_manifest_lazy

TRUES = ("yes", "true", "t", "y", "1")
FALSES = ("no", "false", "f", "n", "0")


def if_continue() -> bool:
    print(f"#### Waiting for confirmation: y:{TRUES} / n:{FALSES}")

    # Read user input
    user_input = input()

    # Check if user wants to continue or not
    if user_input.lower() in TRUES:
        return True
    elif user_input.lower() in FALSES:
        return False
    else:
        raise ValueError(
            f"Truthy value expected: got {user_input} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def main(paths):
    if isinstance(paths, str):
        paths = [paths]
    for i, path in enumerate(paths):
        if i > 0:
            print(f"Do you want to go next? ({path})")
            if not if_continue():
                return

        cuts = load_manifest_lazy(path)
        cuts.describe()
        print(f"INFO {path}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 1:
        print(f"Usage: {sys.argv[0]} [<path1> ...]")
        print("Description: This script processes the given file paths.")
        print(f"Example: python script.py ./data/fbank/aishell_cuts_dev.jsonl.gz")
        sys.exit(1)

    paths = sys.argv[1:]
    paths = paths or "exp/egs/libritts/cuts_train.jsonl.gz"
    main(paths)

# python local/display_manifests.py exp/egs/libritts/cuts_train.jsonl.gz
# Cut statistics:
# ╒═══════════════════════════╤═══════════╕
# │ Cuts count:               │ 354779    │
# ├───────────────────────────┼───────────┤
# │ Total duration (hh:mm:ss) │ 555:09:36 │
# ├───────────────────────────┼───────────┤
# │ mean                      │ 5.6       │
# ├───────────────────────────┼───────────┤
# │ std                       │ 4.5       │
# ├───────────────────────────┼───────────┤
# │ min                       │ 0.1       │
# ├───────────────────────────┼───────────┤
# │ 25%                       │ 2.3       │
# ├───────────────────────────┼───────────┤
# │ 50%                       │ 4.3       │
# ├───────────────────────────┼───────────┤
# │ 75%                       │ 7.6       │
# ├───────────────────────────┼───────────┤
# │ 99%                       │ 20.9      │
# ├───────────────────────────┼───────────┤
# │ 99.5%                     │ 23.1      │
# ├───────────────────────────┼───────────┤
# │ 99.9%                     │ 27.4      │
# ├───────────────────────────┼───────────┤
# │ max                       │ 43.9      │
# ├───────────────────────────┼───────────┤
# │ Recordings available:     │ 354779    │
# ├───────────────────────────┼───────────┤
# │ Features available:       │ 354779    │
# ├───────────────────────────┼───────────┤
# │ Supervisions available:   │ 354779    │
# ╘═══════════════════════════╧═══════════╛
