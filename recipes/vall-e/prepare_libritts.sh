#!/bin/bash

# Copyright xmuspeech (Author: Leo 2024-04)

### Reference
##  Paper: "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech", Heiga Zen, Viet Dang, Rob Clark,
##  Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu, arXiv, 2019
##  Data downloading: https://www.openslr.org/60/

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriTTS
#
# After downloading tar.gz files, you should extract them into dl_dir/LibriTTS.
# Ignoring *.tar.gz files, which you can download into anywhere, the structure of $dl_dir should look like below
#
# dl_dir
# ├── dev-clean.tar.gz
# ├── dev-other.tar.gz
# ├── LibriTTS
# │   ├── BOOKS.txt
# │   ├── CHAPTERS.txt
# │   ├── dev-clean
# │   ├── dev-other
# │   ├── eval_sentences10.tsv
# │   ├── LICENSE.txt
# │   ├── NOTE.txt
# │   ├── reader_book.tsv
# │   ├── README_librispeech.txt
# │   ├── README_libritts.txt
# │   ├── speakers.tsv
# │   ├── SPEAKERS.txt
# │   ├── test-clean
# │   ├── test-other
# │   ├── train-clean-100
# │   ├── train-clean-360
# │   └── train-other-500
# ├── test-clean.tar.gz
# ├── test-other.tar.gz
# ├── train-clean-100.tar.gz
# ├── train-clean-360.tar.gz
# └── train-other-500.tar.gz


set -euo pipefail
stage=1
endstage=10

dl_dir=/work/ldx

 . scripts/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

### Start
if [ $stage -le 0 ] && [ $endstage -ge 0 ]; then
  log "dl_dir: $dl_dir"
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LibriTTS,
  # you can create a symlink
  #
  #   ln -sfv /path/to/LibriTTS $dl_dir/LibriTTS
  #
  if [ ! -d $dl_dir/LibriTTS/dev-other ]; then
    # lhotse download libritts $dl_dir
    lhotse download libritts --dataset-parts all $dl_dir
  fi
fi

lhotse_egsdir=data/libritts/manifests
if [ $stage -le 1 ] && [ $endstage -ge 1 ]; then
  log "Stage 1: Prepare LibriTTS lhotse manifest"
  # We assume that you have downloaded the LibriTTS corpus
  # to $dl_dir/LibriTTS
  lhotse_nj=16
  mkdir -p $lhotse_egsdir
  if [ ! -e $lhotse_egsdir/.libritts.done ]; then
    lhotse prepare libritts ${dataset_parts} -j $lhotse_nj $dl_dir/LibriTTS $lhotse_egsdir
    touch $lhotse_egsdir/.libritts.done
  fi
fi

prefix=libritts
storage=/work/ldx/features/${prefix}/codes
parts='train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other,test-clean,test-other'
if [ $stage -le 2 ] && [ $endstage -ge 2 ]; then
    log "stage 2: Preprocess cuts."
    if [ ! -e data/${prefix}/tokenized/.libritts.tokenize.done ]; then
      echo 'Encodec...'
      python egrecho_inner/prepare_cuts.py encodec \
        --outdir $storage \
        --srcdir $lhotse_egsdir \
        --prefix $prefix \
        --device auto \
        --parts ${parts}

      echo 'Tokenize...'
      python egrecho_inner/prepare_cuts.py tokenize \
        --outdir $storage \
        --mvdir data/${prefix}/tokenized \
        --prefix $prefix \
        --parts ${parts}

      touch data/${prefix}/tokenized/.libritts.tokenize.done
    fi
fi

expegs=exp/egs/$prefix
if [ $stage -le 3 ] && [ $endstage -ge 3 ]; then
  log "Stage 3: Combine LibriTTS train/dev/test to $expegs"
  if [ ! -e ${expegs}/.libritts.train.done ]; then
    mkdir -p $expegs
    # train
    lhotse combine \
      data/${prefix}/tokenized/libritts_cuts_train-clean-100.jsonl.gz \
      data/${prefix}/tokenized/libritts_cuts_train-clean-360.jsonl.gz \
      data/${prefix}/tokenized/libritts_cuts_train-other-500.jsonl.gz \
      ${expegs}/cuts_train.jsonl.gz

    # dev
    lhotse copy \
      data/${prefix}/tokenized/libritts_cuts_dev-clean.jsonl.gz \
      ${expegs}/cuts_dev.jsonl.gz
    lhotse copy \
      data/${prefix}/tokenized/libritts_cuts_dev-other.jsonl.gz \
      ${expegs}/cuts_dev-other.jsonl.gz

    # test
    lhotse copy \
      data/${prefix}/tokenized/libritts_cuts_test-clean.jsonl.gz \
      ${expegs}/cuts_test.jsonl.gz
    lhotse copy \
      data/${prefix}/tokenized/libritts_cuts_test-other.jsonl.gz \
      ${expegs}/cuts_test-other.jsonl.gz
    touch ${expegs}/.libritts.train.done
  fi
fi


if [ $stage -le 4 ] && [ $endstage -ge 4 ]; then
  log "Stage 4: Generate token file"
  # We assume you have installed piper_phonemize and espnet_tts_frontend.
  # If not, please install them with:
  #   - piper_phonemize: refer to https://github.com/rhasspy/piper-phonemize,
  #                      could install the pre-built wheels from https://github.com/csukuangfj/piper-phonemize/releases/tag/2023.12.5
  if [ ! -e ${expegs}/tokens.txt ]; then
    python ./local/prepare_espeak_token_file.py --tokens ${expegs}/tokens.txt
  fi
fi

if [ $stage -le 5 ] && [ $endstage -ge 5 ]; then
  log "Stage 5: Generate prompt test dataset"

  python ./local/make_testdata.py libritts \
    --path_or_paths ${expegs}/cuts_test.jsonl.gz \
    --outdir ./data/prompt_test/libritts \
    --max_prompt_length 3.1 \
    --min_prompt_length 2.1 \
    --max_sample_len 10 \
    --min_sample_len 4

  log "Take first 200 to  ${expegs}/egs_test_prompt_200.jsonl "
  cat ./data/prompt_test/libritts/egs_test_prompt.jsonl | head -n 200 > \
    ${expegs}/egs_test_prompt_200.jsonl

fi

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
# CUT custom fields:
# - dataloading_info (in 354779 cuts)
# - tokens (in 354779 cuts)
# SUPERVISION custom fields:
# Speech duration statistics:
# ╒══════════════════════════════╤═══════════╤══════════════════════╕
# │ Total speech duration        │ 555:09:36 │ 100.00% of recording │
# ├──────────────────────────────┼───────────┼──────────────────────┤
# │ Total speaking time duration │ 555:09:36 │ 100.00% of recording │
# ├──────────────────────────────┼───────────┼──────────────────────┤
# │ Total silence duration       │ 00:00:01  │ 0.00% of recording   │
# ╘══════════════════════════════╧═══════════╧══════════════════════╛
# INFO exp/egs/libritts/cuts_train.jsonl.gz
