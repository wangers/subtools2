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
endstage=2

dl_dir=/work/ldx
dataset_parts="--dataset-parts all"  # all

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
    lhotse download libritts ${dataset_parts} $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $endstage -ge 1 ]; then
  log "Stage 1: Prepare LibriTTS lhotse manifest"
  # We assume that you have downloaded the LibriTTS corpus
  # to $dl_dir/LibriTTS
  lhotse_egsdir=data/libritts/lhotse_egs
  lhotse_nj=16
  mkdir -p $lhotse_egsdir
  if [ ! -e $lhotse_egsdir/.libritts.done ]; then
    lhotse prepare libritts ${dataset_parts} -j $lhotse_nj $dl_dir/LibriTTS $lhotse_egsdir
    touch $lhotse_egsdir/.libritts.done
  fi
fi
