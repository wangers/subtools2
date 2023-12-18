#!/bin/bash

# Copyright xmuspeech (Author: Leo 2023-11)

set -euo pipefail
stage=-1
endstage=2

exp=./exp/ecapa1024

 . scripts/parse_options.sh || exit 1


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $endstage -ge 0 ]; then
  log "stage 0: train, use config config/train.yaml."
  egrecho train-asv -c config/train.yaml
fi


data_type=shard
test_egs=exp/egs/voxceleb1_${data_type}/egs-shards.jsonl
train_egs=exp/egs/voxceleb2_dev_${data_type}/egs-shards-train.jsonl
if [ $stage -le 1 ] && [ $endstage -ge 1 ]; then
  log "stage 1: extract embedding."
  echo "extract testset."
  egrecho extract-embed -c config/extract.yaml \
    --sub_outdir vox1 \
    --data.data_type ${data_type} \
    ${test_egs} ${exp}

  echo "extract trainset."
  egrecho extract-embed -c config/extract.yaml \
    --sub_outdir vox2_train \
    --data.data_type ${data_type} \
    ${train_egs} ${exp}
fi


scp_dir=${exp}/version_0/last_ckpt
extract_outdir_file=${exp}/last_extract_outdir # last model's embed location
skip_mean=false
[ -f $extract_outdir_file ] && scp_dir=${extract_outdir_file}
if [ $stage -le 2 ] && [ $endstage -ge 2 ]; then
  log "stage 2: score."

  egrecho score -c config/score.yaml \
    --scp_dir ${scp_dir} \
    --skip_mean $skip_mean
fi
