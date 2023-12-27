#!/bin/bash

# Copyright xmuspeech (Author: Leo 2023-11)

set -euo pipefail
stage=-1
endstage=6

exp=./exp/ecapa1024
avg_num=5
do_lm=true

 . scripts/parse_options.sh || exit 1


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $endstage -ge 0 ]; then
  log "stage 0: train."
  egrecho train-asv -c config/train_ecapa.yaml --save_dir ${exp}
fi


if [ $stage -le 1 ] && [ $endstage -ge 1 ] && [ $avg_num -ge 1 ]; then
  log "stage 1: average best models."
  egrecho avg-best --version version \
    ${exp} ${avg_num}
fi


data_type=shard
test_egs=exp/egs/voxceleb1_${data_type}/egs-shards.jsonl
train_egs=exp/egs/voxceleb2_dev_${data_type}/egs-shards-train.jsonl

ckpt=last.ckpt
if [ $avg_num -ge 1 ]; then
  ckpt=average_${avg_num}.ckpt
fi

if [ $stage -le 2 ] && [ $endstage -ge 2 ]; then
  log "stage 2: extract embedding."
  echo "extract testset."
  egrecho extract-embed -c config/extract.yaml \
    --sub_outdir vox1 \
    --data.data_type ${data_type} \
    --pipeline.checkpoint ${ckpt} \
    ${test_egs} ${exp}

  echo "extract trainset."
  egrecho extract-embed -c config/extract.yaml \
    --sub_outdir vox2_train \
    --data.data_type ${data_type} \
    --pipeline.checkpoint ${ckpt} \
    ${train_egs} ${exp}
fi


scp_dir=
extract_outdir_file=${exp}/extract_outdir.last  # last model's embed location
skip_mean=false  # can set to skip compute mean vec if that has already been prepared
[ -f $extract_outdir_file ] && scp_dir=${extract_outdir_file}
if [ $stage -le 3 ] && [ $endstage -ge 3 ]; then
  log "stage 3: score."
  egrecho score-sv -c config/score.yaml \
    --skip_mean $skip_mean \
    ${scp_dir:+--scp_dir $scp_dir}
fi


if [ "$do_lm" == "true" ]; then

  if [ $stage -le 4 ] && [ $endstage -ge 4 ]; then
    log "stage 4: Large margin finetune."
    egrecho train-asv -c config/train_ecapa_lm_tune.yaml \
      --save_dir ${exp}_lm \
      --init_weight_params.dirpath ${exp} \
      --init_weight_params.checkpoint ${ckpt}
  fi


  if [ $stage -le 5 ] && [ $endstage -ge 5 ]; then
    log "stage 5: extract fine-tuned embedding."
    echo "extract testset."
    egrecho extract-embed -c config/extract.yaml \
      --sub_outdir vox1 \
      --data.data_type ${data_type} \
      --pipeline.checkpoint last.ckpt \
      ${test_egs} ${exp}_lm

    echo "extract trainset."
    egrecho extract-embed -c config/extract.yaml \
      --sub_outdir vox2_train \
      --data.data_type ${data_type} \
      --pipeline.checkpoint last.ckpt \
      ${train_egs} ${exp}_lm
  fi


  scp_dir=
  extract_outdir_file=${exp}_lm/extract_outdir.last  # last model's embed location
  skip_mean=false  # can set to skip compute mean vec if that has already been prepared
  [ -f $extract_outdir_file ] && scp_dir=${extract_outdir_file}
  if [ $stage -le 6 ] && [ $endstage -ge 6 ]; then
    log "stage 6: score fine-tuned model."

    egrecho score-sv -c config/score.yaml \
      --skip_mean $skip_mean \
      ${scp_dir:+--scp_dir $scp_dir}
  fi

fi
