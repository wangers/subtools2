#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

is_lora=false
model_name=whisper-medium
stage=-1
stop_stage=1

. scripts/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
if [ ${is_lora} == true ]; then
  train_prefix=lora
else
  train_prefix=full
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Finetune start."

  # 29500
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29500 \
    egrecho_inner/${train_prefix}.py \
    -c config/${train_prefix}.yaml \
    --manifest_dir data/fbank_whisper80 \
    --base_model /tsdata/wf/whisper_hf/openai/${model_name} \
    --output_dir exp/${model_name}/${train_prefix} \
    --max_cuts 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 100_000 \
    --use_compile false

  if [ ${is_lora} == true ]; then
    log "Merge lora."
    egrecho merge-lora-whisper --lora_model exp/${model_name}/${train_prefix}/checkpoint-final
  fi


fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Eval."
  if [ ${is_lora} == true ]; then
    ckpt=checkpoint-final-lora-merge
  else
    ckpt=checkpoint-final
  fi
  egrecho eval -c config/eval.yaml \
    --model_path ./exp/${model_name}/${train_prefix}/${ckpt} \
    --outdir exp/${model_name}/${train_prefix}/${ckpt}

fi
