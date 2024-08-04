#!/bin/bash

# Author: Leo 202406

# Eval the acoustic performace of tts model.
# For SV evaluation, suppose we have a trained speaker model. Please refer to ../voxcelebSRC/run.sh
# For whisper wer evaluaton, additional packages lists in requirements_eval.txt

set -euo pipefail
stage=-1
endstage=-1

svdir=/tsdata1/ldx/egrecho/exp/campp
avg_num=5
ckpt=last.ckpt  # average_5.ckpt
ttsdir=data/infer/libritts/valle/test

 . scripts/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

### Start
if [ $stage -le -1 ] && [ $endstage -ge -1 ]; then
  log "stage -1: tts infer with config/tts_valle.yaml"
  egrecho tts-ve \
    -c config/tts_valle.yaml \
    infer \
    --outdir ${ttsdir}
fi

if [ $stage -le 0 ] && [ $endstage -ge 0 ]; then
  log "stage 0: prepare metric files from ${ttsdir}."
  python egrecho_inner/prepare_acoustic_trials.py \
    --tts_egfile ${ttsdir}/egs_infer_tts.jsonl
fi

if [ $avg_num -ge 1 ]; then
  ckpt=average_${avg_num}.ckpt
fi
# Suppose we have a trained speaker model. Please refer to ../voxcelebSRC/run.sh
if [ $stage -le 1 ] && [ $endstage -ge 1 ]; then
  log "stage 1: extract embedding ${ttsdir}/egs_prompt_tts.jsonl"
  echo "extract testset."
  egrecho extract-embed \
    --outdir ${ttsdir}/sv \
    --pipeline.checkpoint ${ckpt} \
    --gpus 0,1 \
    --number_processes 16 \
    ${ttsdir}/egs_prompt_tts.jsonl ${svdir}
fi

if [ $stage -le 2 ] && [ $endstage -ge 2 ]; then
  log "stage 2: Scores similarity via ${ttsdir}/sv_prompt_trials.txt"
  egrecho score-sv \
    --submean false \
    --skip_metric true \
    --scp_dir ${ttsdir} \
    --score_norm false \
    sv/xvector.scp ${ttsdir}/sv_prompt_trials.txt

  if [ -f ${ttsdir}/sv_prompt_trials.txt.score ];then
    cat ${ttsdir}/sv_prompt_trials.txt.score \
      | awk '{sum+=$3} END {printf "SumNum = %d, AvgScore = %3.3f\n", NR, sum/NR}' \
      | tee ${ttsdir}/sv_prompt_trials.mean.score
  fi
fi
# Use whisper model to eval wer of asr.
model_name=whisper-large-v3
if [ $stage -le 3 ] && [ $endstage -ge 3 ]; then
  log "stage 3: Scores wer ${ttsdir}/egs_tts_metric.jsonl"
  # openai/whisper-large-v3
  python egrecho_inner/eval_wer.py \
    --model_id /tsdata/wf/whisper_hf/openai/${model_name} \
    --num_workers 8 \
    --batch_size 2 \
    --log_internal 10 \
    ${ttsdir}/egs_tts_metric.jsonl ${ttsdir}/${model_name}

  if [ -f ${ttsdir}/sv_prompt_trials.mean.score ];then
    echo "Review the precomputed sv similarity:"
    cat ${ttsdir}/sv_prompt_trials.mean.score
  fi
fi
