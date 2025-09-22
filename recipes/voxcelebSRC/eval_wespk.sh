#!/bin/bash

# Copyright xmuspeech (Author: Leo 2023-11)

set -euo pipefail
stage=-1
endstage=6

 . scripts/parse_options.sh || exit 1


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

################ Download & Save a wespk model##########################
########################################################################
# from egrecho.models.wespeaker_sv.model import WeSpkModel,WeSpkSVConfig
# m = WeSpkModel(WeSpkSVConfig('vblinkp'))
# m.save_to('exp/wespk_vblinkp')
#########################################################################


data_type=shard
exp='exp/wespk_vblinkp'  # wespk_vblinkp wespk_vblinkf wespk_resnet221_en
test_egs=exp/egs/voxceleb1_${data_type}/egs-shards.jsonl
ckpt=model_weight.ckpt
if [ $stage -le 0 ] && [ $endstage -ge 0 ]; then
  echo "extract testset."
  egrecho extract-embed -c config/extract.yaml \
    --sub_outdir vox1 \
    --data.data_type ${data_type} \
    --pipeline.checkpoint ${ckpt} \
    --pipeline.feature_config config/wespk_feat.yaml \
    ${test_egs} ${exp}


fi

scp_dir=
extract_outdir_file=${exp}/extract_outdir.last  # last model's embed location
skip_mean=false  # can set to skip compute mean vec if that has already been prepared
[ -f $extract_outdir_file ] && scp_dir=${extract_outdir_file}
if [ $stage -le 1 ] && [ $endstage -ge 1 ]; then
  log "stage 3: score."
  egrecho score-sv -c config/score.yaml \
    --skip_mean $skip_mean \
    --submean false \
    --score_norm false \
    ${scp_dir:+--scp_dir $scp_dir}
fi
