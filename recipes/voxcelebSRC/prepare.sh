#!/bin/bash

# Copyright xmuspeech (Author: Snowdar 2020-05-03)
#					  (Author: Leo 2023-11)

### Reference
##  Paper: Chung, Joon Son, Arsha Nagrani, and Andrew Zisserman. 2018. “Voxceleb2: Deep Speaker Recognition.” 
##         ArXiv Preprint ArXiv:1806.05622.
##  Data downloading: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

### Dataset info (The voxceleb2.test is not used here and the avg length of utterances in both trainset and testset is about 8s.)
##  Only trainset: voxceleb2_dev = voxceleb2.dev (num_utts: 1092009, num_speakers: 5994, duration: >2300h)
##
##  Only testset: voxceleb1 = voxceleb1.dev + voxceleb1.test (num_utts: 153516, num_speakers: 1251)

### Task info
##  Original task of testset: voxceleb1-O 
##       num_utts of enroll: 4715, 
##       num_utts of test: 4713, 
##       total num_utts: just use 4715 from 4874 testset, 
##       num_speakers: 40, 
##       num_trials: 37720 (clean:37611)
##
##  Extended task of testset: voxceleb1-E 
##       num_utts of enroll: 145375, 
##       num_utts of test: 142764, 
##       total num_utts: just use 145375 from 153516 testset, 
##       num_speakers: 1251, 
##       num_trials: 581480 (clean:401308)
##
##  Hard task of testset: voxceleb1-H 
##       num_utts of enroll: 138137, 
##       num_utts of test: 135637, 
##       total num_utts: just use 138137 from 153516 testset, 
##       num_speakers: 1190, 
##       num_trials: 552536 (clean:550894)


set -euo pipefail
stage=-1
endstage=2

vox_data=/data/voxceleb
prepare_musan_rir=true

 . scripts/parse_options.sh || exit 1


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


### Start 
if [ $stage -le 0 ] && [ $endstage -ge 0 ]; then
  log "stage 0: Prepare Kaldi set && trials from downloaded data."
  voxceleb1_path=$vox_data/voxceleb1/wav_b/
  voxceleb2_path=$vox_data/voxceleb2_wav/

  # [1] Prepare the data/voxceleb1_dev, data/voxceleb1_test and data/voxceleb2_dev.
  # ==> Make sure the audio datasets (voxceleb1, voxceleb2, RIRS and Musan) have been downloaded by yourself.
  prepare/make_voxceleb1_v2.pl $voxceleb1_path dev data/voxceleb1_dev
  prepare/make_voxceleb1_v2.pl $voxceleb1_path test data/voxceleb1_test
  prepare/make_voxceleb2.pl $voxceleb2_path dev data/voxceleb2_dev

  # [2] Combine testset voxceleb1 = voxceleb1_dev + voxceleb1_test
  scripts/kaldi_utils/combine_data.sh data/voxceleb1 data/voxceleb1_dev data/voxceleb1_test

  # [3] Get trials
  # ==> Make sure all trials are in data/voxceleb1.
  prepare/get_trials.sh --dir data/voxceleb1 --tasks "voxceleb1-O voxceleb1-E voxceleb1-H \
                                                        voxceleb1-O-clean voxceleb1-E-clean voxceleb1-H-clean"

  # [4] Get the clean copies of dataset which is labeled by a prefix.
  prefix=raw
  scripts/newCopyData.sh $prefix "voxceleb2_dev voxceleb1"
fi

musan_rir_dir=/data
lmdbout=/data/speech_aug
if [ $stage -le 1 ] && [ $endstage -ge 1 ]; then
  if [ "$prepare_musan_rir" == "true" ];then
    log "stage 1: Prepare musan_rir lmdb."
    egrecho mkmr \
    --openrir_folder ${musan_rir_dir} \
    --musan_folder ${musan_rir_dir} \
    -nj 8
    ${lmdbout}
  fi  
fi 

data_type=shard
shard_dir=/data/features/shards
if [ $stage -le 2 ] && [ $endstage -ge 2 ]; then
    log "stage 2: Prepare egs in exp/egs."
    echo "Prepare train."
    egrecho kd2egs -c config/kd2egs_for_train.yaml \
      --dumps.data_type ${data_type} \
      --dumps.shard_dir ${shard_dir}/voxceleb2_dev1k \
      data/raw/voxceleb2_dev exp/egs/voxceleb2_dev_${data_type}
    echo "Prepare test."
    egrecho kd2egs -c config/kd2egs_for_test.yaml \
      --dumps.data_type ${data_type} \
      --dumps.shard_dir ${shard_dir}/voxceleb1 \
      data/raw/voxceleb1 exp/egs/voxceleb1_${data_type}
fi 


