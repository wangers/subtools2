#!/bin/bash

# Please refer to http://cnceleb.org/ for more info




#Evaluation
task1_eval_enroll=eval_enroll
task1_eval_test=eval_test

stage=3
endstage=3

if [[ $stage -le 1 && 1 -le $endstage ]];then
	### Start
	# A example of getting Track2_dev's wav.scp and utt2spk etc. 
	echo "start"
	#eval_enroll
	data_path=/tsdata/xmuspeech/cn-celeb/CN-Celeb_wav/eval
	data=/tsdata1/ldx/egrecho/recipes/cnsrc/data/$task1_eval_enroll
	mkdir -p $data
	awk -v data_path=$data_path '{print $2,data_path"/"$2}' $data_path/lists/enroll.lst > $data/wav.scp
    awk '{print $2,$1}' $data_path/lists/enroll.lst > $data/utt2spk
	
	#eval_test
	data_path=/tsdata/xmuspeech/cn-celeb/CN-Celeb_wav/eval 
	data=/tsdata1/ldx/egrecho/recipes/cnsrc/data/$task1_eval_test
	mkdir -p $data
    awk -v data_path=$data_path '{print $1,data_path"/"$1}' $data_path/lists/test.lst > $data/wav.scp
    awk '{print $1,$1}' $data_path/lists/test.lst > $data/utt2spk
	
	# Fixed dir and make sure that the various files in a data directory are correctly sorted and filtered
	for x in $task1_eval_enroll $task1_eval_test;do
		scripts/kaldi_utils/fix_data_dir.sh data/$x
	done
fi

if [[ $stage -le 2 && 2 -le $endstage ]];then

	# generate egs.
	egrecho kd2egs data/cnsrc_train exp/egs/cnsrc_train
		

fi

if [[ $stage -le 3 && 3 -le $endstage ]];then
	## Pytorch x-vector model training
	egrecho train-asv -c config/train_template.yaml

fi

if [[ $stage -le 4 && 4 -le $endstage ]];then
	## ### x-vector extracting
	egrecho 
fi

if [[ $stage -le 5 && 5 -le $endstage ]];then
	## ### Back-end scoring
	exp=exp/SEResnet34_am_train_fbank81/near_epoch_6
	subtools/recipe/cnsrc/sv/scoreSets_sv.sh --eval false --vectordir $exp --prefix fbank_81  --enrollset $task1_enroll --testset $task1_test --trials data/fbank_81/eval_test/trials/trials.lst
fi

# ### All done ###
