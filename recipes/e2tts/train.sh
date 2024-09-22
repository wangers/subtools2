#!/bin/bash

# Author: Leo 202406

set -euo pipefail
stage=0
endstage=2

 . scripts/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

### Start
if [ $stage -le 0 ] && [ $endstage -ge 0 ]; then
  log "stage 0: train e2tts."
  egrecho train-e2 -c config/e2tts.yaml
fi



if [ $stage -le 2 ] && [ $endstage -ge 2 ]; then
  log "stage 2: tts a demo"
  egrecho tts-e2 \
    -c config/tts_e2.yaml \
    demo
fi
