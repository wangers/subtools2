# bin/bash

export HF_ENDPOINT='https://hf-mirror.com'

localdir=/tsdata/wf/whisper_hf

# "openai/whisper-base openai/whisper-medium openai/whisper-large-v2 openai/whisper-large-v3"
repos="openai/whisper-base openai/whisper-medium openai/whisper-large-v2 openai/whisper-large-v3"
for repo in $repos; do
  huggingface-cli download $repo \
    --exclude *.h5 flax_model* *.safetensors \
    ${localdir:+--local-dir $localdir/$repo --local-dir-use-symlinks False}
done

## GIT_LFS_SKIP_SMUDGE=1 git clone
