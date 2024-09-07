# As the excellent work of lhotse for audio data preparation and
# for compatibility reasons. we borrowed this module from lhotse as kaldi feature extractor:
#   https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/kaldi

from egrecho.data.features.lhotse_kaldi.extractor import (
    FEATURE_EXTRACTORS,
    Fbank,
    FbankConfig,
    LhotseFeat,
    LogSpectrogram,
    LogSpectrogramConfig,
    Mfcc,
    MfccConfig,
)
