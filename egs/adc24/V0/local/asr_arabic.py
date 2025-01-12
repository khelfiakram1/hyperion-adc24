
from speechbrain.inference.ASR import WhisperASR

asr_model = WhisperASR.from_hparams(source="speechbrain/asr-whisper-large-v2-commonvoice-ar", savedir="pretrained_models/asr-whisper-large-v2-commonvoice-ar")
asr_model.transcribe_file("/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/exp/adi17/train_proc_audio_no_sil/-2ByhHYhyGk_095458-095878-ara-acm.flac")
