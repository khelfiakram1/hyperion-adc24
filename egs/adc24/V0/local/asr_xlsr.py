from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
transcriptions = model.transcribe("/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/exp/adi17/dev_proc_audio_no_sil/YFqhxkhEkfk_190965-192209-ara-pal.flac")
print(((transcriptions[0])['transcription']))   

# 
# 
#   
# import soundfile as sf
# import speech_recognition as sr
# from gtts import gTTS
# audio = "/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/exp/adi17/dev_proc_audio_no_sil/Wi4maIrEJBo_000291-001281-ara-leb.flac"
# audio_data, sample_rate = sf.read(audio)
# audio_data_int16 = (audio_data * 32767).astype('int16')
# audio_data_for_sr = sr.AudioData(audio_data_int16.tobytes(), sample_rate, 2)
# r = sr.Recognizer()

# t = r.recognize_google(audio_data_for_sr, language ='ar-LB')

# with open("/home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/local/recognized_text_arabic.txt", "w", encoding="utf-8") as file:
#     file.write(t)
