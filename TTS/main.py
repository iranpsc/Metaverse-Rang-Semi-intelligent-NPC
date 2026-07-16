#from TTS.api import TTS
#tts=TTS(model_path="./best_model_91323.pth",config_path="./config.json")
 

#tts.is_multi_lingual = False
#tts.is_multi_speaker = False

#tts.tts_to_file(".زندگی فقط یک بار است؛ از آن به خوبی استفاده کن",file_path='output.wav')


from TTS.utils.synthesizer import Synthesizer

synth = Synthesizer(
    tts_checkpoint="./best_model_91323.pth",
    tts_config_path="./config.json",
)

wav = synth.tts("زندگی فقط یک بار است؛ از آن به خوبی استفاده کن")

synth.save_wav(wav, "output.wav")


#from TTS.api import TTS

#tts = TTS(model_path="./best_model_91323.pth",config_path="./config.json"
#)

#print("loaded")
#print(tts.__dict__)
