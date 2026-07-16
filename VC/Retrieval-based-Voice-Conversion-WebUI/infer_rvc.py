import os
import torch
import soundfile as sf

from infer.modules.vc.modules import VC


device = "cuda:0"

vc = VC(
    device=device
)


model_path = "weights/speaker_ali.pth"


# Load model

vc.get_vc(
    model_path
)


input_audio = "input.wav"

output_audio = "output.wav"


# Conversion

vc.vc_single(
    sid=0,
    input_audio_path=input_audio,
    f0_up_key=0,
    f0_method="rmvpe",
    file_index="logs/speaker_ali.index",
    file_index2="",
    index_rate=0.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
    protect=0.33
)


print("conversion finished")
