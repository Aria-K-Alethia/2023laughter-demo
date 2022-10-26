import matplotlib.pyplot as plt
import pyworld as pw
import numpy as np
import soundfile as sf
import parselmouth as pm
from glob import glob

SIZE = 18

plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize

def compute_pitch(wav, sr, hop_length):
    pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=hop_length / sr * 1000)
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)
    
    return pitch

filenames = glob('prosody*.wav')
filenames.sort()
print(filenames)
#all_wavs = [sf.read(filename)[0] for filename in filenames]
all_wavs = [pm.Sound(filename) for filename in filenames]

# get f0 and draw
f0s = []
indices = [11, 0, 5, 10]
#indices = list(range(12))
all_wavs = [all_wavs[i] for i in indices]
labels = ['GT', 'Predicted', 'Low tension', 'High tension']

f = plt.figure(figsize=(9.84, 7.47))
count = 0
for i, wav in enumerate(all_wavs):
    pitch = wav.to_pitch(time_step=0.01, pitch_floor=70).smooth(10)
    f0 = pitch.selected_array['frequency']
    f0[f0 == 0] = np.nan
    plt.plot(f0, label=labels[count], linewidth=2)
    #plt.plot(f0, label=str(i), linewidth=2)
    count += 1
plt.ylabel('Pitch (Hz)')
plt.xlabel('Time')
plt.ylim(70, 210)
plt.xlim(left=0)
plt.legend()
# save file
f.savefig('contextual_prosody.png', dpi=300)