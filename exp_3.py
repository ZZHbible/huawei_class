#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/25
# project = exp_3
import os
import wave as we
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

filename = './实验1-语音预处理/data/thchs30/train/A2_0.wav'
WAVE = we.Wave_read(filename)
# for item in enumerate(WAVE.getparams()):
#     print(item)
# 帧总数
a = WAVE.getparams().nframes
# 采样帧率
f = WAVE.getparams().framerate
sample_time = 1 / f  # 采样点的时间间隔
time = a * sample_time  # 声音信号的长度
sample_frequency, auto_sequence = wavfile.read(filename)
# print(auto_sequence, len(auto_sequence))  # 声音信号每一帧的“大小“
x_seq=np.arange(0,time,sample_time)
# print(x_seq,len(x_seq))

# print(WAVE.getparams())

# 查看wav文件波形序列
# plt.plot(x_seq,auto_sequence)
# plt.xlabel('time(s)')
# plt.show()

auto_path='./实验1-语音预处理/data/train/audio/'
pict_Path='./实验1-语音预处理/data/train/audio/'
sample=[]
if not os.path.exists(pict_Path):
    os.mkdir(pict_Path)
subFoldList=[]
for i in os.listdir(auto_path):
    if os.path.isdir(auto_path+i):
        subFoldList.append(i)
    else:
        if not os.path.exists(auto_path+i):
            os.mkdir(auto_path+i)
# print(subFoldList,len(subFoldList))

# 统计每个子文件夹语音文件数量
sample_auto=[]
total=0
for sub_dir in subFoldList:
    all_file=[file for file in os.listdir(auto_path+sub_dir) if file.endswith('.wav')]
    sample_auto.append(auto_path+sub_dir+'/'+all_file[0]) # 第一个文件
    total+=len(all_file)
#     print("文件夹:{} 有{}个.wav文件 ".format(sub_dir,len(all_file)))
# print("total have :{} .wav file".format(total))

# 构建频谱处理函数
def log_specgram(audio,sample_rate,window_size=20,step_size=10,eps=1e-10):
    nperseg=int(round(window_size*sample_rate/1e3))
    noverlap=int(round(step_size*sample_rate/1e3))
    freqs,_,spec=signal.spectrogram(audio,fs=sample_rate,window='hann',nperseg=nperseg,noverlap=noverlap,detrend=False)
    return freqs,np.log(spec.T.astype(np.float32)+eps)

fig=plt.figure(figsize=(20,20))
for i,filepath in enumerate(sample_auto[:16]):
    plt.subplot(4,4,i+1)
    label=filepath.split('/')[-2]
    plt.title(label)
    # print spectrogram
    samplerate,test_sound=wavfile.read(filepath)
    _,spectrogram=log_specgram(test_sound,samplerate)
    # plt.imshow(spectrogram.T,aspect='auto',origin='lower')
    # plt.axis('off')
# plt.show()

# 单样本多个频谱可视化
yes_samples=[auto_path+'yes/'+y for y in os.listdir(auto_path+'yes/')]
print(yes_samples)
fig=plt.figure(figsize=(20,20))
for i,file in enumerate(yes_samples):
    plt.subplot(5,5,i+1)
    label=file.split('/')[-2]
    samplerate,test_sound=wavfile.read(file)
    _,spectrogram=log_specgram(test_sound,samplerate)
    print(type(test_sound))
    print(type(spectrogram))
    # plt.imshow(spectrogram.T,aspect='auto',origin='lower')
    # plt.axis('off')
# plt.show()



