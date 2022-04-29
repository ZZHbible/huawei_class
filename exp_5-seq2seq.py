#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/26
# project = exp_5-seq2seq
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tensorflow.keras.layers import LSTM,Dense,Input
from tensorflow.keras.models import Model

auto_filename="./实验3-深度学习代码和数据/data/audio.wav"
label_filename='./实验3-深度学习代码和数据/data/label.txt'
def sparse_tuple_from(sequences,dtype=np.int64):
    indices=[]
    values=[]
    for i,seq in enumerate(sequences):
        indices.extend(zip([i]*len(seq),range(len(seq))))
        values.extend(seq)
    indices=np.array(indices,dtype=dtype)
    values=np.array(values)
    shape=np.array([len(sequences),np.array(indices).max(0)[1]+1],dtype=dtype)
    return indices,values,shape

def get_auto_feature():
    fs,audio=wav.read(auto_filename)
    inputs=mfcc(audio,samplerate=fs)
    # 对特征数据进行归一化，减去均值除以方差
    feature_inputs=np.array(inputs).reshape((1,inputs.shape[0],inputs.shape[1]))
    feature_inputs=(feature_inputs-np.mean(feature_inputs))/np.std(feature_inputs)
    feature_seq_len=[feature_inputs.shape[1]]
    return feature_inputs,feature_seq_len

feature_inputs,feature_seq_len=get_auto_feature()
def get_audio_label():
    with open(label_filename,'r') as f:
        line=f.readlines()[0].strip()
    labels=line.split()
    labels.insert(0,"<START>")
    labels.append("<END>")
    # 将列表转为稀疏三元组
    train_labels=sparse_tuple_from([labels])
    return labels,train_labels

line_labels,train_labels=get_audio_label()

# 神经网络参数设置
label_characters=list(set(line_labels))
INPUT_LENGTH=feature_inputs.shape[-2]
OUTPUT_LENGTH=train_labels[-1][-1]
INPUT_FEATURE_LENGTH=feature_inputs.shape[-1]
OUTPUT_FEATURE_LENGTH=len(label_characters)
N_UNITS=256
BATCH_SIZE=1
EPOCH=20
NUM_SAMPLES=1
labels_texts=[]
labels_texts.append(line_labels)

# 创建Seq2Seq模型
def create_model(n_input,n_output,n_units):
    # encoder
    encoder_input=Input(shape=(None,n_input))
    encoder=LSTM(n_units,return_state=True)
    _,encoder_h,encoder_c=encoder(encoder_input)
    encoder_state=[encoder_h,encoder_c]     # 保留下来的encoder末状态 为 decoder的初状态

    # decoder
    decoder_input=Input(shape=(None,n_output))
    decoder=LSTM(n_units,return_sequences=True,return_state=True)
    decoder_output,_,_=decoder(decoder_input,initial_state=encoder_state)
    decoder_dense=Dense(n_output,activation='softmax')
    decoder_output=decoder_dense(decoder_output)

    # 生成训练模型
    model=Model([encoder_input,decoder_input],decoder_output)

    # 推断模型 encoder
    encoder_infer=Model(encoder_input,encoder_state)

    # 推断模型 decoder
    decoder_state_input_h=Input(shape=(n_units,))
    decoder_state_input_c=Input(shape=(n_units,))
    decoder_state_input=[decoder_state_input_h,decoder_state_input_c]
    decoder_infer_output,decoder_infer_state_h,decoder_infer_state_c=decoder(decoder_input,initial_state=decoder_state_input)
    decoder_infer_state=[decoder_infer_state_h,decoder_infer_state_c] # 当前时刻的输出
    decoder_infer=Model([decoder_input]+decoder_state_input,[decoder_infer_output]+decoder_infer_state)

    return model,encoder_infer,decoder_infer

model_train,encoder_infer,decoder_infer=create_model(INPUT_FEATURE_LENGTH,OUTPUT_FEATURE_LENGTH,N_UNITS)
model_train.compile(optimizer='adam',loss='categorical_crossentropy')
model_train.summary()


# 配置训练数据
encoder_input=feature_inputs
decoder_input=np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))
decoder_output=np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))
label_dict={char:index for index,char in enumerate(label_characters)}
label_dict_reverse={index:char for index,char in enumerate(label_characters)}
for seq_index,seq in enumerate(labels_texts):
    for char_index,char in enumerate(seq):
        decoder_input[seq_index,char_index,label_dict[char]]=1.0
        if char_index > 0:
            decoder_output[seq_index,char_index-1,label_dict[char]]=1

model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH,validation_split=0)

ans=""

out=model_train.predict([encoder_input,decoder_input])
print(out.shape)
for sequence in out[0]:
    max_index=np.argmax(sequence)
    if label_dict_reverse[max_index]=='<END>':
        break
    ans=ans+" "+label_dict_reverse[max_index]

print(ans)


# 模型测试
# def predict_chinese(source,encoder_inference,decoder_inference,n_step,features):
#     state=encoder_inference.predict(source)
#     predict_seq=np.zeros((1,1,features))
#     predict_seq[0,0,label_dict['<START>']]=1
#     output=""
#     for i in range(n_step): # n_steps 为句子最大长度
#         # 给decoder输入上一个实可的h，c隐状态，以及上一次的预测字符 predict_seq
#         yhat,h,c=decoder_inference.predict([predict_seq]+state)
#         char_index = np.argmax(yhat[0,-1,:])
#         char=label_dict_reverse[char_index]
#         print(char)
#
#         state=[h,c]
#         predict_seq=np.zeros((1,1,features))
#         predict_seq[0,0,char_index]=1
#         if char == '<END>':
#             break
#         output+=" "+char
#     return output
#
# out=predict_chinese(encoder_input,encoder_infer,decoder_infer,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH)
# print(out)








