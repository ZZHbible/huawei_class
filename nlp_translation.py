#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/5/1
# project = nlp_translation
import os
import re
import time

import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import text, sequence
import tensorflow as tf
from model import Encoder, BahdanauAttention, Decoder




path_to_file = './nlp_translation_data/data/cmn.txt'


# 预处理 1.统一小写 2.分词 3.开头加'start',结尾加'end'
def preprocess_eng(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r' \1', w)
    w = re.sub(r'[" "]+', " ", w)  # 多个空格合并成一个
    w = re.sub(r"[^a-zA-Z?.!,]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


def preprocess_chinese(w):
    w = w.lower().strip()
    w = jieba.cut(w, cut_all=False, HMM=True)
    w = " ".join(list(w))
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples=None):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[w for w in l.split('\t')] for l in lines[:num_examples]]
    word_pairs = [[preprocess_eng(w[0]), preprocess_chinese(w[1])] for w in word_pairs]
    return word_pairs


en, chn = zip(*create_dataset(path_to_file))


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = text.Tokenizer(filters="")
    # 生成 词和id的映射词典{word:id}
    lang_tokenizer.fit_on_texts(lang)
    # 将词转换为对应的id
    text_ids = lang_tokenizer.texts_to_sequences(lang)
    # 统一成相同长度
    padded_text_ids = sequence.pad_sequences(text_ids, padding='post')

    return padded_text_ids, lang_tokenizer


def load_dataset(path, num_examples=None):
    # 加载数据做预处理
    # 英文设置为目标语言,中文设置为源语言
    targ_lang, inp_lang = zip(*create_dataset(path, num_examples))

    input_data, inp_lang_tokenizer = tokenize(inp_lang)
    targ_data, targ_lang_tokenizer = tokenize(targ_lang)

    return input_data, targ_data, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = None
input_data, target_data, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

max_length_targ, max_length_inp = max_length(target_data), max_length(input_data)

input_train, input_val, target_train, target_val = train_test_split(input_data, target_data, test_size=0.05)


def convert(lang, data):
    for t in data:
        if t != 0:
            print('{} ----> {}'.format(t, lang.index_word[t]))


# 转换成 tf.data.Dataset
BUFFER_SIZE = len(input_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
# 0 是为padding保留的一个特殊id，所以要+1
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1
# 先shuffle，再取batch
# 先shuffle，再取batch
dataset=tf.data.Dataset.from_tensor_slices((input_train,target_train)).shuffle(BUFFER_SIZE)
dataset=dataset.batch(BATCH_SIZE,drop_remainder=True)

example_input_batch,example_target_batch=next(iter(dataset))


encoder=Encoder(vocab_inp_size,embedding_dim,units,BATCH_SIZE)
sample_hidden=encoder.initialize_hidden_state()
sample_output,sample_hidden=encoder(example_input_batch,sample_hidden)

# print("Encoder 输出的维度{}".format(sample_output.shape))
# print(sample_hidden.shape)
# print(sample_output[-1,-1,:]==sample_hidden[-1,:])

attention_layer=BahdanauAttention(10)
attention_result,attention_weights=attention_layer(sample_hidden,sample_output)

# print(attention_result.shape)
# print(attention_weights.shape)

decoder=Decoder(vocab_tar_size,embedding_dim,units,BATCH_SIZE)
sample_decoder_output,_,_=decoder(tf.random.uniform((64,1)),sample_hidden,sample_output)

# print((sample_decoder_output.shape))

# define optim and loss
optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')

def loss_function(real,pred):
    mask=tf.math.logical_not(tf.math.equal(real,0))
    loss_=loss_object(real,pred)
    mask=tf.cast(mask,dtype=loss_.dtype)
    loss_*=mask
    return tf.reduce_mean(loss_)

# 设置checkpoint
checkpoint_dir='./checkpoints/chinese-eng'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix=os.path.join(checkpoint_dir,"ckpt")
checkpoint=tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)

# 训练模型
def train_step(inp,targ,enc_hidden):
    loss=0
    with tf.GradientTape() as tape:
        enc_output,enc_hidden=encoder(inp,enc_hidden)
        dec_hidden=enc_hidden
        dec_input=tf.expand_dims([targ_lang.word_index['<start>']]*BATCH_SIZE,1)
        for t in range(1,targ.shape[1]):
            predictions,dec_hidden,_=decoder(dec_input,dec_hidden,enc_output)
            loss+=loss_function(targ[:,t],predictions)
            dec_input=tf.expand_dims(targ[:,t],1)

    batch_loss=(loss/int(targ.shape[1]))
    variables=encoder.trainable_variables+decoder.trainable_variables
    gradients=tape.gradient(loss,variables)
    optimizer.apply_gradients(zip(gradients,variables))
    return batch_loss

EPOCHS=30

for epoch in range(EPOCHS):
    start=time.time()
    enc_hidden=encoder.initialize_hidden_state()
    total_loss=0
    for (batch,(inp,targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss=train_step(inp,targ,enc_hidden)
        total_loss+=batch_loss
        if batch%20==0:
            print("epoch {} Batch {} loss{}".format(epoch+1,batch,batch_loss.numpy()))
    if (epoch+1)%5==0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    print("Epoch {} loss{}".format(epoch+1,total_loss/steps_per_epoch))
    print("time taken for 1 epoch{}".format(time.time()-start))

# 定义测试和可视化函数
def evaluate(sentence):
    attention_plot=np.zeros((max_length_targ,max_length_inp))
    sentence=preprocess_chinese(sentence)
    inputs=[inp_lang.word_index[i] for i in sentence.split()]
    inputs=sequence.pad_sequences([inputs],maxlen=max_length_inp,padding='post')
    inputs=tf.convert_to_tensor(inputs)
    result=''
    hidden=[tf.zeros((1,units))]
    enc_out,enc_hidden=encoder(inputs,hidden)
    dec_hidden=enc_hidden
    dec_input=tf.expand_dims([targ_lang.word_index['<start>']],0)
    for t in range(max_length_targ):
        predictions,dec_hidden,attention_weights=decoder(dec_input,dec_hidden,enc_out)
        attention_weights=tf.reshape(attention_weights,(-1,))
        attention_plot[t]=attention_weights.numpy()
        print(attention_weights)
        predicted_id=tf.argmax(predictions[0]).numpy()
        result+=targ_lang.index_word[predicted_id]+' '
        if targ_lang.index_word[predicted_id]=='<end>':
            return result,sentence,attention_plot
        dec_input=tf.expand_dims([predicted_id],0)
    return result,sentence,attention_plot

def translate(sentence):
    result,sentence,attention_plot=evaluate(sentence)
    print("intput: {} ".format(sentence))
    print('predicted translation: {}'.format(result))

checkpoint_dir='checkpoints/chinese-eng'
print(tf.train.latest_checkpoint(checkpoint_dir))
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
translate('我有一只猫')