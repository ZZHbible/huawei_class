#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/28
# project = model
import jieba
import numpy as np
import tensorflow
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input,Embedding,Conv1D,GlobalMaxPooling1D,Concatenate,Dense,GRU
from tensorflow.keras import Model


class NB_Classifier():
    def __init__(self):
        # 朴素贝叶斯分类器
        self.model = MultinomialNB(alpha=1)  # 拉普拉斯平滑:1
        # 使用tf-idf特征提取
        self.feature_processor = TfidfVectorizer(tokenizer=jieba.cut)

        self.label_map = {0: "negative", 1: "positive"}

    def fit(self, x_train, y_train, x_test, y_test):
        # tf-idf 特征提取
        x_train_fea = self.feature_processor.fit_transform(x_train)
        self.model.fit(x_train_fea, y_train)

        train_accuracy = self.model.score(x_train_fea, y_train)
        print("训练集准确率:{}".format(train_accuracy))

        x_test_fea = self.feature_processor.transform(x_test)
        y_predict = self.model.predict(x_test_fea)
        test_accuracy = self.model.score(x_test_fea, y_test)

        print("测试集准确率:{}".format(test_accuracy))

        y_predict = self.model.predict(x_test_fea)
        print("评估结果:")
        print(classification_report(y_test, y_predict, target_names=['0', '1']))

    def single_predict(self, text):
        text_preprocess = [" ".join(jieba.cut(text))]
        text_fea = self.feature_processor.transform(text_preprocess)
        predict_idx = self.model.predict(text_fea)[0]
        predict_label = self.label_map[predict_idx]
        predict_prob = self.model.predict_proba(text_fea)[0][predict_idx]

        return predict_label, predict_prob


class SVM_Classifier():
    def __init__(self, use_chi=False):
        self.use_chi = use_chi  # 是否使用卡方检验做特征选择
        # 朴素贝叶斯分类器
        self.model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        # 使用tf-idf特征提取
        self.feature_process = TfidfVectorizer(tokenizer=jieba.cut)
        if use_chi:
            self.feature_selector = SelectKBest(chi2, k=1e4)

        self.label_map = {0: "negative", 1: "positive"}

    def fit(self, x_train, y_train, x_test, y_test):
        x_train_fea = self.feature_process.fit_transform(x_train)
        if self.use_chi:
            x_train_fea = self.feature_selector.fit_transform(x_train_fea,y_train)
        self.model.fit(x_train_fea, y_train)

        train_accuracy = self.model.score(x_train_fea, y_train)
        print("训练集准确率:{}".format(train_accuracy))

        x_test_fea = self.feature_process.transform(x_test)
        if self.use_chi:
            x_test_fea = self.feature_selector.transform(x_test_fea)
        test_accuracy = self.model.score(x_test_fea, y_test)
        print("测试集准确率:{}".format(test_accuracy))
        print("测试评估矩阵:")
        print(classification_report(y_test, self.model.predict(x_test_fea), target_names=['negative', 'positive']))

    def single_predict(self, text):
        text_preprocess = [" ".join(jieba.cut(text))]
        text_fea = self.feature_process.transform(text_preprocess)
        if self.use_chi:
            text_fea = self.feature_selector.transform(text_fea)
        predict_idx = self.model.predict(text_fea)[0]
        predict_label = self.label_map[predict_idx]
        return predict_label


class Preprocess():
    def __init__(self,config):
        self.config=config
        token2idx={"[PAD]":0,"[UNK]":1} # {word:id}
        with open(config.vocab_file,'r',encoding='utf-8') as f:
            for index,line in enumerate(f.readlines()):
                token=line.strip()
                token2idx[token]=index+2
        self.token2idx=token2idx
    def transform(self,text_list):
        # 文本分词，并将词转换成相应的id，最后不同长度的文本padding长统一长度，后面补0
        idx_list=[[self.token2idx.get(word.strip(),self.token2idx['[UNK]']) for word in jieba.cut(text)] for text in text_list]
        idx_padding=pad_sequences(idx_list,self.config.max_seq_len,padding='post')
        return idx_padding

class TextCNN():
    def __init__(self,config):
        self.config=config
        self.preprocess=Preprocess(config)
        self.class_name={0:'negative',1:'positive'}

    def build_model(self):
        # 模型架构搭建
        idx_input=Input(self.config.max_seq_len)
        input_embedding=Embedding(len(self.preprocess.token2idx),self.config.embedding_dim,input_length=self.config.max_seq_len,mask_zero=True)(idx_input)
        convs=[]
        for kernel_size in [3,4,5]:
            c=Conv1D(128,kernel_size,activation='relu')(input_embedding)
            c=GlobalMaxPooling1D()(c)
            convs.append(c)
        fea_cnn=Concatenate()(convs)

        fea_dense=Dense(128,activation='relu')(fea_cnn)
        output=Dense(2,activation='softmax')(fea_dense)

        model=Model(inputs=idx_input,outputs=output)
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

        self.model=model

    def fit(self,x_train,y_train,x_valid=None,y_valid=None,epoch=5,batch_size=128,**kwargs):
        # 训练
        self.build_model()
        x_train=self.preprocess.transform(x_train)
        if x_valid is not None and y_valid is not None:
            x_valid=self.preprocess.transform(x_valid)
        self.model.fit(x_train,y_train,batch_size,epoch,validation_data=(x_valid,y_valid) if x_valid is not None and y_valid is not None else None)

    def evaluate(self,x_test,y_test):
        x_test=self.preprocess.transform(x_test)
        y_pred_probs=self.model.predict(x_test)
        y_pred=np.argmax(y_pred_probs,axis=-1)
        result=classification_report(y_test,y_pred,target_names=['negative','positive'])
        print(result)

    def single_predict(self,text):
        input_idx=self.preprocess.transform([text])
        predict_prob=self.model.predict(input_idx)[0]
        predict_label_id=np.argmax(predict_prob)
        predict_name=self.class_name[predict_label_id]
        predict_label_prob=predict_prob[predict_label_id]
        return predict_name,predict_label_prob


class Encoder(Model):
    def __init__(self,vocab_size,embedding_dim,enc_units,batch_size):
        super(Encoder, self).__init__()
        self.batch_size=batch_size
        self.enc_units=enc_units
        self.embedding=Embedding(vocab_size,embedding_dim)
        self.gru=GRU(self.enc_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')

    def call(self,x,hidden):
        x=self.embedding(x)
        output,state=self.gru(x,initial_state=hidden)
        return output,state

    def initialize_hidden_state(self):
        return tensorflow.zeros((self.batch_size,self.enc_units))

class BahdanauAttention(Model):
    def __init__(self,units):
        super(BahdanauAttention, self).__init__()
        self.W1=Dense(units)
        self.W2=Dense(units)
        self.V=Dense(1)
    def call(self,query,values):
        hidden_with_time_axis=tensorflow.expand_dims(query,1)
        score=self.V(tensorflow.nn.tanh(self.W1(values)+self.W2(hidden_with_time_axis)))
        attention_weights=tensorflow.nn.softmax(score,axis=1)
        context_vector=attention_weights*values
        context_vector=tensorflow.reduce_sum(context_vector,axis=1)
        return context_vector,attention_weights

class Decoder(Model):
    def __init__(self,vocab_size,embedding_dim,dec_units,batch_size):
        super(Decoder, self).__init__()
        self.batch_size=batch_size
        self.dec_units=dec_units
        self.embedding=Embedding(vocab_size,embedding_dim)
        self.gru=GRU(self.dec_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.fc=Dense(vocab_size)
        self.attention=BahdanauAttention(self.dec_units)

    def call(self,x,hidden,enc_output):
        context_vector,attention_weights=self.attention(hidden,enc_output)
        x=self.embedding(x)
        x=tensorflow.concat([tensorflow.expand_dims(context_vector,1),x],axis=-1)
        output,state=self.gru(x)
        output=tensorflow.reshape(output,(-1,output.shape[2]))
        x=self.fc(output)
        return x,state,attention_weights





