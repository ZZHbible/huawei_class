#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/28
# project = exp_6-text_classify
import jieba
import pandas as pd
from model import NB_Classifier,SVM_Classifier,Preprocess,TextCNN

train_data = pd.read_csv('./nlp_data/data/chnsenticorp/train.tsv', '\t')
valid_data=pd.read_csv('./nlp_data/data/chnsenticorp/dev.tsv','\t')
test_data = pd.read_csv('./nlp_data/data/chnsenticorp/test.tsv', '\t')

x_train, y_train = train_data.text_a.values, train_data.label.values
x_test, y_test = test_data.text_a.values, test_data.label.values
x_valid,y_valid=valid_data.text_a.values,valid_data.label.values

# nb_classifier=NB_Classifier()
# nb_classifier.fit(x_train, y_train, x_test, y_test)
#
# print(nb_classifier.single_predict("外观很漂亮，做工还蛮好"))

# SvmClassifier
# svm_classifier=SVM_Classifier()
# svm_classifier.fit(x_train, y_train, x_test, y_test)

# def feature_analysis():
#     feature_names=svm_classifier.feature_process.get_feature_names()
#     feature_scores=svm_classifier.feature_selector.scores_
#     feature_score_tups=list(zip(feature_names,feature_scores))
#     feature_score_tups.sort(key=lambda t:t[1],reverse=True)
#     return feature_score_tups


# print(feature_analysis()[:500])

# print(svm_classifier.single_predict("外观很漂亮，做工很好"))

# textCNN
# 构建词汇表
vocab=set()
cut_docs=train_data.text_a.apply(lambda x:jieba.cut(x)).values
for doc in cut_docs:
    for word in doc:
        if word.strip():
            vocab.add(word.strip())
with open('./nlp_data/data/vocab.txt','w',encoding='utf-8') as f:
    for word in vocab:
        f.write(word+'\n')
class Config():
    embedding_dim=300
    max_seq_len=200
    vocab_file="./nlp_data/data/vocab.txt"
config=Config()
process=Preprocess(config)
# print(process.transform(["我爱上海", "我爱杭州"]))

textcnn=TextCNN(config)
textcnn.fit(x_train,y_train,x_valid,y_valid,epoch=1)

textcnn.evaluate(x_test,y_test)
print(textcnn.single_predict("这本书写的真烂"))