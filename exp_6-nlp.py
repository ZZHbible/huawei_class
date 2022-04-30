# coding: utf-8
# !/usr/bin/env python
# author = 'ZZH'
# time = 2022/4/28
# project = exp_6-nlp
import json
from huaweicloud_nlp.MtClient import MtClient
from huaweicloud_nlp.NlpfClient import NlpfClient
from huaweicloud_nlp.HWNlpClientAKSK import HWNlpClientAKSK
from huaweicloud_nlp.NlgClient import NlgClient
from huaweicloud_nlp.HWNlpClientToken import HWNlpClientToken
from huaweicloud_nlp.NluClient import NluClient

with open('./as_ak.json','r') as f:
    as_ak=json.load(f)
    app_key = as_ak['as']
    app_password = as_ak['ak']

project_id = "e1a62164540747f4934d30f8619a2765"
region = 'cn-north-4'

# 初始化客户端
hwNlpClient = HWNlpClientAKSK(app_key, app_password, region, project_id)
nlpClient = NlpfClient(hwNlpClient)

# 分词
# response=nlpClient.segment("今天天气真好",1,'zh',"PKU")
# print(response.code)
# print(json.dumps(response.res,ensure_ascii=False))

# 多粒度分词
# response=nlpClient.multi_grained_segment("华为技术有限公司的总部",'zh')
# print(json.dumps(response.res,ensure_ascii=False))

# 命名实体识别
# response=nlpClient.ner("昨天程序员小张来到长春喜家德吃了一份喜三鲜水饺表示很好吃，如果有女朋友一起吃就更好了",'zh')
# print(json.dumps(response.res,ensure_ascii=False))

# 文本相似度
# response=nlpClient.get_text_similarity("今天天气晴朗","今天天气晴",'zh')
# print(json.dumps(response.res,ensure_ascii=False))

# 句向量
# response = nlpClient.get_sentence_vectors(["男儿当自强"],'general')
# print(json.dumps(response.res,ensure_ascii=False))

# 实体链接
# response=nlpClient.get_entity_linking("李娜在青藏高原唱歌真好听",'zh')
# print(json.dumps(response.res,ensure_ascii=False))

# 关键词抽取
# response=nlpClient.extract_keyword("华为技术有限公司成立于1987年，是一个很厉害的公司",2,'zh')
# print(json.dumps(response.res,ensure_ascii=False))

# 文本摘要
nlgClient = NlgClient(hwNlpClient)
# response=nlgClient.summary("视频提供了功能强大的方法帮助您证明您的观点。当您单击联机视频时，可以在想要添加的视频的嵌入代码中进行粘贴。您也可以键入一个关键字以联机搜索最适合您的文档的视频。为使您的文档具有专业外观，Word 提供了页眉、页脚、封面和文本框设计，这些设计可互为补充。例如，您可以添加匹配的封面、页眉和提要栏。单击“插入”，然后从不同库中选择所需元素。主题和样式也有助于文档保持协调。当您单击设计并选择新的主题时，图片、图表或 SmartArt 图形将会更改以匹配新的主题。当应用样式时，您的标题会进行更改以匹配新的主题。使用在需要位置出现的新按钮在 Word 中保存时间。若要更改图片适应文档的方式，请单击该图片，图片旁边将会显示布局选项按钮。当处理表格时，单击要添加行或列的位置，然后单击加号。在新的阅读视图中阅读更加容易。可以折叠文档某些部分并关注所需文本。如果在达到结尾处之前需要停止读取，Word 会记住您的停止位置 - 即使在另一个设备上。视频提供了功能强大的方法帮助您证明您的观点。当您单击联机视频时，可以在想要添加的视频的嵌入代码中进行粘贴。您也可以键入一个关键字以联机搜索最适合您的文档的视频。为使您的文档具有专业外观，Word 提供了页眉、页脚、封面和文本框设计，这些设计可互为补充。例如，您可以添加匹配的封面、页眉和提要栏。单击“插入”，然后从不同库中选择所需元素。主题和样式也有助于文档保持协调。当您单击设计并选择新的主题时，图片、图表或 SmartArt 图形将会更改以匹配新的主题。当应用样式时，您的标题会进行更改以匹配新的主题。使用在需要位置出现的新按钮在 Word 中保存时间。若要更改图片适应文档的方式，请单击该图片，图片旁边将会显示布局选项按钮。当处理表格时，单击要添加行或列的位置，然后单击加号。在新的阅读视图中阅读更加容易。可以折叠文档某些部分并关注所需文本。如果在达到结尾处之前需要停止读取，Word 会记住您的停止位置 - 即使在另一个设备上。视频提供了功能强大的方法帮助您证明您的观点。当您单击联机视频时，可以在想要添加的视频的嵌入代码中进行粘贴。",'老师',None,'zh')
# print(json.dumps(response.res,ensure_ascii=False))

# 诗歌生成 （功能未开通
# response=nlgClient.generate_poem("秋冬",0,False)
# print(json.dumps(response.res,ensure_ascii=False))

nluClient = NluClient(hwNlpClient)
# 情感分析
# response=nluClient.get_sentiment("浑浑噩噩的头脑，失魂落魄的身体")
# print(json.dumps(response.res,ensure_ascii=False))

# 判断是否是广告
# response=nluClient.classify_text("小张牌洗发液，江浙沪包邮",1)
# print(json.dumps(response.res,ensure_ascii=False))

# 意图理解
response = nluClient.get_intent("来一首周杰伦的青花瓷", 'zh')
print(json.dumps(response.res, ensure_ascii=False))
