# -*- coding:utf-8 -*-

import json
from huaweicloud_nlp.MtClient import MtClient
from huaweicloud_nlp.NlpfClient import NlpfClient
from huaweicloud_nlp.NluClient import NluClient
from huaweicloud_nlp.NlgClient import NlgClient
from huaweicloud_nlp.HWNlpClientAKSK import HWNlpClientAKSK

class AKSKDemo(object):

    akskClient = HWNlpClientAKSK("your_ak", "your_sk", "cn-north-4"
                                 , "0463e37ea59b4f6cb5b8d381ad92f852")

    def __init__(self):
        pass

    def test_segment(self):
        nlpfClient = NlpfClient(AKSKDemo.akskClient)
        response = nlpfClient.segment("今天天气真好", 1,"zh","PKU")
        print(response.code)
        print(json.dumps(response.res, ensure_ascii=False))

    def test_ner(self):
        nlpfClient = NlpfClient(AKSKDemo.akskClient)
        response = nlpfClient.ner("未来的发展是不确定的", "zh")
        print(response.code)
        print(json.dumps(response.res, ensure_ascii=False))

    def test_domain_ner(self):
        nlpfClient = NlpfClient(AKSKDemo.akskClient)
        response = nlpfClient.domain_ner("未来的发展是不确定的", "zh", "general")
        print(json.dumps(response.res,ensure_ascii=False))

    def test_get_text_similarity(self):
        nlpfClient = NlpfClient(AKSKDemo.akskClient)
        response = nlpfClient.get_text_similarity("文字1", "文字2", "zh")
        print(json.dumps(response.res,ensure_ascii=False))

    def test_get_sentence_vectors(self):
        nlpfClient = NlpfClient(AKSKDemo.akskClient)
        response = nlpfClient.get_sentence_vectors(["男儿当自强"], "general")
        print(json.dumps(response.res, ensure_ascii=False))

    def test_get_entity_linking(self):
        nlpfClient = NlpfClient(AKSKDemo.akskClient)
        response = nlpfClient.get_entity_linking("周杰伦的歌", None)
        print(json.dumps(response.res, ensure_ascii=False))

    def test_extract_keywordself(self):
        nlpfClient = NlpfClient(AKSKDemo.akskClient)
        response = nlpfClient.extract_keyword("华为技术有限公司成立于1987年，总部位于广东省深圳市龙岗区。", 5, "zh")
        print(json.dumps(response.res, ensure_ascii=False))

    def test_get_sentiment(self):
        nluClient = NluClient(AKSKDemo.akskClient)
        response = nluClient.get_sentiment("浑浑噩噩的头脑、失魂落魄的身体…")
        print(json.dumps(response.res, ensure_ascii=False))

    def test_get_domain_sentiment(self):
        nluClient = NluClient(AKSKDemo.akskClient)
        response = nluClient.get_domain_sentiment("浑浑噩噩的头脑、失魂落魄的身体…", 0)
        print(json.dumps(response.res, ensure_ascii=False))

    def test_get_intent(self):
        nluClient = NluClient(AKSKDemo.akskClient)
        response = nluClient.get_intent("来一首周杰伦的青花瓷","zh")
        print(json.dumps(response.res, ensure_ascii=False))

    def test_classify_text(self):
        nluClient = NluClient(AKSKDemo.akskClient)
        response = nluClient.classify_text("XXX去屑洗发水，全国包邮", 1)
        print(json.dumps(response.res, ensure_ascii=False))

    def test_summary(self):
        nlgClient = NlgClient(AKSKDemo.akskClient)
        response = nlgClient.summary("华为刀片式基站解决方案是华为在深入理解客户诉求基础上，引领业界的创新解决方案。"
                                     ,"华为", None, "zh")
        print(json.dumps(response.res, ensure_ascii=False))

    def test_domain_summary(self):
        nlgClient = NlgClient(AKSKDemo.akskClient)
        response = nlgClient.domain_summary("华为刀片式基站解决方案是华为在深入理解客户诉求基础上，引领业界的创新解决方案。"
                                     ,"华为", None, "zh", 0)
        print(json.dumps(response.res, ensure_ascii=False))

    def test_generate_poem(self):
        nlgClient = NlgClient(AKSKDemo.akskClient)
        response = nlgClient.generate_poem("写诗", 0, False)
        print(json.dumps(response.res, ensure_ascii=False))

    def test_translate_text(self):
        mtClient = MtClient(AKSKDemo.akskClient)
        response = mtClient.translate_text("how are you", "en", "zh", "common")
        print(json.dumps(response.res, ensure_ascii=False))

    def test_dectect_lang(self):
        mtClient = MtClient(AKSKDemo.akskClient)
        response = mtClient.detect_language("how are you")
        print(json.dumps(response.res, ensure_ascii=False))

    def text_multi_grained_segment(self):
        nlpfClient = NlpfClient(AKSKDemo.akskClient)
        response = nlpfClient.multi_grained_segment("华为技术有限公司的总部", "zh")
        print(json.dumps(response.res, ensure_ascii=False))

def main():
    userDemo = AKSKDemo()
    userDemo.test_segment()
    userDemo.test_ner()
    userDemo.test_domain_ner()
    userDemo.test_get_text_similarity()
    userDemo.test_get_sentence_vectors()
    userDemo.test_get_entity_linking()
    userDemo.test_get_sentiment()
    userDemo.test_get_domain_sentiment()
    userDemo.test_get_intent()
    userDemo.test_classify_text()
    userDemo.test_summary()
    userDemo.test_domain_summary()
    userDemo.test_generate_poem()
    userDemo.test_translate_text()
    userDemo.test_dectect_lang()
    userDemo.test_extract_keywordself()
    userDemo.text_multi_grained_segment()

if __name__ == '__main__':
    main()