from transformers import BertTokenizer, BertForSequenceClassification


def get_bert_tokenizer(model_profile="bert-base-cased"):
    return BertTokenizer.from_pretrained(model_profile)


def get_bert_model(model_profile="bert-base-cased"):
    return BertForSequenceClassification.from_pretrained(model_profile)

