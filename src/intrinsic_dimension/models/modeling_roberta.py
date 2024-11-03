from transformers import RobertaTokenizer, RobertaForSequenceClassification


def get_roberta_tokenizer(model_profile="roberta-base"):
    return RobertaTokenizer.from_pretrained(model_profile)


def get_roberta_model(model_profile="roberta-base"):
    return RobertaForSequenceClassification.from_pretrained(model_profile)
