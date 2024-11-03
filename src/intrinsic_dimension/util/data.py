from datasets import load_dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


def fetch_splits(clazz):
    train_dataset = clazz(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_dataset = clazz(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return train_dataset, test_dataset


def get_dataset(name, tokenizer=None):
    """
    utility to download and return standard datasets
    """

    if name == "mnist":
        return fetch_splits(datasets.MNIST)
    elif name == "cifar10":
        return fetch_splits(datasets.CIFAR10)
    elif name == "mrpc":
        raw_datasets = load_dataset("glue", "mrpc")
        tokenized_datasets = (raw_datasets
                              .map(lambda examples: tokenizer(examples['sentence1'], examples['sentence2'], truncation=True), batched=True)
                              .rename_column("label", "labels")
                              .remove_columns(["sentence1", "sentence2", "idx"]))

        tokenized_datasets.set_format("torch")

        return tokenized_datasets["train"], tokenized_datasets["validation"]
    elif name == "qqp":
        raw_datasets = load_dataset("glue", "qqp")
        tokenized_datasets = (raw_datasets
                              .map(lambda examples: tokenizer(examples['question1'], examples['question2'], truncation=True), batched=True)
                              .remove_columns(["question1", "question2", "idx"])
                              .rename_column("label", "labels")
                              .remove_columns(["sentence1", "sentence2", "idx"]))

        tokenized_datasets.set_format("torch")

        return tokenized_datasets["train"], tokenized_datasets["validation"]
    else:
        raise AttributeError(f"Unknown dataset: {name}")
