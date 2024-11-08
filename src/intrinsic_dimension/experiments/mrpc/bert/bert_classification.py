import gc
from collections import defaultdict

import torch
from transformers import set_seed, get_scheduler, TrainingArguments, Trainer, DataCollatorWithPadding

from ....models.modeling_bert import get_bert_model, get_bert_tokenizer
from ....util.constants import METRIC, DEVICE
from ....util.data import get_dataset
from ....util.plotter import plot_model, plot_results
from ....util.util import parse_args, generate_intrinsic_dimension_candidates, count_params
from ....wrappers.modeling_transformer_layer import TransformerSubspaceWrapper

basedir = "src/intrinsic_dimension/experiments/mrpc/bert"


def accuracy_criterion(eval_prediction):
    logits, labels = eval_prediction
    predictions = logits.argmax(axis=1)
    accuracy = METRIC.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}


def train(args, model, train_dataset, eval_dataset, data_collator):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0
    )

    num_training_steps = args.num_epochs * len(train_dataset) // args.train_batch_size
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    training_args = TrainingArguments(
        output_dir=f"{basedir}/checkpoints",
        eval_strategy="epoch",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        seed=args.seed,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        lr_scheduler_type=args.lr_scheduler_type,
        save_steps=500,
        logging_dir=f"{basedir}/logs",
        report_to="none",
        do_train=True,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=accuracy_criterion
    )

    # Train and evaluate
    trainer.train()

    return trainer.evaluate()["eval_accuracy"]


if __name__ == '__main__':
    print(f"Using {DEVICE}")

    args = parse_args("bert", "mrpc")

    set_seed(args.seed)

    tokenizer = get_bert_tokenizer()

    train_dataset, eval_dataset = get_dataset("mrpc", tokenizer=tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # sample data for plotting model (primarily for dimensions)
    sample = next(iter(torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=data_collator)))

    # baseline
    baseline_model = get_bert_model()
    baseline_accuracy = train(args, baseline_model, train_dataset, eval_dataset, data_collator)
    plot_model(baseline_model, "baseline", basedir, sample)

    del baseline_model
    gc.collect()

    # run for 10 d values from 1 to 10000 equally spaced on a log scale
    history = defaultdict(dict)
    found_did_dint90, found_said_dint90 = False, False
    for dint in generate_intrinsic_dimension_candidates(start=3, end=4, n_points=10, plot=True, basedir=basedir):
        # wrap all linear and conv layers with the subspace layer
        did_model = TransformerSubspaceWrapper(get_bert_model(), dint)
        history[dint]["did"] = train(args, did_model, train_dataset, eval_dataset, data_collator)

        if not found_did_dint90 and history[dint]["did"] >= 0.9 * baseline_accuracy:
            found_did_dint90 = True
            n_params = count_params(did_model)
            print("n_params in intrinsic model: ", n_params)
            plot_model(did_model, "did_intrinsic_dim", basedir, sample)

        del did_model
        gc.collect()

        said_model = TransformerSubspaceWrapper(get_bert_model(), dint, said=True)
        history[dint]["said"] = train(args, said_model, train_dataset, eval_dataset, data_collator)

        if not found_said_dint90 and history[dint]["said"] >= 0.9 * baseline_accuracy:
            found_said_dint90 = True
            n_params = count_params(said_model)
            print("n_params in intrinsic model: ", n_params)
            plot_model(said_model, "said_intrinsic_dim", basedir, sample)

        del said_model
        gc.collect()

    plot_results(
        baseline=baseline_accuracy,
        dints=history.keys(),
        performance={
            "did": [perf["did"] for perf in history.values()],
            "said": [perf["said"] for perf in history.values()]
        },
        basedir=basedir,
        name=f"mrpc-bert",
        xlabel="subspace dim d",
        ylabel="validation accuracy",
        show_dint90=True
    )
