from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


def get_trainer(model, train_dataset, valid_dataset, compute_metrics=None, data_collator=None, *args, **kwargs):
    training_args = TrainingArguments(*args, **kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    return trainer

