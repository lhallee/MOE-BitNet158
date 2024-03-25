from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


def fine_tune_trainer(model,
                      train_dataset,
                      valid_dataset,
                      tokenizer=None,
                      compute_metrics=None,
                      data_collator=None,
                      patience=3,
                      *args, **kwargs):
    training_args = TrainingArguments(*args, **kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )
    return trainer


def pretrain_trainer(model,
                     train_dataset,
                     tokenizer,
                     *args, **kwargs):
    training_args = TrainingArguments(*args, **kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )
    return trainer