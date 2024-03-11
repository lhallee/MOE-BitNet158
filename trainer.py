import torch
import numpy as np
from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, confusion_matrix


def compute_metrics_single_label_classification(p: EvalPrediction):

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids
    try:
        preds = torch.tensor(np.array(preds))
        y_true = torch.tensor(np.array(labels), dtype=torch.int).flatten()
    except:
        preds = torch.tensor(np.concatenate(preds, axis=0))
        y_true = torch.tensor(np.concatenate(labels, axis=0), dtype=torch.int).flatten()
    
    if preds.flatten().size() == y_true.size():
        y_pred = preds.flatten()
    else: 
        preds = preds.reshape(y_true.size(0), -1)
        y_pred = preds.argmax(dim=-1).flatten()
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    fmax = f1_score(y_true, y_pred, average='weighted')
    best_precision = precision_score(y_true, y_pred, average='weighted')
    best_recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'f1': fmax,
        'precision': best_precision,
        'recall': best_recall,
        'accuracy': accuracy,
    }



def get_trainer(model, train_dataset, valid_dataset, data_collator=None, *args, **kwargs):
    training_args = TrainingArguments(*args, **kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics_single_label_classification,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    return trainer

