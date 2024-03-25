import yaml
from transformers import AutoTokenizer
from datasets import load_dataset

from trainer import pretrain_trainer, fine_tune_trainer
from models.config import BitformerConfig
from models.model_zoo import BitformerForLM, BitformerForSequenceClassification


def get_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        args = yaml.safe_load(file)
    return args


def pretrain(args, yargs):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    train_dataset = load_dataset(args.data_path, 'en',
                                split='train',
                                streaming=args.streaming,
                                trust_remote_code=True)
    
    remove = list(next(iter(train_dataset)).keys())

    train_dataset = train_dataset.map(encode, batched=True, remove_columns=remove)
    train_dataset = train_dataset.with_format('torch')

    cfg = BitformerConfig(**yargs['model_config'])
    cfg.bos_token_id = tokenizer.bos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id
    is_causal = yargs['model_config']['is_causal']
    model = BitformerForLM(config=cfg)
    trainer = pretrain_trainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        mlm=not is_causal,
        **yargs['training_args']
    )
    trainer.train()
    trainer.push_to_hub(args.save_path, token=args.token)


def finetine(args, yargs):
    train_dataset = load_dataset(args.data_path, split='train', trust_remote_code=True).with_format('torch')
    valid_dataset = load_dataset(args.data_path, split='valid', trust_remote_code=True).with_format('torch')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    cfg = BitformerConfig(**yargs['model_config'])
    cfg.bos_token_id = tokenizer.bos_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id
    model = BitformerForSequenceClassification(config=cfg, num_labels=yargs['general_config']['num_labels'])

    compute_metrics = None # impement
    data_collator = None

    trainer = fine_tune_trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        patience=yargs['general_config']['patience'],
        **yargs['training_args']
    )
    trainer.train()
    trainer.push_to_hub(args.save_path, token=args.token)
