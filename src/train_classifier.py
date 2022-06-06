import argparse
import csv
import torch

from pathlib import Path
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, set_seed

set_seed(5)


def read_qw_data(f_path):
    f_path = Path(f_path)
    texts, labels = [], []
    with open(f_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            texts.append(row['question'].lower())
            labels.append(0 if float(row['score']) < 0.5 else 1)
    return texts, labels


class QWDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    parser = argparse.ArgumentParser(description='Train DistilBERT classifier on query wellformedness data.')

    val_texts, val_labels = read_qw_data('dev.tsv')
    trn_texts, trn_labels = read_qw_data('train.tsv')

    trn_encodings = tokenizer(trn_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    trn_dataset = QWDataset(trn_encodings, trn_labels)
    val_dataset = QWDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='results',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='logs',  # directory for storing logs
        logging_steps=10,
        run_name="qw_bert-base",  # name of the W&B run (optional)
        # run_name="qw_distilbert-base",  # name of the W&B run (optional)
        report_to=["wandb"],  # enable logging to W&B
    )

    # model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=trn_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()
    # model.save_pretrained("model/qw_distilbert-base")
    model.save_pretrained("model/qw_bert-base")
