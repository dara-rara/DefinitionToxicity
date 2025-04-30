import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score


data = pd.read_csv("archive.zip")
texts = data['comment'].values
labels = data['toxic'].values

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


def encode_texts(texts, max_len=128):
    return tokenizer(
        texts.tolist(),
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )


train_encodings = encode_texts(X_train)
test_encodings = encode_texts(X_test)


class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ToxicDataset(train_encodings, y_train)
test_dataset = ToxicDataset(test_encodings, y_test)


model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=2
)

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    report_to="none",
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

model.save_pretrained("toxic_classifier_model")
tokenizer.save_pretrained("toxic_classifier_model")

eval_results = trainer.evaluate(test_dataset)
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Loss: {eval_results['eval_loss']:.4f}")