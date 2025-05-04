import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import gc


data = pd.read_csv("archive.zip")
print("\nРаспределение значений:")
print(data['toxic'].value_counts())

texts = data['comment'].values
labels = data['toxic'].values.astype(np.int64)

# Балансировка (66% нетоксичных, 33% токсичных)
toxic_idx = np.where(labels == 1)[0]
nontoxic_idx = np.where(labels == 0)[0]

# Ограничиваем выборку, чтобы избежать ошибок при малом количестве данных
n_samples = min(2 * len(toxic_idx), len(nontoxic_idx))
nontoxic_selected = np.random.choice(nontoxic_idx, size=n_samples, replace=False)
balanced_idx = np.concatenate([toxic_idx, nontoxic_selected])
np.random.shuffle(balanced_idx)

texts = texts[balanced_idx]
labels = labels[balanced_idx]

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Токенизация
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

# Датасет с весами классов
class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

        # Вычисляем веса для семплера
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        self.weights = torch.tensor(class_weights[labels], dtype=torch.float32)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

    def get_sampler(self):
        return WeightedRandomSampler(self.weights, len(self.weights))

train_dataset = ToxicDataset(train_encodings, y_train)
test_dataset = ToxicDataset(test_encodings, y_test)

# Кастомный Bert
class CustomBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, config.num_labels)
        )
        self.init_weights()
        self.class_weights = torch.tensor([1.0, 2.0])

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        return {'loss': loss, 'logits': logits}


# Инициализация модели
model = CustomBert.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=2
)

# Параметры обучения
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=1e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_precision",
    greater_is_better=True,
    report_to="none"
)

# Метрики
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions)
    }

# Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Параметры обучения
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=1e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_precision",
    greater_is_better=True,
    report_to="none"
)

# Метрики
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions)
    }

# Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

