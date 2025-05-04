from torch import nn
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from flask import Flask, request, render_template
import torch


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

app = Flask(__name__)

# Загрузка модели и токенизатора
model = CustomBert.from_pretrained("./toxic_classifier_model")
tokenizer = BertTokenizer.from_pretrained("./toxic_classifier_model")
model.eval()


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1).tolist()[0]

            # Подготавливаем данные для шаблона
            class_probs = [
                {"class": i, "prob": round(p, 4), "percent": round(p * 100, 2)}
                for i, p in enumerate(probabilities)
            ]

            result = {
                'text': text,
                'class_probs': class_probs,
                'prediction': int(torch.argmax(logits).item()),
                'is_toxic': int(torch.argmax(logits).item()) != 0
            }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
