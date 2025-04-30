import customtkinter as ctk
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Настройка внешнего вида
ctk.set_appearance_mode("System")  # Режим (Light/Dark)
ctk.set_default_color_theme("blue")  # Темы: blue, green, dark-blue


class ToxicityApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Анализатор токсичности")
        self.geometry("650x600")
        self.resizable(False, False)

        # Загрузка модели
        self.model_path = "toxic_classifier_model"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)

        self.create_widgets()

    def create_widgets(self):
        # Основной фрейм
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Заголовок
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Проверка токсичности текста",
            font=("Arial", 20, "bold")
        )
        self.title_label.pack(pady=20)

        # Поле ввода
        self.text_input = ctk.CTkTextbox(
            self.main_frame,
            height=200,
            width=550,
            font=("Arial", 14),
            wrap="word"
        )
        self.text_input.pack(pady=10, padx=10)

        # Кнопка
        self.check_button = ctk.CTkButton(
            self.main_frame,
            text="Проверить",
            command=self.check_toxicity,
            font=("Arial", 14),
            height=40,
            corner_radius=8
        )
        self.check_button.pack(pady=20)

        # Результат
        self.result_frame = ctk.CTkFrame(self.main_frame)
        self.result_frame.pack(fill="x", padx=10, pady=10)

        # Общий результат
        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Результат: ",
            font=("Arial", 16, "bold")
        )
        self.result_label.pack(pady=5)

        # Детализация вероятностей
        self.details_frame = ctk.CTkFrame(self.main_frame)
        self.details_frame.pack(fill="x", padx=10, pady=5)

        # Прогресс-бары для визуализации
        self.progress_frame = ctk.CTkFrame(self.main_frame)
        self.progress_frame.pack(fill="x", padx=10, pady=10)

        # Прогресс-бар для нетоксичности
        self.non_toxic_frame = ctk.CTkFrame(self.progress_frame)
        self.non_toxic_frame.pack(fill="x", pady=5)

        self.non_toxic_label = ctk.CTkLabel(
            self.non_toxic_frame,
            text="Нетоксичный:",
            width=100
        )
        self.non_toxic_label.pack(side="left")

        self.non_toxic_progress = ctk.CTkProgressBar(
            self.non_toxic_frame,
            orientation="horizontal",
            width=400,
            height=10
        )
        self.non_toxic_progress.pack(side="left", padx=5)
        self.non_toxic_progress.set(0)

        self.non_toxic_percent = ctk.CTkLabel(
            self.non_toxic_frame,
            text="0%",
            width=50
        )
        self.non_toxic_percent.pack(side="left")

        # Прогресс-бар для токсичности
        self.toxic_frame = ctk.CTkFrame(self.progress_frame)
        self.toxic_frame.pack(fill="x", pady=5)

        self.toxic_label = ctk.CTkLabel(
            self.toxic_frame,
            text="Токсичный:",
            width=100
        )
        self.toxic_label.pack(side="left")

        self.toxic_progress = ctk.CTkProgressBar(
            self.toxic_frame,
            orientation="horizontal",
            width=400,
            height=10
        )
        self.toxic_progress.pack(side="left", padx=5)
        self.toxic_progress.set(0)

        self.toxic_percent = ctk.CTkLabel(
            self.toxic_frame,
            text="0%",
            width=50
        )
        self.toxic_percent.pack(side="left")

    def check_toxicity(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            self.show_error("Введите текст для анализа!")
            return

        try:
            # Токенизация и предсказание
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

            # Получение вероятностей
            non_toxicity_percent = round(probs[0] * 100, 2)
            toxicity_percent = round(probs[1] * 100, 2)

            # Обновление интерфейса
            self.update_results(toxicity_percent, non_toxicity_percent)

        except Exception as e:
            self.show_error(f"Ошибка: {str(e)}")

    def update_results(self, toxic_prob, non_toxic_prob):
        # Общий результат
        if toxic_prob > 70:
            result = "Токсичный текст - " + str(toxic_prob)
            color = "#ff4d4d"  # Красный
        elif toxic_prob > 40:
            result = "Потенциально токсичный - " + str(toxic_prob)
            color = "#ff9933"  # Оранжевый
        else:
            result = "Нейтральный текст - " + str(toxic_prob)
            color = "#33cc33"  # Зеленый

        self.result_label.configure(text=f"Результат: {result}", text_color=color)

        # Детализация вероятностей
        self.non_toxic_progress.set(non_toxic_prob / 100)
        self.non_toxic_percent.configure(text=f"{non_toxic_prob}%")

        self.toxic_progress.set(toxic_prob / 100)
        self.toxic_percent.configure(text=f"{toxic_prob}%")

        # Цвета прогресс-баров
        self.set_progress_color(self.non_toxic_progress, non_toxic_prob, False)
        self.set_progress_color(self.toxic_progress, toxic_prob, True)

    def set_progress_color(self, progress_bar, percent, is_toxic):
        if is_toxic:
            if percent > 70:
                progress_bar.configure(progress_color="#ff4d4d")  # Красный
            elif percent > 40:
                progress_bar.configure(progress_color="#ff9933")  # Оранжевый
            else:
                progress_bar.configure(progress_color="#33cc33")  # Зеленый
        else:
            if percent > 70:
                progress_bar.configure(progress_color="#33cc33")  # Зеленый
            elif percent > 40:
                progress_bar.configure(progress_color="#ff9933")  # Оранжевый
            else:
                progress_bar.configure(progress_color="#ff4d4d")  # Красный

    def show_error(self, message):
        self.result_label.configure(text="Ошибка!", text_color="#ff4d4d")
        self.non_toxic_percent.configure(text="0%")
        self.toxic_percent.configure(text="0%")
        self.non_toxic_progress.set(0)
        self.toxic_progress.set(0)


if __name__ == "__main__":
    app = ToxicityApp()
    app.mainloop()