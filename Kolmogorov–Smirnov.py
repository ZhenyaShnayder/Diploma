import math
import os
import time
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import random

class Pixel:
    def __init__(self, red, green, blue):
        self.red = red
        self.green = green
        self.blue = blue

    def get_red(self):
        return self.red

    def get_green(self):
        return self.green

    def get_blue(self):
        return self.blue

    def set_red(self, value):
        self.red = max(0, min(255, int(value)))

    def set_green(self, value):
        self.green = max(0, min(255, int(value)))

    def set_blue(self, value):
        self.blue = max(0, min(255, int(value)))

class CustomImage:
    def __init__(self, width, height, pixels):
        self.width = width
        self.height = height
        self.pixels = pixels

    def pixel(self, w, h):
        return self.pixels[w][h]

def kolmogorov_filter(image):
    rows_processed = 0

    for h in range(image.height):
        freq_green = [0] * 256
        freq_red = [0] * 256
        freq_blue = [0] * 256

        # Считаем частоты для каждого канала
        for w in range(image.width):
            pixel = image.pixel(w, h)
            freq_green[pixel.get_green()] += 1
            freq_red[pixel.get_red()] += 1
            freq_blue[pixel.get_blue()] += 1

        F_r, F_exp_r = 0, 0
        F_g, F_exp_g = 0, 0
        F_b, F_exp_b = 0, 0
        diff = 0

        # Вычисляем эмпирическое и ожидаемое распределение
        for i in range(256):
            F_g += freq_green[i]
            F_r += freq_red[i]
            F_b += freq_blue[i]

            if i & 1 == 0:  # четный индекс
                if i + 1 < 256:
                    F_exp_g += (freq_green[i] + freq_green[i + 1]) // 2
                    F_exp_r += (freq_red[i] + freq_red[i + 1]) // 2
                    F_exp_b += (freq_blue[i] + freq_blue[i + 1]) // 2
            else:
                F_exp_g += (freq_green[i - 1] + freq_green[i] + 1) // 2
                F_exp_r += (freq_red[i - 1] + freq_red[i] + 1) // 2
                F_exp_b += (freq_blue[i - 1] + freq_blue[i] + 1) // 2

            current_diff = max(abs(F_r - F_exp_r), abs(F_g - F_exp_g), abs(F_b - F_exp_b))
            if current_diff > diff:
                diff = current_diff

        D_n = diff * math.sqrt(1.0 / F_r) if F_r > 0 else 0

        if D_n <= 0.2:
            rows_processed += 1

    threshold_percentage = 0.75  # 50%
    min_rows_required = image.height * threshold_percentage

    return rows_processed >= min_rows_required

def process_single_image(image_path):
    """Обрабатывает одно изображение и возвращает только условие D_n"""
    try:
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Создаем структуру пикселей
        pixels = []
        for x in range(width):
            row = []
            for y in range(height):
                r, g, b = image.getpixel((x, y))
                row.append(Pixel(r, g, b))
            pixels.append(row)

        custom_image = CustomImage(width, height, pixels)

        # Применяем фильтр Колмогорова
        condition_met = kolmogorov_filter(custom_image)

        return {
            'D_n_condition_met': condition_met
        }

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None

def calculate_classification_metrics(stego_results, usual_results):
    """Вычисляет метрики классификации"""

    y_true = []
    y_pred = []

    # Стего-изображения (класс 1)
    for result in stego_results:
        y_true.append(1)
        y_pred.append(1 if result['D_n_condition_met'] else 0)

    # Обычные изображения (класс 0)
    for result in usual_results:
        y_true.append(0)
        y_pred.append(1 if result['D_n_condition_met'] else 0)

    # Вычисляем метрики
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'confusion_matrix': cm,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_samples': len(y_true)
    }

def test_kolmogorov_filter(stego_images, usual_images):

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ФИЛЬТРА КОЛМОГОРОВА")

    # Берем по 2000 изображений
    stego_sample = stego_images[:2000]
    usual_sample = usual_images[:2000]

    stego_results = []
    usual_results = []

    # Обрабатываем стего-изображения
    print(f"\nОбработка стего-изображений:")
    for i, img_path in enumerate(stego_sample):
        print(f"  Обработано: {i}/2000")
        result = process_single_image(img_path)
        if result:
            stego_results.append(result)

    # Обрабатываем обычные изображения
    print(f"\nОбработка обычных изображений:")
    for i, img_path in enumerate(usual_sample):
        print(f"  Обработано: {i}/2000")
        result = process_single_image(img_path)
        if result:
            usual_results.append(result)

    # Вычисляем метрики
    metrics = calculate_classification_metrics(stego_results, usual_results)

    # Выводим результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("=" * 60)

    print(f"Общее количество: {metrics['total_samples']} изображений")
    print(f"  Стего: {len(stego_results)}")
    print(f"  Обычные: {len(usual_results)}")
    print()

    print("Матрица ошибок:")
    print(f"  TP: {metrics['tp']} | FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']} | TN: {metrics['tn']}")
    print()

    print("Метрики:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")

    return metrics

test_kolmogorov_filter(stego_images, usual_images)
