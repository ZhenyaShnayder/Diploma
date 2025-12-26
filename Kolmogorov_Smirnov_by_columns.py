import math
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

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

def kolmogorov_filter_by_columns(image, threshold_dn, min_fill_percentage):
    columns_processed = 0

    for w in range(image.width):
        freq_green = [0] * 256
        freq_red = [0] * 256
        freq_blue = [0] * 256

        for h in range(image.height):
            pixel = image.pixel(w, h)
            freq_green[pixel.get_green()] += 1
            freq_red[pixel.get_red()] += 1
            freq_blue[pixel.get_blue()] += 1

        expected_green = [0] * 256
        expected_red = [0] * 256
        expected_blue = [0] * 256

        for i in range(128):
            expected_green[2*i] = expected_green[2*i+1] = (freq_green[2*i] + freq_green[2*i+1]) / 2
            expected_red[2*i] = expected_red[2*i+1] = (freq_red[2*i] + freq_red[2*i+1]) / 2
            expected_blue[2*i] = expected_blue[2*i+1] = (freq_blue[2*i] + freq_blue[2*i+1]) / 2

        F_r, F_exp_r = 0, 0
        F_g, F_exp_g = 0, 0
        F_b, F_exp_b = 0, 0
        diff = 0

        for i in range(256):
            F_g += freq_green[i]
            F_r += freq_red[i]
            F_b += freq_blue[i]
            
            F_exp_g += expected_green[i]
            F_exp_r += expected_red[i]
            F_exp_b += expected_blue[i]

            current_diff = max(abs(F_r - F_exp_r), abs(F_g - F_exp_g), abs(F_b - F_exp_b))
            if current_diff > diff:
                diff = current_diff

        total_pixels_in_column = image.height
        D_n = diff / math.sqrt(total_pixels_in_column) if total_pixels_in_column > 0 else 0

        if D_n > threshold_dn:
            columns_processed += 1

    fill_percentage = (columns_processed / image.width) * 100 if image.width > 0 else 0
    condition_met = fill_percentage >= min_fill_percentage
    
    return condition_met

def process_single_image_by_columns(image_path, threshold_dn, threshold_percentage):
    try:
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        pixels = []
        for x in range(width):
            row = []
            for y in range(height):
                r, g, b = image.getpixel((x, y))
                row.append(Pixel(r, g, b))
            pixels.append(row)

        custom_image = CustomImage(width, height, pixels)
        condition_met = kolmogorov_filter_by_columns(custom_image, threshold_dn, threshold_percentage)

        return {
            'D_n_condition_met': condition_met
        }

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None

def calculate_classification_metrics(stego_results, usual_results):
    y_true = []
    y_pred = []

    for result in stego_results:
        y_true.append(1)
        y_pred.append(1 if result['D_n_condition_met'] else 0)

    for result in usual_results:
        y_true.append(0)
        y_pred.append(1 if result['D_n_condition_met'] else 0)

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

def test_kolmogorov_filter_by_columns(stego_images, usual_images):
    threshold_dn = float(input("Введите пороговое значение D_n: "))
    threshold_percentage = float(input("Введите минимальный процент заполнения контейнера: "))

    stego_sample = stego_images[:1000]
    usual_sample = usual_images[:1000]

    stego_results = []
    usual_results = []

    for i, img_path in enumerate(stego_sample):
        result = process_single_image_by_columns(img_path, threshold_dn, threshold_percentage)
        if result:
            stego_results.append(result)

    for i, img_path in enumerate(usual_sample):
        result = process_single_image_by_columns(img_path, threshold_dn, threshold_percentage)
        if result:
            usual_results.append(result)

    metrics = calculate_classification_metrics(stego_results, usual_results)

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
