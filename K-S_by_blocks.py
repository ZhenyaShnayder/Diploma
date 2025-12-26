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

def analyze_block(image, start_x, start_y, block_width, block_height):
    freq_green = [0] * 256
    freq_red = [0] * 256
    freq_blue = [0] * 256

    for x in range(start_x, min(start_x + block_width, image.width)):
        for y in range(start_y, min(start_y + block_height, image.height)):
            pixel = image.pixel(x, y)
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

    total_pixels_in_block = block_width * block_height
    D_n = diff / math.sqrt(total_pixels_in_block) if total_pixels_in_block > 0 else 0

    return D_n

def kolmogorov_filter_by_blocks(image, threshold_dn, min_fill_percentage, block_size_percent=5):
    block_width = max(1, int(image.width * block_size_percent / 100))
    block_height = max(1, int(image.height * block_size_percent / 100))
    
    blocks_processed = 0
    total_blocks = 0
    
    for start_x in range(0, image.width, block_width):
        for start_y in range(0, image.height, block_height):
            total_blocks += 1
            
            actual_block_width = min(block_width, image.width - start_x)
            actual_block_height = min(block_height, image.height - start_y)
            
            if actual_block_width < 2 or actual_block_height < 2:
                continue
            
            D_n = analyze_block(image, start_x, start_y, actual_block_width, actual_block_height)
            
            if D_n > threshold_dn:
                blocks_processed += 1
    
    if total_blocks > 0:
        fill_percentage = (blocks_processed / total_blocks) * 100
    else:
        fill_percentage = 0
    
    condition_met = fill_percentage >= min_fill_percentage
    
    return condition_met

def process_single_image_by_blocks(image_path, threshold_dn, threshold_percentage):
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
        
        condition_met = kolmogorov_filter_by_blocks(custom_image, threshold_dn, threshold_percentage)

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

def test_kolmogorov_filter_by_blocks(stego_images, usual_images):
    threshold_dn = float(input("Введите пороговое значение D_n: "))
    threshold_percentage = float(input("Введите минимальный процент заполнения контейнера: "))

    stego_sample = stego_images[:1000]
    usual_sample = usual_images[:1000]

    stego_results = []
    usual_results = []

    print("Обработка стегоизображений...")
    for i, img_path in enumerate(stego_sample):
        result = process_single_image_by_blocks(img_path, threshold_dn, threshold_percentage)
        if result:
            stego_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"Обработано стегоизображений: {i + 1}/{len(stego_sample)}")

    print("\nОбработка обычных изображений...")
    for i, img_path in enumerate(usual_sample):
        result = process_single_image_by_blocks(img_path, threshold_dn, threshold_percentage)
        if result:
            usual_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"Обработано обычных изображений: {i + 1}/{len(usual_sample)}")

    metrics = calculate_classification_metrics(stego_results, usual_results)

    print("\n" + "="*50)
    print("МАТРИЦА ОШИБОК:")
    print("="*50)
    print(f"  Истинно положительные (TP): {metrics['tp']}")
    print(f"  Ложно положительные (FP):  {metrics['fp']}")
    print(f"  Истинно отрицательные (TN): {metrics['tn']}")
    print(f"  Ложно отрицательные (FN):  {metrics['fn']}")
    print()

    print("МЕТРИКИ КАЧЕСТВА:")
    print("="*50)
    print(f"  Точность (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Прецизионность (Precision): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Полнота (Recall):     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-мера:              {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print()

test_kolmogorov_filter_by_blocks(stego_images, usual_images)
