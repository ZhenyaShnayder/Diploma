import os  # Для работы с файловой системой
import shutil  # Для операций с файлами и папками
import kagglehub  # Для загрузки датасетов с Kaggle
import torch  # Основной фреймворк для глубокого обучения
from torch.utils.data import Dataset, DataLoader  # Для создания датасетов и загрузчиков данных
from torchvision import transforms  # Для преобразований изображений
from PIL import Image  # Для работы с изображениями
import torch.nn as nn  # Для создания нейронных сетей
 
# 1. Функция очистки предыдущих загрузок
def clear_previous_downloads():
    """
    Удаляет все файлы и папки в текущей рабочей директории.
    Это необходимо для предотвращения конфликтов при повторных запусках и для освобождения места на диске.
    """
    current_dir = os.getcwd()  # Получаем текущую рабочую директорию
    for filename in os.listdir(current_dir):
        if filename.endswith(('.bmp', '.zip', '.tar', '.gz')):  # Фильтрация по расширениям
            file_path = os.path.join(current_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Удаление файла
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Рекурсивное удаление папки
            except Exception as e:
                print(f'Ошибка при удалении {file_path}: {e}')

# 2. Очистка перед новой загрузкой данных
clear_previous_downloads()
def get_images_from_dataset(path):
    """
    Функция для сбора всех изображений из датасета.
    Args:
        path (str): Путь к директории с датасетом
    Returns:
        list: Список полных путей ко всем найденным изображениям
    """
    all_images = []
    image_extensions = ('.bmp')  # Расширения файлов для поиска
    # Рекурсивный обход директорий
    for root, dirs, files in os.walk(path):
        for file in files:
            # Проверяем расширение файла
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)  # Формируем полный путь
                all_images.append(file_path)
    return all_images

# 3. Загрузка датасетов с Kaggle
# Датасет с изображениями, содержащими стеганографические данные
path = kagglehub.dataset_download("diegozanchett/digital-steganography")
# Датасет без стеговставки
usual_paths = [
    kagglehub.dataset_download("vbookshelf/v2-plant-seedlings-dataset"),
    kagglehub.dataset_download("olgabelitskaya/flower-color-images"),
    kagglehub.dataset_download("muratkokludataset/grapevine-leaves-image-dataset")
] 
# 4. Получение списков изображений
stego_images = get_images_from_dataset(path)  # Изображения со стеговставкой
usual_images = []
for path in usual_paths: # Обычные изображения
    usual_images.extend(get_images_from_dataset(path))

# 5. Разделение данных на обучающую и тестовую выборки
# Вычисляем количество изображений для обучения (80%)
min_count = min(len(stego_images), len(usual_images))
stego_images = stego_images[:min_count]
usual_images = usual_images[:min_count]

split_idx = int(min_count * 0.8)
 
# Формируем обучающую выборку (первые 80% изображений)
stego_train = stego_images[:split_idx] # Стегоконтейнеры для обучения
usual_train = usual_images[:split_idx] # Обычные изображения для обучения
 
# Формируем тестовую выборку (оставшиеся 20% изображений)
stego_test = stego_images[split_idx:] # Стегоконтейнеры для тестирования
usual_test = usual_images[split_idx:] # Обычные изображения для тестирования
# Класс датасета для стегоанализа 
class StegoDataset(Dataset):
    """
    Класс датасета для работы со стегоизображениями.
    Объединяет стегоконтейнеры и обычные изображения в один датасет.
    """
    def __init__(self, stego, normal):
        """
        Инициализация датасета.
        Args:
            stego (list): Список путей к стегоизображениям
            normal (list): Список путей к обычным изображениям
        """
        # Создаем список кортежей (путь_к_изображению, метка)
        # Метка 1 - стегоизображение, 0 - обычное
        self.samples = [(img, 1) for img in stego] + [(img, 0) for img in normal]
 
        # Преобразования для изображений:
        # 1. Конвертация в тензор
        # 2. Нормализация (приводим значения пикселей к диапазону [-1, 1])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # Для всех 3 каналов (RGB)
        ])
 
    def __len__(self):
        """Возвращает количество изображений в датасете"""
        return len(self.samples)
 
    def __getitem__(self, idx):
        """
        Возвращает один элемент датасета по индексу.
        Args:
            idx (int): Индекс элемента
 
        Returns:
            tuple: (преобразованное изображение, метка)
        """
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')  # Загрузка и конвертация в RGB
        return self.transform(img), torch.tensor(label, dtype=torch.float32)
 
 
# Нейросетевая модель для стегоанализа
class SimpleStegoNet(nn.Module):
    """
    Простая CNN для классификации стегоизображений.
    Состоит из сверточных слоев и полносвязного классификатора.
    """
    def __init__(self, input_h, input_w):
        """
        Инициализация модели.
        Args:
            input_h (int): Высота входного изображения
            input_w (int): Ширина входного изображения
        """
        super().__init__()
 
        # Сверточная часть сети:
        self.cnn = nn.Sequential(
            # Первый сверточный блок
            nn.Conv2d(3, 16, 3, padding=1),  # 3 входных канала (RGB), 16 выходных
            nn.ReLU(),                       # Функция активации
            nn.MaxPool2d(2),                 # Уменьшение размерности в 2 раза
            # Второй сверточный блок
            nn.Conv2d(16, 32, 3, padding=1), # 16 входных, 32 выходных канала
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
 
        # Вычисление размеров после сверточных слоев
        cnn_out_h = input_h // 4  # Два пулинга с stride=2 уменьшают размер в 4 раза
        cnn_out_w = input_w // 4
 
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Преобразование в одномерный вектор
            nn.Dropout(0.3),  # Регуляризация
            nn.Linear(32 * cnn_out_h * cnn_out_w, 128),  # Полносвязный слой
            nn.ReLU(),
            nn.Linear(128, 1),  # Выходной слой (бинарная классификация)
            nn.Sigmoid()  # Приводим выход к диапазону [0, 1]
        )
 
    def forward(self, x):
        """Прямой проход через сеть"""
        x = self.cnn(x)
        return self.classifier(x)
 
 
# Подготовка данных
train_data = StegoDataset(stego_train, usual_train)
test_data = StegoDataset(stego_test, usual_test)
 
# Фиксированные размеры изображений (после дополнения нулями)
max_h, max_w = 128, 128
 
# Функция для обработки батча
def collate_fn(batch):
    """
    Функция для обработки батча:
    - Дополняет изображения нулями до максимального размера
    - Объединяет в тензоры
    """
    images, labels = zip(*batch)
    # Дополнение каждого изображения нулями до размера max_h x max_w
    padded = [torch.nn.functional.pad(img, (0, max_w-img.shape[2], 0, max_h-img.shape[1])) 
              for img in images]
    return torch.stack(padded), torch.stack(labels)  # Объединение в тензоры
 
 
# Создание загрузчиков данных
batch_size = 1  # Используем batch_size=1 для экономии памяти
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
 
# Обучение модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Используем GPU если доступен
model = SimpleStegoNet(input_h=max_h, input_w=max_w).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Оптимизатор Adam
criterion = nn.BCELoss()  # Функция потерь для бинарной классификации
 
# Цикл обучения
for epoch in range(10):
    model.train()  # Переводим модель в режим обучения
    for images, labels in train_loader:
        # Перенос данных на устройство (GPU/CPU)
        images, labels = images.to(device), labels.to(device)
 
        # Обнуляем градиенты
        optimizer.zero_grad()
 
        # Прямой проход
        outputs = model(images)
 
        # Вычисление ошибки
        loss = criterion(outputs, labels.unsqueeze(1))
 
        # Обратное распространение и обновление весов
        loss.backward()
        optimizer.step()
 
        # Очистка памяти 
        del images, labels, outputs
        torch.cuda.empty_cache()
 
 
# Оценка модели на тестовых данных
model.eval()  # Переводим модель в режим оценки
correct = total = 0
 
with torch.no_grad():  # Отключаем вычисление градиентов
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
 
        # Подсчет правильных предсказаний (порог 0.5)
        correct += ((outputs > 0.5).float() == labels.unsqueeze(1)).sum().item()
        total += labels.size(0)
 
        # Очистка памяти
        del images, labels, outputs
        torch.cuda.empty_cache()
 
# Вывод точности модели
print(f'Точность модели на тестовых данных: {100*correct/total:.2f}%')
