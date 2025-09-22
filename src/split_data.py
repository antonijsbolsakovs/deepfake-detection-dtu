import os
import shutil
import random

# Пути к изображениям
DATA_DIR = "faceforensics_data/images"
TRAIN_DIR = "faceforensics_data/train"
TEST_DIR = "faceforensics_data/test"

# Процент данных для обучения
SPLIT_RATIO = 0.8

# Создаём папки train/test
for folder in [TRAIN_DIR, TEST_DIR]:
    os.makedirs(os.path.join(folder, "real"), exist_ok=True)
    os.makedirs(os.path.join(folder, "fake"), exist_ok=True)

def split_data(data_dir, train_dir, test_dir, split_ratio):
    """ Разбивает изображения на train/test """
    for category in ["real", "fake"]:
        images = os.listdir(os.path.join(data_dir, category))
        random.shuffle(images)  # Перемешиваем
        split_idx = int(len(images) * split_ratio)

        train_images = images[:split_idx]
        test_images = images[split_idx:]

        for img in train_images:
            shutil.move(os.path.join(data_dir, category, img), os.path.join(train_dir, category, img))

        for img in test_images:
            shutil.move(os.path.join(data_dir, category, img), os.path.join(test_dir, category, img))

split_data(DATA_DIR, TRAIN_DIR, TEST_DIR, SPLIT_RATIO)
print("✅ Данные успешно разделены на train/test!")