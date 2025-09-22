import os
import cv2

# Пути к видео
VIDEO_DIR_REAL = "faceforensics_data/original_sequences/youtube/c40/videos"
VIDEO_DIR_FAKE = "faceforensics_data/manipulated_sequences/Deepfakes/c40/videos"

# Пути для сохранения изображений
OUTPUT_DIR_REAL = "faceforensics_data/images/real"
OUTPUT_DIR_FAKE = "faceforensics_data/images/fake"

# Создаём папки, если их нет
os.makedirs(OUTPUT_DIR_REAL, exist_ok=True)
os.makedirs(OUTPUT_DIR_FAKE, exist_ok=True)

def extract_frames(video_path, output_folder, num_frames=10):
    """ Извлекает кадры из видео """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)  # Берём 10 равномерных кадров

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)  # Переход к нужному кадру
        ret, frame = cap.read()
        if not ret:
            break
        img_filename = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_{i}.jpg")
        cv2.imwrite(img_filename, frame)  # Сохраняем кадр

    cap.release()

# Обрабатываем реальные видео
for video in os.listdir(VIDEO_DIR_REAL):
    video_path = os.path.join(VIDEO_DIR_REAL, video)
    extract_frames(video_path, OUTPUT_DIR_REAL, num_frames=10)

# Обрабатываем deepfake-видео
for video in os.listdir(VIDEO_DIR_FAKE):
    video_path = os.path.join(VIDEO_DIR_FAKE, video)
    extract_frames(video_path, OUTPUT_DIR_FAKE, num_frames=10)

print("Извлечение изображений завершено!")