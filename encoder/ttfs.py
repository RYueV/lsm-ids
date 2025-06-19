import json
import numpy as np



# Максимально возможная задержка (общее время обработки одной записи), мс
MAX_DELAY_MS = 20.0
# Количество колец задержек (временных каналов)
NUM_RINGS = 6
# Ширина кольца задержек, мс
RING_WIDTH = MAX_DELAY_MS / NUM_RINGS
# Степень нелинейности
GAMMA = 0.7
# Максимально возможное колебание времени, мс
JITTER_FRAC = 0.005
# Фиксация рандома
RAND = np.random.RandomState(42)



# Загрузка json-файла с min/max диапазонами признаков
def load_feature_ranges(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        feature_ranges = json.load(f)
    # Формирование словаря вида {"название признака": (min_value, max_value)}
    return {k: (float(v[0]), float(v[1])) for k, v in feature_ranges.items()}



def build_encoder(
    feature_ranges,         # словарь диапазонов значений признаков
    skip_zeros=False        # пропускать ли нулевые значения при кодировании
):
    # Сортировка признаков по их наименованию в алфавитном порядке
    num_cols = sorted(feature_ranges)
    # Количество признаков
    n_feats = len(num_cols)

    # Вычисление индекса канала (входного нейрона) для признака
    def calc_channel_idx(
            feature_idx,        # индекс признака в отсортированном датасете
            ring_idx            # индекс кольца
        ):
        return ring_idx * n_feats + feature_idx


    # TTFS-кодирование записи датасета в спайки
    def encode(
            record      # одна запись из датасета
        ):
        # Список кортежей спайков вида (канал, время)
        spikes = []

        # feat_idx - индекс признака, col_name - наименование колонки
        for feat_idx, col_name in enumerate(num_cols):
            # Извлекаем диапазоны признака
            min_value, max_value = feature_ranges[col_name]
            # Если разница слишком мала, пропускаем
            delta_value = max_value - min_value
            if delta_value < 1e-9:
                continue

            # Нормализация значения признака
            norm_value = (float(record[col_name]) - min_value) / delta_value
            norm_value = np.clip(norm_value, 0.0, 1.0)
            
            # Если установлен флаг "не кодировать нули" и norm_value близко к 0, пропускаем
            if skip_zeros and norm_value <= 1e-9:
                continue

            # Кодируем спайки с задержками в каждое из NUM_RINGS колец
            for r_idx in range(NUM_RINGS):
                base = r_idx * RING_WIDTH 
                # Вычисление задержки в мс (чем больше значение, тем раньше спайк)
                delay = base + RING_WIDTH  * (1.0 - norm_value**GAMMA)
                # Случайное колебание
                delay += RAND.uniform(-JITTER_FRAC, JITTER_FRAC) * RING_WIDTH 
                delay = np.clip(delay, base, base + RING_WIDTH )
                # Обновляем список спайков
                spikes.append((calc_channel_idx(feat_idx, r_idx), delay))

        # Сортировка спайков по времени
        spikes.sort(key=lambda x: x[1])
        return np.asarray(spikes, dtype=np.float32)

    # Общее количество входных нейронов
    encode.num_neurons = NUM_RINGS * n_feats
    # Упорядоченный набор наименований признаков
    encode.feature_list = num_cols

    return encode


