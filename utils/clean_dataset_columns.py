import json
import pandas as pd
from pathlib import Path

# Пути к тренировочному и тестовому датасетам 
TRAIN_IN = Path("data/split/train.parquet")
TEST_IN = Path("data/split/test.parquet")
# Путь к файлу со списком "лишних" признаков
REMOVED_PATH = Path("analysis/features_analysis/removed_features.json")
# Пути для сохранения файлов без лишних колонок
TRAIN_OUT = Path("data/split/train_clean.parquet")
TEST_OUT = Path("data/split/test_clean.parquet")


with REMOVED_PATH.open(encoding="utf-8") as f:
    removed_features = json.load(f)
train_df = pd.read_parquet(TRAIN_IN)
test_df = pd.read_parquet(TEST_IN)
print(f"train.parquet: {train_df.shape}")
print(f"test.parquet : {test_df.shape}")

# Удаление колонок
train_df_clean = train_df.drop(columns=[col for col in removed_features if col in train_df.columns])
test_df_clean = test_df.drop(columns=[col for col in removed_features if col in test_df.columns])

# Сохранение
train_df_clean.to_parquet(TRAIN_OUT, index=False)
test_df_clean.to_parquet(TEST_OUT, index=False)

print(f"Удалено {len(removed_features)} признаков")
print(f"train_clean.parquet: {train_df_clean.shape}")
print(f"test_clean.parquet : {test_df_clean.shape}")
