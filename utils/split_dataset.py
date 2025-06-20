import argparse
import logging
import sys
from pathlib import Path
import pandas as pd



# Путь к очищенному датасету
IN_PATH = Path("data/clean/cicids2017_clean.parquet")
# Папка для сохранения результатов
OUT_PATH = Path("data/split")
# Если Label встречается в датасете реже FREQ_TH раз, то все записи такого типа удаляются
FREQ_TH = 200
# Доля записей из конца каждого файла, которые идут в тестовую выборку
TEST_SHARE = 0.25
# Наименование колонки с временными метками
TIME_COL = "Timestamp"
# Наименование колонки с типом трафика
LABEL_COL = "Label"
# Формат для записи логов
LOG_FORMAT = "[%(levelname)s] %(message)s"



# Загрузка и проверка корректности исходного датасета
def load_df(path):
    if not path.exists():
        raise FileNotFoundError(path)
    # Загрузка датасета
    df = pd.read_parquet(path)
    # Датасет должен обязательно содержать метки времени и метки трафика
    if TIME_COL not in df.columns or LABEL_COL not in df.columns:
        raise KeyError(f"Файл поврежден: отсутствуют {TIME_COL} или {LABEL_COL}")
    # Приведение значений в колонке с временными метками к типу datetime
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    # Удаление строк с битыми TIME_COL и/или LABEL_COL
    df.dropna(subset=[TIME_COL, LABEL_COL], inplace=True)
    return df



# Удаление классов с количеством записей меньше FREQ_TH
def drop_rare_rows(df):
    # Вычисление количества строк для каждого класса
    count_rows = df[LABEL_COL].value_counts()
    # Получение имен классов с размером меньше FREQ_TH
    rare_labels = count_rows[count_rows < FREQ_TH].index.tolist()
    # Создание датасета без редких классов
    new_df = df[~df[LABEL_COL].isin(rare_labels)].copy()
    if rare_labels:
        logging.info(f"Удалены редкие классы: {rare_labels}")
    return new_df, rare_labels



# Разделение очищенного датасета на train и test
def split_by_label_and_time(df):
    # Списки для хранения блоков обучающей и тестовой выборки
    train_parts, test_parts = [], []
    # Цикл по группам строк одного класса
    for _, group in df.groupby(LABEL_COL, sort=False):
        # Сортировка по времени
        group = group.sort_values(TIME_COL)
        # Количество записей этого класса, которые должны попасть в тестовую выборку
        n_test = max(1, int(len(group) * TEST_SHARE))
        # Последние n_test строк в тестовую выборку, остальные в обучающую
        test_parts.append(group.tail(n_test))
        train_parts.append(group.iloc[:-n_test])
    # Объединение кусков внутри датасетов со сбросом старых индексов
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df



# Балансировка Normal относительно наибольшего из классов attack
def undersample_majority(
        train_df,           # тренировочный датасет
        random_state=42     # инициализация генератора случайных чисел
    ):
    # Вычисление количества строк для каждого класса в тренировочном датасете
    count_rows = train_df[LABEL_COL].value_counts()
    # Если нет записей класса Normal, то пропускаем
    if "Normal" not in count_rows:
        logging.warning("Класс Normal не найден. Балансировка пропущена.")
        return train_df
    # Целевой размер класса Normal - размер максимального класса из Attack
    target_size = count_rows.drop("Normal").max()
    # Считаем количество строк, которые нужно удалить
    need_remove = count_rows["Normal"] - target_size
    cnt_r = count_rows["Normal"]
    # Если разность между текущим и целевым размером класса неположительна
    if need_remove <= 0:
        # Пропускаем балансировку
        logging.info(f"Балансировка не нужна ({cnt_r} <= {target_size})")
        return train_df
    # Формирование списка индексов строк, которые нужно оставить
    keep_idx = (
        train_df[train_df[LABEL_COL] == "Normal"]
        .sample(n=target_size, random_state=random_state)       # случайный выбор target_size строк Normal
        .index                                                  # все строки остальных классов оставляем
        .union(train_df[train_df[LABEL_COL] != "Normal"].index) # объединяем эти индексы в единый набор
    )
    logging.info(f"Количество строк в классе Normal: {cnt_r} -> {target_size}")
    # Возвращаем итоговую таблицу со сбросом индексов
    return train_df.loc[keep_idx].sample(frac=1.0, random_state=random_state).reset_index(drop=True)



# Сохранение parquet-файлов
def save_parquets(train_df, test_df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", compression="snappy", engine="pyarrow")
    test_df.to_parquet(out_dir / "test.parquet",  compression="snappy", engine="pyarrow")
    logging.info(f"Сохранено в {out_dir}")



# Сохранение статистики по файлам
def save_distribution(train_df, test_df, out_dir):
    dist = {
        "all": pd.concat([train_df, test_df])[LABEL_COL].value_counts(),
        "train": train_df[LABEL_COL].value_counts(),
        "test": test_df[LABEL_COL].value_counts()
    }
    dist_df = pd.DataFrame(dist).fillna(0).astype(int)
    tot = dist_df["all"].sum()
    dist_df["train_%"] = (dist_df["train"] / dist_df["all"] * 100).round(2)
    dist_df["test_%"] = (dist_df["test"] / dist_df["all"] * 100).round(2)
    dist_df["all_%"] = (dist_df["all"] / tot * 100).round(2)
    csv_path = out_dir / "class_distribution.csv"
    dist_df.to_csv(csv_path, encoding="utf-8")
    logging.info(f"Сводная таблица классов сохранена: {csv_path}")
    logging.info(f"Сводная таблица по классам:\n{dist_df}")



def parse_cli_args(argv=None):
    p = argparse.ArgumentParser()
    add = p.add_argument
    add("--input", type=Path, default=IN_PATH, help="Входной parquet")
    add("--out_dir", type=Path, default=OUT_PATH, help="Папка вывода")
    add("--no_balance", action="store_true", help="Не усекать 'Normal'")
    add("--save_csv", action="store_true", help="Сохранить class_distribution.csv")
    add("--log_file", type=str, default="", help="Также писать лог в файл")
    return p.parse_args(argv)


# Настройки логирования в file
def setup_log(file=""):
    # Для вывода логов в стандартный поток
    handlers = [logging.StreamHandler()]
    # Для вывода логов в file, если существует
    if file:
        handlers.append(logging.FileHandler(file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)




def main(argv=None):
    args = parse_cli_args(argv)
    setup_log(args.log_file)
    try:
        df = load_df(args.input)
        logging.info(f"Всего строк: {len(df)}")
        # Удаление редких классов
        df,_ = drop_rare_rows(df)
        # Разделение на тренировочную и тестовую выборки
        train_df, test_df = split_by_label_and_time(df)
        # Повторная проверка на редкие классы после разделения
        combined = pd.concat([train_df, test_df], ignore_index=True)
        filtered_df,_ = drop_rare_rows(combined)
        # Финальное разделение после чистки
        train_df, test_df = split_by_label_and_time(filtered_df)
        # Балансировка класса Normal
        if not args.no_balance:
            train_df = undersample_majority(train_df)
        logging.info(f"Итоговые размеры: train={len(train_df)} | test={len(test_df)}")
        # Сохранение датасетов
        save_parquets(train_df, test_df, args.out_dir)
        if args.save_csv:
            save_distribution(train_df, test_df, args.out_dir)
        logging.info("Завершено без ошибок")

    except Exception as exc:
        logging.exception(f"Ошибка выполнения: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
