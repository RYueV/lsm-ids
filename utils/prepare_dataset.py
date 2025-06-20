import json
import pathlib
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Папка с исходными csv
RAW_DIR = pathlib.Path("data/raw")
# Папка для сохранения результатов
CLEAN_DIR = pathlib.Path("data/clean")
# Возможные кодировки исходных файлов
POSSIBLE_ENCODINGS = (
    "utf-8",
    "latin1",
    "cp1252"
)


# Максимальное количество строк, обрабатываемых за раз при потоковой обработке
NUM_STR = 100000
# Столбцы, которые удаляем
DROP_COLS = [
    "Flow ID",                      # номер записи не содержит полезной информации для классификации
    "Source IP", "Destination IP",  # IP источника и получателя нельзя использовать (утечка информации)
    "Source Port",                  # порт источника не содержит полезной информации для классификации
    "Fwd Header Length.1",          # дубликат (такой столбец встречается в датасете дважды)
]
# Разделение возможных меток трафика по 6 группам атак + Normal
# (Normal, Bot, BruteForce, DoS_DDoS, Infiltration, PortScan, WebAttack)
LABEL_MAP = {
    "BENIGN": "Normal",
    "Bot": "Bot",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "DDoS": "DoS_DDoS",
    "DoS Hulk": "DoS_DDoS",
    "DoS GoldenEye": "DoS_DDoS",
    "DoS Slowhttptest": "DoS_DDoS",
    "DoS slowloris": "DoS_DDoS",
    "Heartbleed": "DoS_DDoS",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack"
}

# Список признаков с чрезмерно широким диапазоном min/max значений
WIDE_RANGE_FEATURES = {
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Total Length of Bwd Packets",
    "Total Length of Fwd Packets",
    "Packet Length Variance",
    "Destination Port",
    "Fwd Packet Length Max",
    "Bwd Packet Length Max",
    "Bwd Packet Length Std",
    "Fwd Header Length",
    "Bwd Header Length",
    "Max Packet Length",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Packet Length Mean",
    "Packet Length Std",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size"
}
# Шаблон для определения наименований признаков, которые могут иметь широкий диапазон
WR_AUTODETECT_PATTERN = re.compile(r"(Duration|IAT|Active|Idle|Bytes/s|Packets/s|Variance)", re.I)



# Защита от экстремальных выбросов в значениях числовых признаков
# (ограничение значений по перцентилям)
def _clip_by_percentiles(
        df,             # pandas DataFrame с признаками
        col_names       # список наименований колонок, в которых нужно устранить выбросы
    ):
    for name in col_names:
        val_min, val_max = df[name].quantile([0.001, 0.999])
        df[name] = df[name].clip(val_min, val_max)
    return df


# Логарифмирование и ограничение значений числовых признаков с широким диапазоном
def _log_and_clip_scaling(
        df,             # pandas DataFrame с признаками
        col_names       # список наименований колонок, в которых нужно устранить выбросы
    ):
    for name in col_names:
        if name in WIDE_RANGE_FEATURES or WR_AUTODETECT_PATTERN.search(name):
            df[name] = np.log10(df[name].clip(lower=0) + 1)
        val_min, val_max = df[name].quantile([0.001, 0.999])
        df[name] = df[name].clip(val_min, val_max)
    return df




# Обработка одного csv-файла
def _process_csv(
        csv_path,               # путь к одному из csv-файлов исходного датасета
        writer,                 # объект для записи данных в parquet
        global_minmax           # словарь min/max значений каждого числового признака
    ):
    print(f"[PROCESS] Обрабатывается {csv_path.name}")

    # Перебор возможных кодировок
    for enc in POSSIBLE_ENCODINGS:
        try:
            # Флаг корректной обработки данных
            prcs_ok = True

            # Чтение файла фрагментами размером NUM_STR
            reader = pd.read_csv(csv_path, chunksize=NUM_STR, low_memory=False, encoding=enc)

            # Для каждого фрагмента данных
            for batch in reader:
                # Удаление пробелов из названий колонок
                batch.columns = batch.columns.str.strip()
                # Замена бесконечных значений на NaN
                batch.replace([np.inf, -np.inf], np.nan, inplace=True)
                # Удаление NaN значений
                batch.dropna(inplace=True)

                # Пропуск блока, если все строки были NaN
                if batch.empty:
                    continue

                # Удаление лишних колонок
                batch.drop(columns=[col for col in DROP_COLS if col in batch], inplace=True)

                # Поиск колонки с временными метками
                time_col = next((col for col in batch.columns if "Timestamp" in col), None)

                # Если такой нет, файл обработать нельзя (для snn нужны временные метки)
                if time_col is None:
                    prcs_ok = False
                    break

                # Приведение значений в колонке временных меток в тип datetime
                batch[time_col] = pd.to_datetime(batch[time_col], errors="coerce")
                batch.rename(columns={time_col: "Timestamp"}, inplace=True)
                # Удаление строк где не получилось обработать время
                batch.dropna(subset=["Timestamp"], inplace=True)
                if batch.empty:
                    continue

                # Файлы без меток трафика тоже нельзя обработать
                if "Label" not in batch.columns:
                    prcs_ok = False
                    break

                # Группировка типов записей по 7 группам (6 видов атак + normal)
                batch["Label"] = batch["Label"].map(LABEL_MAP)
                batch.dropna(subset=["Label"], inplace=True)
                if batch.empty:
                    continue

                # Колонки с числовыми признаками приводятся к float64
                int_cols = batch.select_dtypes(include=["int", "int64", "int32"]).columns
                batch[int_cols] = batch[int_cols].astype("float64")

                # Колонки с категориальными - к числовым
                obj_cols = [
                    col for col in batch.select_dtypes(include=["object"]).columns
                    if col not in ("Timestamp", "Label")
                ]
                for col in obj_cols:
                    batch[col] = pd.to_numeric(batch[col].str.replace(',', ''), errors="coerce")
                batch.dropna(inplace=True)
                if batch.empty:
                    continue

                # Ограничение и логарифмирование значений
                num_cols = batch.select_dtypes(include=[np.number]).columns
                batch = _clip_by_percentiles(batch, num_cols)
                batch = _log_and_clip_scaling(batch, num_cols)

                # Сохранение новых диапазонов возможных значений
                for col in num_cols:
                    val_min, val_max = float(batch[col].min()), float(batch[col].max())
                    if col not in global_minmax:
                        global_minmax[col] = [val_min, val_max]
                    else:
                        global_minmax[col][0] = min(global_minmax[col][0], val_min)
                        global_minmax[col][1] = max(global_minmax[col][1], val_max)

                writer.write_table(pa.Table.from_pandas(batch, preserve_index=False))

        except UnicodeDecodeError:
            prcs_ok = False
        if prcs_ok:
            break
        else:
            print(f"[WARNING] Для {csv_path.name} кодировка {enc} не подходит, переход к другой")
            continue
    else:
        print(f"[ERROR] Невозможно обработать {csv_path.name}")




# Запуск обработки исходного датасета
def prepare(rewrite=False):
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] В {RAW_DIR} нет csv-файлов")
        return
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = CLEAN_DIR / "cicids2017_clean.parquet"
    out_ranges = CLEAN_DIR / "feature_ranges.json"
    if out_parquet.exists() and not rewrite:
        print("[WARNING]", out_parquet.name, "уже создан")
        return

    # Обработка небольшого фрагмента первого файла для составления схемы (таблицы)
    # Принцип тот же, что и в _process_csv
    enc = "utf-8"
    try:
        df_prev = pd.read_csv(csv_files[0], nrows=3000, low_memory=False, encoding=enc)
    except UnicodeDecodeError:
        enc = "latin1"
        df_prev = pd.read_csv(csv_files[0], nrows=3000, low_memory=False, encoding=enc)
    df_prev.columns = df_prev.columns.str.strip()
    df_prev.drop(columns=[col for col in DROP_COLS if col in df_prev], inplace=True)
    time_col = next((c for c in df_prev.columns if "Timestamp" in c), None)
    if time_col:
        df_prev[time_col] = pd.to_datetime(df_prev[time_col], errors="coerce")
        df_prev.rename(columns={time_col: "Timestamp"}, inplace=True)
    int_col = df_prev.select_dtypes(include=["int", "int64", "int32"]).columns
    df_prev[int_col] = df_prev[int_col].astype("float64")
    obj_col = [
        col for col in df_prev.select_dtypes(include=["object"]).columns 
        if col not in ("Timestamp", "Label")
    ]
    for col in obj_col:
        df_prev[col] = pd.to_numeric(df_prev[col].str.replace(',', ''), errors="coerce")
    df_prev.dropna(inplace=True)
    nums = df_prev.select_dtypes(include=[np.number]).columns
    df_prev = _clip_by_percentiles(df_prev, nums)
    df_prev = _log_and_clip_scaling(df_prev, nums)

    # Составление схему
    schema = pa.Table.from_pandas(df_prev[:1], preserve_index=False).schema

    # Обработка всех файлов и сохранение новых диапазонов и чистого датасета
    global_minmax = {}
    with pq.ParquetWriter(out_parquet, schema, compression="snappy") as writer:
        for csv_path in csv_files:
            _process_csv(csv_path, writer, global_minmax)
    with out_ranges.open("w", encoding="utf-8") as fp:
        json.dump(global_minmax, fp, ensure_ascii=False, indent=2)

    print(f"[DONE] Сохранены в {CLEAN_DIR}")




if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rewrite", action="store_true", help="Перезаписать существующие файлы")
    args = p.parse_args()
    prepare(rewrite=args.rewrite)
