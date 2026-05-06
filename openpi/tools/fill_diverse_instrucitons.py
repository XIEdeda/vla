import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import random

data_dir = Path("/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/018_20260115_lyx_pour_wine_8_3")
tasks_file = data_dir / "meta" / "tasks.jsonl"
episodes_dir = data_dir / "data" / "chunk-000"

tasks_list = [
    "pour wine",
    "pick up the cup and pour wine from the dispenser",
    "hold the cup under the wine outlet and dispense wine",
    "place the cup at the dispenser and pour wine",
    "grasp the cup, activate the dispenser, and fill it with wine",
    "dispense a glass of wine",
    "serve some wine from the machine",
    "refill the glass with wine",
    "get some wine from the tap",
    "initiate wine pouring",
    "start the wine dispenser",
    "fill the vessel with wine",
    "draw wine from the dispenser",
    "fetch a pour of wine",
    "transfer wine from the tank to the glass",
    "pour a drink of wine",
    "operate the dispenser to pour wine",
    "let the wine flow into the cup",
    "provide a serving of wine",
    "execute the wine pouring sequence"
]
json_lines = []
for index, task in enumerate(tasks_list):
    task_dict = {"task_index": index, "task": task}
    json_lines.append(json.dumps(task_dict))
tasks = "\n".join(json_lines)

with open(tasks_file, "w") as f:
    f.write(tasks)

parquet_files = list(episodes_dir.glob("*.parquet"))
pbar = tqdm(parquet_files)

for p_file in pbar:
    pbar.set_description(f"Processing {p_file.name}")
    
    table = pq.read_table(p_file)
    
    # 检查列是否存在
    if 'episode_index' not in table.column_names:
        pbar.write(f"Skipping {p_file.name}: 'episode_index' not found.")
        continue

    df_indices = table.select(['episode_index']).to_pandas()
    
    unique_episodes = df_indices['episode_index'].unique()
    mapping = {ep: np.random.randint(0, len(tasks_list)) for ep in unique_episodes}
    
    new_task_indices = df_indices['episode_index'].map(mapping).astype(np.int64)
    
    if 'task_index' in table.column_names:
        table = table.drop(['task_index'])
    
    table = table.append_column('task_index', pa.array(new_task_indices))
    
    pq.write_table(table, p_file, compression='snappy')
