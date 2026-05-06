MODE = "inspect"

if MODE=="inspect":
    import pyarrow.parquet as pq
    import json
    import pprint

    def inspect_parquet_meta(file_path):
        # 读取 Parquet 文件元数据
        schema = pq.read_schema(file_path)
        
        print(f"--- 文件: {file_path.name} ---")
        
        # 1. 查看 Schema（列名和数据类型）
        print("\n[Schema 信息]:")
        for name, type in zip(schema.names, schema.types):
            print(f"列名: {name:20} | 类型: {type}")
            
        # 2. 查看 Custom Metadata (这是 Hugging Face 存储特征定义的关键位置)
        print("\n[Custom Metadata (自定义元数据)]:")
        if schema.metadata:
            for key, value in schema.metadata.items():
                # 这里的 key 通常是 b'huggingface' 或 b'pandas'
                print(f"Key: {key}")
                try:
                    # 尝试解码并美化打印 JSON 内容
                    decoded_val = json.loads(value.decode('utf-8'))
                    pprint.pprint(decoded_val)
                except:
                    print(f"Value: {value[:100]}... (无法作为JSON解析)")
        else:
            print("无自定义元数据")
        print("-" * 50)

    # 替换为你实际的文件路径
    from pathlib import Path
    good_file = Path("/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/019_20260115_lyx_pour_wine_8_3_hzttest/data/chunk-000/episode_000000.parquet")
    bad_file = Path("/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/merge_001-010/data/chunk-000/episode_000000.parquet")
    # bad_file = Path("/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/018_20260115_lyx_pour_wine_8_3/data/chunk-000/episode_000000.parquet")

    print("正常文件")
    inspect_parquet_meta(good_file)
    print("转换错误文件")
    inspect_parquet_meta(bad_file)

elif MODE=="fix":
    import pyarrow.parquet as pq
    import pyarrow as pa
    from pathlib import Path
    from tqdm import tqdm

    # 1. 设定路径
    data_dir = Path("/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/merge_001-010")
    episodes_dir = data_dir / "data" / "chunk-000"
    template_file = Path("/mnt/nvme0n1/nvme1n1/dataset_1w/dst_dataset/pour_wine/019_20260115_lyx_pour_wine_8_3_hzttest/data/chunk-000/episode_000000.parquet") 

    # 2. 获取metadata
    templ_pq = pq.ParquetFile(template_file)
    perfect_schema = templ_pq.schema_arrow
    perfect_metadata = perfect_schema.metadata # 这里包含了 b'huggingface' 的所有二进制信息

    # 3. 开始修复
    parquet_files = list(episodes_dir.glob("*.parquet"))
    pbar = tqdm(parquet_files)

    for p_file in pbar:
        pbar.set_description(f"Injecting HF Meta: {p_file.name}")
        try:
            # 读取损坏文件的数据
            bad_table = pq.read_table(p_file)
            
            # 关键一步：重新构造 Schema，强行把完美 metadata 塞进去
            # 这会替换掉那个讨厌的 b'pandas' 标签
            new_schema = perfect_schema.with_metadata(perfect_metadata)
            
            # 使用坏数据 + 好 Schema 组装新表
            # 注意：bad_table.cast(new_schema) 可以确保 fixed_size_list 等类型强制转换成功
            fixed_table = bad_table.cast(new_schema)
            
            # 4. 覆盖写入
            pq.write_table(fixed_table, p_file, compression='snappy')
            
        except Exception as e:
            pbar.write(f"修复失败 {p_file.name}: {e}")

    print("\n✨ 修复完成！所有文件的 Schema 和 HuggingFace 元数据已恢复。")