import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, List, Union  # 导入Union用于联合类型注解

def check_parquet_file(parquet_path: Path) -> Dict[str, Any]:
    """检查单个Parquet文件的元数据和Schema"""
    result = {
        "file": str(parquet_path),
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        # 读取Parquet文件
        table = pq.read_table(parquet_path)
        schema = table.schema
        metadata = schema.metadata
        
        # 1. 检查huggingface元数据是否存在
        if b"huggingface" not in metadata:
            result["valid"] = False
            result["errors"].append("未找到'huggingface'元数据")
            return result
        
        # 解析huggingface元数据
        try:
            huggingface_metadata = json.loads(metadata[b"huggingface"].decode("utf-8"))
        except json.JSONDecodeError as e:
            result["valid"] = False
            result["errors"].append(f"元数据解析失败: {str(e)}")
            return result
        
        # 2. 验证元数据结构是否完整
        required_keys = ["info", "info.features"]
        for key in required_keys:
            parts = key.split(".")
            current = huggingface_metadata
            try:
                for part in parts:
                    current = current[part]
            except KeyError:
                result["valid"] = False
                result["errors"].append(f"缺少必要字段: {key}")
        
        # 3. 定义预期的字段配置
        expected_features = {
            "head_image": {"_type": "Image"},
            "left_wrist_image": {"_type": "Image"},
            "right_wrist_image": {"_type": "Image"},
            "state": {
                "_type": "Sequence", 
                "feature": {"_type": "Value", "dtype": "float32"}
            },
            "actions": {
                "_type": "Sequence", 
                "feature": {"_type": "Value", "dtype": "float32"}
            },
            "timestamp": {"_type": "Value", "dtype": "float32"},
            "frame_index": {"_type": "Value", "dtype": "int64"},
            "episode_index": {"_type": "Value", "dtype": "int64"},
            "index": {"_type": "Value", "dtype": "int64"},
            "task_index": {"_type": "Value", "dtype": "int64"}
        }
        
        # 4. 检查每个字段的元数据定义
        actual_features = huggingface_metadata.get("info", {}).get("features", {})
        for field_name, expected in expected_features.items():
            if field_name not in actual_features:
                result["valid"] = False
                result["errors"].append(f"元数据缺少字段定义: {field_name}")
                continue
            
            actual = actual_features[field_name]
            
            if actual.get("_type") != expected.get("_type"):
                result["valid"] = False
                result["errors"].append(
                    f"字段{field_name}的_type不匹配: "
                    f"预期{expected['_type']}, 实际{actual.get('_type')}"
                )
            
            if expected["_type"] == "Sequence":
                expected_feature = expected.get("feature", {})
                actual_feature = actual.get("feature", {})
                
                if actual_feature.get("_type") != expected_feature.get("_type"):
                    result["valid"] = False
                    result["errors"].append(
                        f"字段{field_name}的内部feature._type不匹配: "
                        f"预期{expected_feature['_type']}, 实际{actual_feature.get('_type')}"
                    )
                
                if actual_feature.get("dtype") != expected_feature.get("dtype"):
                    result["valid"] = False
                    result["errors"].append(
                        f"字段{field_name}的内部feature.dtype不匹配: "
                        f"预期{expected_feature['dtype']}, 实际{actual_feature.get('dtype')}"
                    )
            
            elif expected["_type"] == "Value":
                if actual.get("dtype") != expected.get("dtype"):
                    result["valid"] = False
                    result["errors"].append(
                        f"字段{field_name}的dtype不匹配: "
                        f"预期{expected['dtype']}, 实际{actual.get('dtype')}"
                    )
        
        # 5. 检查Schema字段类型
        expected_schema = {
            "head_image": pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
            "left_wrist_image": pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
            "right_wrist_image": pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]),
            "state": pa.list_(pa.float32()),
            "actions": pa.list_(pa.float32()),
            "timestamp": pa.float32(),
            "frame_index": pa.int64(),
            "episode_index": pa.int64(),
            "index": pa.int64(),
            "task_index": pa.int64()
        }
        
        for field_name, expected_type in expected_schema.items():
            if field_name not in schema.names:
                result["valid"] = False
                result["errors"].append(f"Schema缺少字段: {field_name}")
                continue
            
            actual_type = schema.field(field_name).type
            
            if not actual_type.equals(expected_type):
                result["valid"] = False
                result["errors"].append(
                    f"字段{field_name}的Schema类型不匹配: "
                    f"预期{expected_type}, 实际{actual_type}"
                )
        
        # 6. 检查数据示例
        if table.num_rows > 0:
            row = table.take([0]).to_pylist()[0]
            
            if not isinstance(row.get("state"), list):
                result["valid"] = False
                result["errors"].append(f"state不是列表类型，实际类型: {type(row.get('state'))}")
            
            if not isinstance(row.get("actions"), list):
                result["valid"] = False
                result["errors"].append(f"actions不是列表类型，实际类型: {type(row.get('actions'))}")
                
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"处理文件时出错: {str(e)}")
    
    return result

# 使用Union[str, Path]替代str or Path，修复类型注解问题
def check_parquet_directory(directory_path: Union[str, Path]) -> Dict[str, Any]:
    """检查目录下所有Parquet文件"""
    directory_path = Path(directory_path)
    results = {
        "directory": str(directory_path),
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "file_results": []
    }
    
    if not directory_path.exists():
        results["error"] = f"目录不存在: {directory_path}"
        return results
    
    if not directory_path.is_dir():
        results["error"] = f"不是目录: {directory_path}"
        return results
    
    # 查找所有Parquet文件
    parquet_files = list(directory_path.glob("*.parquet"))
    results["total_files"] = len(parquet_files)
    
    if results["total_files"] == 0:
        results["warning"] = "目录下没有找到Parquet文件"
        return results
    
    # 检查每个文件
    for file in parquet_files:
        file_result = check_parquet_file(file)
        results["file_results"].append(file_result)
        
        if file_result["valid"]:
            results["valid_files"] += 1
        else:
            results["invalid_files"] += 1
    
    return results

def print_directory_check_results(results: Dict[str, Any]):
    """打印目录检查结果"""
    print(f"检查目录: {results['directory']}")
    print(f"总文件数: {results['total_files']}")
    print(f"有效文件: {results['valid_files']}")
    print(f"无效文件: {results['invalid_files']}\n")
    
    if "error" in results:
        print(f"错误: {results['error']}")
        return
    
    if "warning" in results:
        print(f"警告: {results['warning']}")
    
    # 打印无效文件的错误信息
    if results["invalid_files"] > 0:
        print("\n无效文件详情:")
        for i, file_result in enumerate(results["file_results"], 1):
            if not file_result["valid"]:
                print(f"\n文件 {i}: {file_result['file']}")
                for j, error in enumerate(file_result["errors"], 1):
                    print(f"  错误 {j}: {error}")

if __name__ == "__main__":
    # 直接在代码中指定路径
    # dir_path = "/home/xiededa/VLA/pi0_dataset/lerobot_dataset_1016/data/chunk-000"
    # dir_path = "/home/xiededa/VLA/pi0_dataset/lerobot_dataset_1111/data/chunk-000"
    dir_path = "/root/xie/dual_arm/data/chunk-000"
    # dir_path = "/home/xiededa/VLA/pi0_dataset"    
    results = check_parquet_directory(dir_path)
    print_directory_check_results(results)
    