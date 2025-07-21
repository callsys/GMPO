from datasets import load_from_disk
import pandas as pd

def convert_single_dataset(input_path, output_path, data_source_name):
    """Convert a single Arrow dataset to target parquet format."""
    
    # Load Arrow dataset
    dataset = load_from_disk(input_path)
    
    # Handle dataset dict
    if hasattr(dataset, 'items'):
        dataset = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
    
    # Convert format
    converted_data = []
    for idx, example in enumerate(dataset):
        record = {
            "data_source": "",
            "prompt": [{"content": example["problem"], "role": "user"}],
            "ability": "math",
            "reward_model": {"ground_truth": example["answer"], "style": "rule"},
            "extra_info": {"index": idx, "split": data_source_name}
        }
        converted_data.append(record)
    
    # Save as parquet
    df = pd.DataFrame(converted_data)
    df.to_parquet(output_path, index=False)
    print(f"Converted {len(converted_data)} examples to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Convert train dataset
    convert_single_dataset(
        "./understand_r1_zero_main/datasets/train/math_lvl3to5_8k", 
        "/mnt/msranlp_yaru/yaru/data/rl/understand_r1_zero_main/train/math_lvl3to5_8k.parquet",
        "train"
    )

    # Convert evaluation dataset
    test_data_names = ["aime", "math", "amc", "minerva", "olympiad_bench"]
    for data_name in test_data_names:
        convert_single_dataset(
            f"./understand_r1_zero_main/datasets/evaluation_suite/{data_name}",
            f"/mnt/msranlp_yaru/yaru/data/rl/understand_r1_zero_main/evaluation_suite/{data_name}.parquet",
            "test"
        )