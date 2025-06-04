import os
import json
import pandas as pd
import numpy as np
import argparse

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_dataset import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def make_prefix(dp, template_type):
    problem = dp['question']
    prefix = (
        f"Please solve the following math problem: {problem}. "
        "The assistant first thinks about the reasoning process step by step and then provides the user with the answer. "
        "Return the final answer in \\boxed{{}} tags, for example \\boxed{{1}}. Let's solve this step by step."
    )
    return prefix


def load_local_jsonl(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_path', default='./raw-data/gaokao-2023en-test.jsonl')
    parser.add_argument('--local_dir', default='./data/gaokao')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    # Load local JSONL data
    dataset = load_local_jsonl(args.jsonl_path)

    processed_data = []

    for idx, example in enumerate(dataset):
        # random_int = np.random.randint(0, 10)

        question = make_prefix(example, template_type=args.template_type)
        solution = example['answer']
        #remove $ signs in solution
        solution = solution.replace('$', '')

        # if random_int == 0:
        #     print(f">>> Question: \n{question}")
        #     print(f">>> Solution: \n{solution}")

        data = {
            "data_source": "gaokao",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': 'train',
                'index': idx
            }
        }
        processed_data.append(data)

    # Save to Parquet
    os.makedirs(args.local_dir, exist_ok=True)
    output_path = os.path.join(args.local_dir, 'test.parquet')
    pd.DataFrame(processed_data).to_parquet(output_path, index=False)

    # print length of processed data
    print(f"Length of processed data: {len(processed_data)}")

    # Optionally copy to HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
