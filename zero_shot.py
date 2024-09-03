import argparse
import random
import os
import torch
import numpy as np
import json
import time

from tqdm import tqdm
from sklearn.metrics import classification_report
from get_task import get_task
from utils import (
    gpt_completion, match_answer,
    Logger,
)
from two_steps import tiktoken_truncate
from dotenv import load_dotenv

load_dotenv()
GPT_KEY = os.getenv("GPT_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="vulfix", type=str)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--data_cache_dir", required=True, type=str)
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

        


if __name__ == "__main__":
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    _, eval_examples, _, _, format_example, label_map = get_task(args=args)

    # print("Eval example 0:")
    # print(eval_examples[0])
    # print("Formatted example 0:")
    # print(format_example(train_examples[0],label_map=label_map,args=args))
    # print("Label map:")
    # print(label_map)

    single_input_len = 1024

    bar = tqdm(range(len(eval_examples)), desc="Generate and Evaluate 0-shot results")
    ouput_cache_dir = os.path.join(args.output_dir, "zero_shot_output")
    if not os.path.isdir(ouput_cache_dir):
        os.makedirs(ouput_cache_dir, exist_ok=True)
    prompt_cache_dir = os.path.join(args.output_dir, "zero_shot_prompt")
    if not os.path.isdir(prompt_cache_dir):
        os.makedirs(prompt_cache_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, "zero_shot.log"), file_mode="w")
    logger.log("Zero_shot Vulfix evaluation")

    labels = []
    preds = []
    MAX_FETCH_TIMES = 5

    for id, instance in enumerate(eval_examples):
        result_file = os.path.join(ouput_cache_dir, f"{instance['commit_id']}.json")
        if not os.path.isfile(result_file):
            promt_file = os.path.join(prompt_cache_dir, f"{instance['commit_id']}.json")
            if os.path.isfile(promt_file):
                with open(promt_file, "r") as f:
                    question = json.load(f)[1]
            else:
                question, _ = format_example(instance, label_map=label_map)
                question = tiktoken_truncate(question, max_len=single_input_len * 2)
                with open(promt_file, "w") as f:
                    json.dump([id, question, instance], f, indent=4)
            fail_count = 0
            existed = False
            while fail_count < MAX_FETCH_TIMES and not existed:
                try:
                    gpt_completion(
                        key=GPT_KEY,
                        output_path=result_file,
                        prompt_path=promt_file,
                        logprobs=False,
                    )
                    existed = True
                except Exception as e:
                    print(e)
                    fail_count += 1
                    logger.error(f"Error in {instance['commit_id']}, retrying {fail_count}/3...")
                    time.sleep(3)
            if fail_count == MAX_FETCH_TIMES:
                logger.error(f"Failed to fetch API for {instance['commit_id']}.")
        label = instance["label"]
        with open(os.path.join(result_file), "r") as f:
            pred_dict = json.load(f)
        prediction = pred_dict["choices"][0]["message"]["content"].lower()
        pred = match_answer(prediction)
        if pred == "yes":
            pred = 1
        elif pred == "no":
            pred = 0
        else:
            logger.log(f"Undefined result in :{result_file}\n{prediction}")
            pred = 1 - label
        labels.append(label)
        preds.append(pred)
        bar.update(1)
    bar.close()
    logger.log("\n" + classification_report(labels, preds))
    