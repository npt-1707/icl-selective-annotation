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
parser.add_argument("--model_name", default="gpt-4o-mini", type=str)
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
    output_dir = os.path.join(args.output_dir, args.task_name, "zero_shot")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    result_cache_dir = os.path.join(output_dir, "results")
    if not os.path.isdir(result_cache_dir):
        os.makedirs(result_cache_dir, exist_ok=True)
    prompt_cache_dir = os.path.join(output_dir, "prompts")
    if not os.path.isdir(prompt_cache_dir):
        os.makedirs(prompt_cache_dir, exist_ok=True)
    logger = Logger(os.path.join(output_dir, "zero_shot.log"), args.task_name, file_mode="w")
    logger.log(f"Zero_shot {args.task_name} evaluation")

    labels = []
    preds = []

    for id, instance in enumerate(eval_examples):
        file_name = (
            f"{instance['id']}.json"
            if "id" in instance
            else (
                f"{instance['commit_id']}.json"
                if "commit_id" in instance
                else instance[0]["cve_list"]
                + "-".join(instance[0]["repo"].split("/"))
                + instance[0]["commit_id"]
                + ".json"
            )
        )
        result_file = os.path.join(result_cache_dir, file_name)
        if not os.path.isfile(result_file):
            promt_file = os.path.join(prompt_cache_dir, file_name)
            if os.path.isfile(promt_file):
                with open(promt_file, "r") as f:
                    question = json.load(f)[1]
            else:
                question, _ = format_example(instance, label_map=label_map)
                question = tiktoken_truncate(question, max_len=single_input_len * 2)
                with open(promt_file, "w") as f:
                    json.dump([id, question, instance], f, indent=4)
                try:
                    gpt_completion(
                        key=GPT_KEY,
                        output_path=result_file,
                        prompt_path=promt_file,
                        sys_prompt_path=os.path.join("sys_prompts", f"{args.task_name}.json"),
                        model_name=args.model_name,
                        logprobs=False,
                    )
                    existed = True
                except Exception as e:
                    print(e)
                    time.sleep(1)
        if args.task_name == "vulfix":
            label = instance["label"]
            with open(os.path.join(result_file), "r") as f:
                pred_dict = json.load(f)
            prediction = pred_dict["choices"][0]["message"]["content"].lower()
            pred = match_answer(args.task_name, prediction)
            if pred == "yes":
                pred = 1
            elif pred == "no":
                pred = 0
            else:
                logger.log(f"Undefined result in :{result_file}\n{prediction}")
                pred = 1 - label
            labels.append(label)
            preds.append(pred)
        elif args.task_name == "treevul":
            label = instance[0]["cwe_list"]
            with open(os.path.join(result_file), "r") as f:
                pred_dict = json.load(f)
            prediction = pred_dict["choices"][0]["message"]["content"].lower()
            pred = match_answer(args.task_name, prediction)
            labels.append(label)
            preds.append(pred)
        bar.update(1)
    bar.close()
    logger.log("\n" + classification_report(labels, preds, zero_division=0))
