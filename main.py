import argparse
import random
import os

# import copy
import torch
import numpy as np
import json
import nltk
import time

# import time
# from torch import nn
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer, GPTJForCausalLM
from dotenv import load_dotenv

load_dotenv()
# from sentence_transformers import SentenceTransformer
# from datasets import load_dataset
# from sklearn.metrics import f1_score
# from MetaICL.metaicl.data import MetaICLData
# from MetaICL.metaicl.model import MetaICLModel

# from collections import defaultdict
from get_task import get_task
from utils import (
    calculate_sentence_transformer_embedding,
    codex_execution,
    expand_to_aliases,
    gpt_completion,
    match_answer,
    Logger,
)
from two_steps import selective_annotation, prompt_retrieval
from embedding_model import get_embedding_model

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", required=True, type=str)
parser.add_argument("--selective_annotation_method", required=True, type=str)
parser.add_argument("--model_cache_dir", type=str)
parser.add_argument("--data_cache_dir", required=True, type=str)
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--model_key", type=str)
parser.add_argument("--prompt_retrieval_method", default="similar", type=str)
parser.add_argument("--model_name", default="gpt-4o-mini", type=str)
parser.add_argument("--emb_model", default="bert", type=str)
parser.add_argument("--emb_batch_size", default=10, type=int)
parser.add_argument("--use_diff", action="store_true")
parser.add_argument("--annotation_size", default=100, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


if __name__ == "__main__":
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    (
        train_examples,
        eval_examples,
        train_text_to_encode,
        eval_text_to_encode,
        format_example,
        label_map,
    ) = get_task(args=args)
    emb_model = get_embedding_model(args.emb_model, use_diff=args.use_diff)
    output_dir = os.path.join(
        args.output_dir,
        args.task_name,
        args.emb_model if args.use_diff else f"{args.emb_model}_only_message",
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if args.task_name in ["vulfix", "treevul"]:
        train_embed_file = os.path.join(
            output_dir, f"total_train_embeds_{args.seed}.npy"
        )
        eval_embed_file = os.path.join(
            output_dir, f"total_eval_embeds_{args.seed}.npy"
        )
        if os.path.isfile(train_embed_file) and os.path.isfile(eval_embed_file):
            total_train_embeds = np.load(train_embed_file)
            total_eval_embeds = np.load(eval_embed_file)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            total_train_embeds = emb_model.calculate_sentence_embedding(
                sentences=train_text_to_encode,
                device=device,
                emb_batch_size=args.emb_batch_size,
            )
            np.save(
                train_embed_file,
                total_train_embeds,
            )
            total_eval_embeds = emb_model.calculate_sentence_embedding(
                sentences=eval_text_to_encode,
                device=device,
                emb_batch_size=args.emb_batch_size,
            )
            np.save(
                eval_embed_file,
                total_eval_embeds,
            )
    else:

        total_train_embeds = calculate_sentence_transformer_embedding(
            text_to_encode=train_text_to_encode, args=args
        )
        total_eval_embeds = calculate_sentence_transformer_embedding(
            text_to_encode=eval_text_to_encode, args=args
        )

    if args.task_name in [
        "mnli",
        "rte",
        "sst5",
        "mrpc",
        "dbpedia_14",
        "hellaswag",
        "xsum",
        "nq",
        "vulfix",
        "treevul",
    ]:
        if args.task_name == "xsum":
            tokenizer_gpt = AutoTokenizer.from_pretrained(
                args.model_name, cache_dir=args.model_cache_dir
            )
            inference_model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", cache_dir=args.model_cache_dir
            )
            inference_model.cuda()
            inference_model.eval()
            data_module = None
            return_string = True
            device = torch.device("cuda")
            single_input_len = None
            maximum_input_len = 1900
        elif args.task_name == "nq":
            maximum_input_len = 3800
            return_string = True
            single_input_len = None
            inference_model = None
            data_module = None
            tokenizer_gpt = None
            model_keys = args.model_key.split("##")
        elif args.task_name in ["vulfix", "treevul"]:
            maximum_input_len = 16384
            single_input_len = 1024
            inference_model = None
            data_module = None
            tokenizer_gpt = None
            return_string = True
        else:
            data_module = MetaICLData(
                method="direct", max_length=1024, max_length_per_example=256
            )
            inference_model = MetaICLModel(args=args)
            inference_model.load()
            inference_model.cuda()
            inference_model.eval()
            tokenizer_gpt = None
            return_string = False
            single_input_len = 250
            maximum_input_len = 1000

        output_dir = os.path.join(output_dir, args.selective_annotation_method)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if args.selective_annotation_method == "random":
            first_phase_selected_indices_file = os.path.join(
                output_dir, f"first_phase_selected_indices_{args.seed}.json"
            )
        else:
            first_phase_selected_indices_file = os.path.join(
                output_dir, "first_phase_selected_indices.json"
            )
        if os.path.isfile(first_phase_selected_indices_file):
            with open(first_phase_selected_indices_file, "r") as f:
                first_phase_selected_indices = json.load(f)
        else:
            first_phase_selected_indices = selective_annotation(
                embeddings=total_train_embeds,
                train_examples=train_examples,
                return_string=return_string,
                format_example=format_example,
                maximum_input_len=maximum_input_len,
                label_map=label_map,
                single_context_example_len=single_input_len,
                inference_model=inference_model,
                inference_data_module=data_module,
                tokenizer_gpt=tokenizer_gpt,
                output_dir=output_dir,
                args=args,
            )
            with open(
                first_phase_selected_indices_file,
                "w",
            ) as f:
                json.dump(first_phase_selected_indices, f, indent=4)
        processed_train_examples = [
            train_examples[idx] for idx in first_phase_selected_indices
        ]
        processed_eval_examples = eval_examples

        prompt_identifier = (
            f"prompts_{args.seed}"
            if args.selective_annotation_method == "random"
            else "prompts"
        )
        prompt_cache_dir = os.path.join(output_dir, prompt_identifier)
        if not os.path.isdir(prompt_cache_dir):
            os.makedirs(prompt_cache_dir, exist_ok=True)
        prompt_retrieval(
            train_embs=total_train_embeds[first_phase_selected_indices],
            test_embs=total_eval_embeds,
            train_examples=processed_train_examples,
            eval_examples=eval_examples,
            return_string=return_string,
            format_example=format_example,
            maximum_input_len=maximum_input_len,
            single_context_example_len=single_input_len,
            label_map=label_map,
            prompt_cache_dir=prompt_cache_dir,
            args=args,
        )

        candidate_prompt_files = os.listdir(prompt_cache_dir)
        prompt_files = [f for f in candidate_prompt_files if f.endswith(".json")]
        assert len(prompt_files) == len(processed_eval_examples), (
            f"len(prompt_files)={len(prompt_files)},"
            f"len(processed_eval_examples)={len(processed_eval_examples)}"
        )
        result_dir = os.path.join(output_dir, f"results_{args.model_name}")
        if args.selective_annotation_method == "random":
            result_dir += f"_{args.seed}"
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        count = 0
        running_flag = True
        golds = []
        preds = []
        if not args.task_name in ["hellaswag", "xsum", "nq", "treevul"]:
            all_labels = []
            label_to_digit = {}
            for k, v in label_map.items():
                all_labels.append(v)
                label_to_digit[v] = k
        execution_count = 0
        while running_flag:
            running_flag = False
            count += 1
            bar = tqdm(range(len(prompt_files)), desc=f"  LLM inference")
            for file in prompt_files:
                bar.update(1)
                if not os.path.isfile(os.path.join(result_dir, file)):
                    running_flag = True

                    if args.task_name == "hellaswag":
                        with open(os.path.join(prompt_cache_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        cur_input = {
                            "input": format_example(
                                one_test_example[2], label_map=label_map, args=args
                            )[0],
                            "options": one_test_example[2]["endings"],
                        }
                        data_module.k = len(cur_train_data)
                        data_module.tensorize(cur_train_data, [cur_input])
                        prediction = inference_model.do_predict(data_module)[0]
                        assert prediction in one_test_example[2]["endings"]
                        with open(f"{result_dir}/{file}", "w") as f:
                            json.dump(
                                [
                                    prediction,
                                    one_test_example[2]["endings"][
                                        one_test_example[2]["label"]
                                    ],
                                ],
                                f,
                            )
                        preds.append(prediction)
                        golds.append(
                            one_test_example[2]["endings"][one_test_example[2]["label"]]
                        )
                    elif args.task_name == "xsum":
                        with open(os.path.join(prompt_cache_dir, file)) as f:
                            one_test_example = json.load(f)
                        context = one_test_example[1]
                        input_ids = tokenizer_gpt(
                            context, return_tensors="pt"
                        ).input_ids
                        input_ids = input_ids[:, :1900]
                        input_len = input_ids.shape[1]
                        input_ids = input_ids.to(device)
                        gen_tokens = inference_model.generate(
                            input_ids,
                            do_sample=False,
                            temperature=0.7,
                            max_length=input_len + 64,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                        generated_text = tokenizer_gpt.batch_decode(
                            gen_tokens.sequences.view(-1, 1)
                        )
                        stop = ["--", "\n", ";", "#"]
                        stop_index = len(generated_text)
                        for i, c in enumerate(generated_text):
                            if i > input_len and c.strip(" ") in stop:
                                stop_index = i
                                break
                        prediction = " ".join(generated_text[input_len:stop_index])
                        golds.append(one_test_example[2]["summary"])
                        preds.append(prediction)
                        with open(f"{output_dir}/{file}", "w") as f:
                            json.dump(
                                [
                                    " ".join(generated_text[input_len:]),
                                    " ".join(generated_text[input_len:stop_index]),
                                    one_test_example[2]["summary"],
                                    input_len,
                                    stop_index,
                                ],
                                f,
                                indent=4,
                            )
                    elif args.task_name == "nq":
                        cur_key = model_keys[execution_count % len(model_keys)]
                        execution_count += 1
                        try:
                            codex_execution(
                                key=cur_key,
                                output_path=os.path.join(output_dir, file),
                                prompt_path=os.path.join(prompt_cache_dir, file),
                            )
                        except Exception as e:
                            print(e)
                            time.sleep(3)
                    elif args.task_name in ["vulfix", "treevul"]:
                        assert (
                            "gpt" in args.model_name
                        ), f"Unsupported model {args.model_name}"
                        cur_key = os.environ["GPT_KEY"]
                        try:
                            gpt_completion(
                                key=cur_key,
                                output_path=os.path.join(result_dir, file),
                                prompt_path=os.path.join(prompt_cache_dir, file),
                                sys_prompt_path=os.path.join(
                                    "sys_prompts", f"{args.task_name}.json"
                                ),
                                model_name=args.model_name,
                                logprobs=False,
                            )
                        except Exception as e:
                            print(e)
                            time.sleep(1)
                    else:
                        with open(os.path.join(prompt_cache_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        for idx in range(len(cur_train_data)):
                            cur_train_data[idx]["options"] = all_labels
                        for idx in range(len(cur_train_data)):
                            cur_train_data[idx]["options"] = all_labels
                        cur_input = format_example(
                            one_test_example[2], label_map=label_map, args=args
                        )[0]
                        data_module.k = len(cur_train_data)
                        data_module.tensorize(
                            cur_train_data, [cur_input], options=all_labels
                        )
                        prediction = inference_model.do_predict(data_module)[0]
                        with open(os.path.join(result_dir, file), "w") as f:
                            json.dump([prediction, one_test_example[2]["label"]], f)
                        preds.append(label_to_digit[prediction])
                        golds.append(one_test_example[2]["label"])
        if args.task_name == "xsum":
            assert len(golds) == len(
                preds
            ), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
            preds, golds = postprocess_text(preds, golds)
            metric = load_metric("rouge")
            result = metric.compute(
                predictions=preds, references=golds, use_stemmer=True
            )
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}
            with open(os.path.join(args.output_dir, "result_summary.json"), "w") as f:
                json.dump(result, f)
            print(result)
        elif args.task_name == "nq":
            correct = 0
            total = 0
            for file in prompt_files:
                with open(os.path.join(prompt_cache_dir, file)) as f:
                    one_test_example = json.load(f)
                answers = expand_to_aliases(
                    one_test_example[2]["long"] + one_test_example[2]["short_targets"],
                    make_sub_answers=True,
                )
                with open(os.path.join(output_dir, file)) as f:
                    pred_dict = json.load(f)
                prediction = pred_dict["choices"][0]["text"].replace("\n", " ")
                prediction = " ".join(prediction.split(" ")[1:])
                predictions = expand_to_aliases([prediction])
                if len(list(answers & predictions)) > 0:
                    correct += 1
                total += 1
            with open(os.path.join(args.output_dir, "result_summary.txt"), "w") as f:
                f.write(f"{total} examples, accuracy is: {correct / total}\n")
            print(f"{total} examples, accuracy is: {correct / total}\n")
        elif args.task_name == "vulfix":
            logger = Logger(os.path.join(output_dir, "vulfix_log.log"), file_mode="w")
            logger.log("Vulfix evaluation")
            correct = 0
            total = 0
            unknown = 0
            labels = []
            preds = []
            for file in prompt_files:
                with open(os.path.join(prompt_cache_dir, file)) as f:
                    one_test_example = json.load(f)
                label = one_test_example[2]["label"]
                labels.append(label)
                with open(os.path.join(result_dir, file)) as f:
                    pred_dict = json.load(f)
                prediction = pred_dict["choices"][0]["message"]["content"].lower()
                pred = match_answer(args.task_name, prediction)
                if pred == "yes":
                    pred = 1
                elif pred == "no":
                    pred = 0
                else:
                    logger.log(f"Undefined result in :{file}\n{prediction}")
                    pred = 1 - label
                    # pred = label
                    unknown += 1
                preds.append(pred)
                if pred == label:
                    correct += 1
                total += 1
            logger.log(
                f"{total} examples - correct predictions: {correct} - undefined predictions: {unknown}\n"
            )
            logger.log(
                f"{total-unknown} examples, accuracy is: {correct / (total-unknown)}\n"
            )
            from sklearn.metrics import classification_report

            logger.log(classification_report(labels, preds))
            # save labels and preds
            with open(os.path.join(output_dir, "out.json"), "w") as f:
                json.dump({"labels": labels, "preds": preds}, f)
        elif args.task_name == "treevul":
            print("TreeVul evaluation")
            labels = []
            preds = []
            for file in prompt_files:
                with open(os.path.join(prompt_cache_dir, file)) as f:
                    one_test_example = json.load(f)
                label = one_test_example[2][0]["cwe_list"]
                labels.append(label)
                with open(os.path.join(result_dir, file)) as f:
                    pred_dict = json.load(f)
                prediction = pred_dict["choices"][0]["message"]["content"]
                pred = match_answer(args.task_name, prediction)
                preds.append(pred)
            from sklearn.metrics import classification_report

            if args.selective_annotation_method == "random":
                report_file = f"classification_report_{args.model_name}_{args.seed}.txt"
            else:
                report_file = f"classification_report_{args.model_name}.txt"
            with open(
                os.path.join(output_dir, report_file),
                "w",
            ) as f:
                f.write(classification_report(labels, preds, zero_division=0))

        else:
            assert len(golds) == len(
                preds
            ), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
            total = len(golds)
            correct = 0
            for p, g in zip(golds, preds):
                if p == g:
                    correct += 1
            with open(os.path.join(args.output_dir, "result_summary.txt"), "w") as f:
                f.write(f"{len(golds)} examples, accuracy is: {correct / total}\n")
            print(f"The accuracy is {correct / total}\n")
