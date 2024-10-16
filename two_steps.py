import os
import random
import json
import torch
import time
import numpy as np
import tiktoken
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from utils import codex_execution, gpt_completion, get_ColBERT_indexer_searcher


def prompt_retrieval(
    train_embs,
    test_embs,
    train_examples,
    eval_examples,
    return_string,
    format_example,
    maximum_input_len,
    args,
    label_map,
    prompt_cache_dir,
    single_context_example_len=None,
):
    if args.prompt_retrieval_method == "similar":
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    elif args.prompt_retrieval_method == "colbert":
        def get_messages(examples):
            return [example[0]["msg"] for example in examples]
        train_msgs = get_messages(train_examples)
        test_msgs = get_messages(eval_examples)
        searcher = get_ColBERT_indexer_searcher(train_msgs, args)
    train_embs = torch.tensor(train_embs)
    test_embs = torch.tensor(test_embs)
    eval_example_num = len(eval_examples)
    bar = tqdm(range(eval_example_num), desc="Retrieve examples from annotated pool")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    for test_id, one_test_instance in enumerate(eval_examples):
        file_name = (
            f"{one_test_instance['id']}.json"
            if "id" in one_test_instance
            else (
                f"{one_test_instance['commit_id']}.json"
                if "commit_id" in one_test_instance
                else one_test_instance[0]["cve_list"]
                + "-".join(one_test_instance[0]["repo"].split("/"))
                + one_test_instance[0]["commit_id"]
                + ".json"
            )
        )
        if os.path.isfile(os.path.join(prompt_cache_dir, file_name)):
            bar.update(1)
            continue
        one_test_instance_input_text, one_test_instance_output_text = format_example(
            example=one_test_instance, args=args, label_map=label_map
        )
        if args.task_name in ["vulfix", "treevul"]:
            one_test_instance_input_text = tiktoken_truncate(
                one_test_instance_input_text, max_len=single_context_example_len * 2
            )
            cur_prompt_string_len = num_tokens_from_string(
                one_test_instance_input_text, one_test_instance_output_text
            )[0]
        else:
            cur_prompt_string_len = get_instance_length(
                one_test_instance_input_text, one_test_instance_output_text, tokenizer
            )[0]
        selected_shot = 0
        if one_test_instance[0]["msg"] == "":
            sorted_indices = []
            indices_scores = []
            
        else:
            if args.prompt_retrieval_method == "similar":
                test_e_reshape = test_embs[test_id].reshape(1, -1)
                scores = cos(test_e_reshape, train_embs).numpy()
                sorted_indices = np.argsort(scores)
                indices_scores = []
                
            elif args.prompt_retrieval_method == "colbert":
                msg = test_msgs[test_id]
                results = searcher.search(msg, k=5)
                indices_scores = []
                sorted_indices = []
                for passage_id, passage_rank, passage_score in zip(*results):
                    if searcher.collection[passage_id] != msg:
                        indices_scores.append([int(passage_id), float(passage_score)])
                        selected_shot += 1
            elif args.prompt_retrieval_method == "random":
                sorted_indices = np.random.permutation(range(eval_example_num))
                indices_scores = []
                
            else:
                raise ValueError(
                    f"The prompt retrieval method {args.prompt_retrieval_method} is not supported"
                )

        for idx in sorted_indices[::-1]:
            if (
                args.prompt_retrieval_method == "similar"
                and scores[idx] == 1
            ):
                continue
            if args.few_shot == selected_shot or scores[idx] < args.threshold:
                break
            cur_example_input_text, cur_example_output_text = format_example(
                example=train_examples[idx],
                args=args,
                label_map=label_map,
            )
            if args.task_name in ["vulfix", "treevul"]:
                cur_example_input_text = tiktoken_truncate(
                    cur_example_input_text, max_len=single_context_example_len
                )
                input_len, output_len = num_tokens_from_string(
                    cur_example_input_text, cur_example_output_text
                )
                cur_len = input_len + output_len
            else:
                cur_len = sum(
                    get_instance_length(
                        cur_example_input_text,
                        cur_example_output_text,
                        tokenizer=tokenizer,
                    )
                )
            cur_prompt_string_len += cur_len
            if cur_prompt_string_len > maximum_input_len:
                break
            indices_scores.append([idx.item(), scores[idx].item()])
            selected_shot += 1

        if return_string:
            cur_train_data = ""
        else:
            cur_train_data = []
        for idx, score in indices_scores:
            cur_input_text, cur_output_text = format_example(
                example=train_examples[idx],
                args=args,
                label_map=label_map,
            )
            if args.task_name in ["vulfix", "treevul"]:
                cur_input_text = tiktoken_truncate(
                    cur_input_text, max_len=single_context_example_len
                )
            if return_string:
                cur_train_data += f"{cur_input_text}\n{cur_output_text}\n\n"
            else:
                if args.task_name == "hellaswag":
                    cur_train_data.append(
                        {
                            "input": cur_input_text,
                            "output": cur_output_text,
                            "options": train_examples[
                                idx
                            ]["endings"],
                        }
                    )
                else:
                    cur_train_data.append(
                        {"input": cur_input_text, "output": cur_output_text}
                    )
        if return_string:
            cur_train_data += one_test_instance_input_text
        assert (
            num_tokens_from_string(cur_train_data, " ")[0]
            <= maximum_input_len + single_context_example_len
        ), f"The prompt of {one_test_instance['commit_id']} is too long: {num_tokens_from_string(cur_train_data, ' ')[0]}"
        with open(os.path.join(prompt_cache_dir, file_name), "w") as f:
            json.dump(
                [
                    [
                        test_id,
                        indices_scores,
                        [[scores[i].item(), train_examples[i][0]["path_list"]] for i in range(len(scores))],
                        (
                            one_test_instance["label"]
                            if "label" in one_test_instance
                            else one_test_instance[0]["cwe_list"]
                        ),
                    ],
                    cur_train_data,
                    one_test_instance,
                ],
                f,
                indent=4,
            )
        bar.update(1)


def fast_votek(embeddings, examples, select_num, k, args, vote_file=None, searcher=None):
    n = len(embeddings)
    if vote_file is not None and os.path.isfile(vote_file):
        with open(vote_file) as f:
            vote_stat = json.load(f)
    else:
        bar = tqdm(range(n), desc=f"voting")
        vote_stat = defaultdict(list)
        for i in range(n):
            if args.prompt_retrieval_method == "simalar":
                cur_emb = embeddings[i].reshape(1, -1)
                cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
                sorted_indices = np.argsort(cur_scores).tolist()[-k - 1 : -1]
            elif args.prompt_retrieval_method == "colbert":
                cur_msg = examples[i][0]["msg"]
                results = searcher.search(cur_msg, k=k)
                sorted_indices = [int(passage_id) for passage_id in results[0]]
            for idx in sorted_indices:
                if idx != i:
                    vote_stat[idx].append(i)
            bar.update(1)
        if vote_file is not None:
            with open(vote_file, "w") as f:
                json.dump(vote_stat, f)
    votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices) < select_num:
        cur_scores = defaultdict(int)
        for idx, candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices


def iterative_selection(
    train_embs,
    test_embs,
    train_examples,
    test_examples,
    return_string,
    format_example,
    maximum_input_len,
    label_map,
    single_context_example_len,
    inference_model,
    inference_data_module,
    tokenizer_gpt,
    output_dir,
    args,
):
    if args.selective_annotation_method == "least_confidence":
        selected_indices = random.sample(range(len(train_examples)), args.batch_size)
    elif args.selective_annotation_method == "votek":
        vote_file = os.path.join(output_dir, "votek_cache.json")
        train_msgs = [example[0]["msg"] for example in train_examples]
        colbert_searcher = get_ColBERT_indexer_searcher(train_msgs, args)
        selected_indices = fast_votek(
            embeddings=train_embs,
            examples=train_examples,
            select_num=args.batch_size,
            k=150,
            vote_file=vote_file,
            args=args,
            searcher=colbert_searcher,
        )
    else:
        raise ValueError(
            f"iterative selection does not support {args.selective_annotation_method}"
        )
    if not args.task_name in ["hellaswag", "xsum", "nq", "treevul"]:
        all_labels = []
        label_to_digit = {}
        for k, v in label_map.items():
            all_labels.append(v)
            label_to_digit[v] = k
    batch_count = 0
    device = torch.device("cuda")
    while len(selected_indices) < args.annotation_size:
        batch_count += 1
        cur_annotated_examples = [train_examples[idx] for idx in selected_indices]
        prompt_dir = os.path.join(output_dir, f"prompts_{batch_count}")
        if not os.path.isdir(prompt_dir):
            os.makedirs(prompt_dir, exist_ok=True)
        prompt_retrieval(
            train_embs=train_embs[selected_indices],
            test_embs=test_embs,
            train_examples=cur_annotated_examples,
            eval_examples=test_examples,
            return_string=return_string,
            format_example=format_example,
            maximum_input_len=maximum_input_len,
            args=args,
            label_map=label_map,
            prompt_cache_dir=prompt_dir,
            single_context_example_len=single_context_example_len,
        )

        prompt_files = [f for f in os.listdir(prompt_dir) if f.endswith(".json")]
        assert len(prompt_files) == len(test_examples), (
            f"len(prompt_files)={len(prompt_files)},"
            f"len(processed_eval_examples)={len(test_examples)}"
        )
        result_dir = os.path.join(
            output_dir,
            f"results_iteration_{batch_count}_{args.model_name}",
        )
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        count = 0
        execution_count = 0
        # model_keys = args.model_key.split('##')
        running_flag = True
        while running_flag:
            running_flag = False
            count += 1
            bar = tqdm(
                range(len(prompt_files)), desc=f"  prediction iteration {batch_count}"
            )
            for file in prompt_files:
                bar.update(1)
                if not os.path.isfile(os.path.join(result_dir, file)):
                    running_flag = True

                    if args.task_name == "hellaswag":
                        with open(os.path.join(prompt_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        cur_input = {
                            "input": format_example(
                                one_test_example[2], label_map=label_map, args=args
                            )[0],
                            "options": one_test_example[2]["endings"],
                        }
                        inference_data_module.k = len(cur_train_data)
                        inference_data_module.tensorize(cur_train_data, [cur_input])
                        prediction = inference_model.do_predict(
                            inference_data_module, require_loss=True
                        )[0]
                        with open(f"{output_dir}/{file}", "w") as f:
                            json.dump(prediction, f)
                    elif args.task_name == "xsum":
                        with open(os.path.join(prompt_dir, file)) as f:
                            one_test_example = json.load(f)
                        context = one_test_example[1]
                        input_ids = tokenizer_gpt(
                            context, return_tensors="pt"
                        ).input_ids
                        input_ids = input_ids[:, :1900]
                        input_len = input_ids.shape[1]
                        input_ids = input_ids.to(device)
                        # print(input_ids.shape)
                        # print(os.path.join(prompt_dir,file))
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
                        )  #
                        stop = ["--", "\n", ";", "#"]
                        stop_index = len(generated_text)
                        for i, c in enumerate(generated_text):
                            if i > input_len and c.strip(" ") in stop:
                                stop_index = i
                                break
                        prediction = [
                            " ".join(generated_text[input_len:stop_index]),
                            sum(gen_tokens.probs[: stop_index - input_len]),
                        ]
                        with open(f"{output_dir}/{file}", "w") as f:
                            json.dump(prediction, f)
                    elif args.task_name == "nq":
                        cur_key = model_keys[execution_count % len(model_keys)]
                        execution_count += 1
                        try:
                            codex_execution(
                                key=cur_key,
                                output_path=os.path.join(output_dir, file),
                                prompt_path=os.path.join(prompt_dir, file),
                            )
                        except Exception as e:
                            print(e)
                            time.sleep(3)
                    elif args.task_name in ["vulfix", "treevul"]:
                        assert (
                            "gpt" in args.model_name
                        ), f"Unsupported model {args.model_name}"
                        cur_key = os.environ["GPT_KEY"]
                        success_flag = False
                        while not success_flag:
                            try:
                                gpt_completion(
                                    key=cur_key,
                                    output_path=os.path.join(result_dir, file),
                                    sys_prompt_path=os.path.join(
                                        "sys_prompts", f"{args.task_name}.json"
                                    ),
                                    prompt_path=os.path.join(prompt_dir, file),
                                    model_name=args.model_name,
                                )
                                success_flag = True
                            except Exception as e:
                                print(e)
                                time.sleep(1)
                    else:
                        with open(os.path.join(prompt_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        for idx in range(len(cur_train_data)):
                            cur_train_data[idx]["options"] = all_labels
                        cur_input = format_example(
                            one_test_example[2], label_map=label_map, args=args
                        )[0]
                        inference_data_module.k = len(cur_train_data)
                        inference_data_module.tensorize(
                            cur_train_data, [cur_input], options=all_labels
                        )
                        prediction = inference_model.do_predict(
                            inference_data_module, require_loss=True
                        )[0]
                        with open(f"{result_dir}/{file}", "w") as f:
                            json.dump(prediction, f)

        idx_scores = {}
        n = len(test_examples)
        for idx in range(n):
            if idx in selected_indices:
                if args.task_name in ["xsum", "nq"]:
                    idx_scores[idx] = float("inf")
                else:
                    idx_scores[idx] = float("-inf")
                continue
            file_name = (
                f"{test_examples[idx]['id']}.json"
                if "id" in test_examples[idx]
                else (
                    f"{test_examples[idx]['commit_id']}.json"
                    if "commit_id" in test_examples[idx]
                    else test_examples[idx][0]["cve_list"]
                    + "-".join(test_examples[idx][0]["repo"].split("/"))
                    + test_examples[idx][0]["commit_id"]
                    + ".json"
                )
            )
            with open(f"{result_dir}/{file_name}") as f:
                one_pred = json.load(f)
                if args.task_name in ["nq"]:
                    idx_scores[idx] = sum(
                        one_pred["choices"][0]["logprobs"]["token_logprobs"]
                    ) / len(one_pred["choices"][0]["logprobs"]["token_logprobs"])
                if args.task_name in ["vulfix", "treevul"]:
                    idx_scores[idx] = np.mean(
                        [
                            x["logprob"]
                            for x in one_pred["choices"][0]["logprobs"]["content"]
                        ]
                    )
                else:
                    idx_scores[idx] = one_pred[1]
        if args.task_name in ["xsum", "nq"]:
            sorted_scores = sorted(idx_scores.items(), key=lambda x: x[1])
        else:
            sorted_scores = sorted(idx_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_scores_len = len(sorted_scores)
        if args.selective_annotation_method == "least_confidence":
            cur_selected = []
            cur_select_num = min(
                args.batch_size, args.annotation_size - len(selected_indices)
            )
            for sorted_scores_iter in range(sorted_scores_len):
                if len(cur_selected) >= cur_select_num:
                    break
                if not sorted_scores[sorted_scores_iter][0] in selected_indices:
                    cur_selected.append(sorted_scores[sorted_scores_iter][0])
            selected_indices += cur_selected
        else:
            with open(vote_file, "r") as f:
                vote_stat = json.load(f)
            selected_times = defaultdict(int)
            select_num_1 = args.annotation_size - len(selected_indices)
            inter = int(len(train_examples) * 0.9 / select_num_1)
            for prev_idx in selected_indices:
                for idx_support in vote_stat[str(prev_idx)]:
                    selected_times[idx_support] += 1
            count_t = 0
            while (
                len(selected_indices) < args.annotation_size
                and count_t * inter < sorted_scores_len
            ):
                cur_scores = defaultdict(int)
                for idx, _ in sorted_scores[count_t * inter : (count_t + 1) * inter]:
                    if not str(idx) in vote_stat:
                        cur_scores[idx] = 0
                        continue
                    candidates = vote_stat[str(idx)]
                    if idx in selected_indices:
                        cur_scores[idx] = -100
                        continue
                    for one_support in candidates:
                        if not one_support in selected_indices:
                            cur_scores[idx] += 10 ** (-selected_times[one_support])
                cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
                selected_indices.append(cur_selected_idx)
                if cur_selected_idx in vote_stat:
                    for idx_support in vote_stat[cur_selected_idx]:
                        selected_times[idx_support] += 1
                count_t += 1
            if len(selected_indices) < args.annotation_size:
                unselected_indices = []
                for unselected_i in range(len(train_examples)):
                    if not unselected_i in selected_indices:
                        unselected_indices.append(unselected_i)
                selected_indices += random.sample(
                    unselected_indices, args.annotation_size - len(selected_indices)
                )
                print(
                    f"{args.annotation_size - len(selected_indices)} examples are randomly selected"
                )
    return selected_indices


def selective_annotation(args, **kwargs):
    if args.selective_annotation_method == "random":
        train_examples = kwargs["train_examples"]
        selected_indices = random.sample(
            range(len(train_examples)), args.annotation_size
        )
    elif args.selective_annotation_method == "all":
        selected_indices = list(range(len(kwargs["train_examples"])))
    elif args.selective_annotation_method == "diversity":
        embeddings = kwargs["embeddings"]
        selected_indices = []
        first_id = random.choice(range(len(embeddings)))
        selected_indices.append(first_id)
        selected_representations = embeddings[first_id].reshape(1, -1)
        for count in range(args.annotation_size - 1):
            scores = np.sum(
                cosine_similarity(embeddings, selected_representations), axis=1
            )
            for i in selected_indices:
                scores[i] = float("inf")
            min_idx = np.argmin(scores)
            selected_representations = torch.cat(
                (selected_representations, embeddings[min_idx].reshape(1, -1)), 0
            )
            selected_indices.append(min_idx.item())
    elif args.selective_annotation_method == "fast_votek":
        selected_indices = fast_votek(
            embeddings=kwargs["embeddings"],
            select_num=args.annotation_size,
            k=150,
            vote_file=os.path.join(args.output_dir, "nearest_neighbors.json"),
        )
    elif args.selective_annotation_method == "mfl":
        embeds = kwargs["embeddings"]
        N, D = embeds.shape
        norm_embeds = embeds / embeds.norm(dim=1, keepdim=True)
        cosine = torch.einsum("nd,md->nm", norm_embeds, norm_embeds)
        selected = torch.zeros(N, dtype=torch.bool)
        max_similarity = torch.zeros(N) - 1
        for k in tqdm(range(args.annotation_size)):
            marginal_gain = torch.relu(cosine - max_similarity).sum(dim=1) * (
                1 - selected.float()
            )
            node = torch.argmax(marginal_gain)
            selected[node] = True
            max_similarity = torch.max(max_similarity, cosine[node])
        selected_indices = torch.nonzero(selected).squeeze().tolist()
    elif args.selective_annotation_method in ["votek", "least_confidence"]:
        selected_indices = iterative_selection(
            train_embs=kwargs["embeddings"],
            test_embs=kwargs["embeddings"],
            train_examples=kwargs["train_examples"],
            test_examples=kwargs["train_examples"],
            return_string=kwargs["return_string"],
            format_example=kwargs["format_example"],
            maximum_input_len=kwargs["maximum_input_len"],
            label_map=kwargs["label_map"],
            single_context_example_len=kwargs["single_context_example_len"],
            inference_model=kwargs["inference_model"],
            inference_data_module=kwargs["inference_data_module"],
            tokenizer_gpt=kwargs["tokenizer_gpt"],
            output_dir=kwargs["output_dir"],
            args=args,
        )
    else:
        raise ValueError(
            f"The selective annotation method {args.selective_annotation_method} is not supported"
        )
    return selected_indices


def get_instance_length(input_text, output_text, tokenizer):
    return len(tokenizer(input_text)["input_ids"]), len(
        tokenizer(output_text)["input_ids"]
    )


def num_tokens_from_string(input: str, output: str, encoding_name: str = "gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(encoding_name)
    input_len = len(encoding.encode(input))
    output_len = len(encoding.encode(output))
    return input_len, output_len


def tiktoken_truncate(string, encoding_name="gpt-4o-mini", max_len=1024):
    encoding = tiktoken.encoding_for_model(encoding_name)
    encoded_text = encoding.encode(string)
    if len(encoded_text) <= max_len:
        return string
    return encoding.decode(encoded_text[:max_len])
