import torch
import openai
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from transformers import GPTJForCausalLM
from collections import OrderedDict
from openai import OpenAI
import os
import re

def bert_tokenize_commit(tokenizer, message, diff, output_length):
    try:
        mes_tokens = tokenizer.tokenize(message)
        diff_tokens = tokenizer.tokenize(diff)
    except Exception as e:
        print("Error in tokenization: ", e)
        print("Message: ", message)
        print("Diff: ", diff)
        print("Output length: ", output_length)
        assert False, "Error in tokenization"
    len_mes = len(mes_tokens)
    len_diff = len(diff_tokens)

    output = []
    if len_mes + len_diff + 3 < output_length:
        output += (
            [tokenizer.cls_token]
            + mes_tokens
            + [tokenizer.sep_token]
            + diff_tokens
            + [tokenizer.sep_token]
            + [tokenizer.pad_token] * (output_length - len_mes - len_diff - 3)
        )
        output_mask = [
            1 if i < len_mes + len_diff + 3 else 0 for i in range(output_length)
        ]
    else:
        output_mask = [1 for i in range(output_length)]
        if len_mes + len_diff + 3 == output_length:
            output += (
                [tokenizer.cls_token]
                + mes_tokens
                + [tokenizer.sep_token]
                + diff_tokens
                + [tokenizer.sep_token]
            )
        else:
            if len_mes > output_length - 2:
                output += (
                    [tokenizer.cls_token]
                    + mes_tokens[: output_length - 2]
                    + [tokenizer.sep_token]
                )
            else:
                output += (
                    [tokenizer.cls_token]
                    + mes_tokens
                    + [tokenizer.sep_token]
                    + diff_tokens[: output_length - 3 - len_mes]
                    + [tokenizer.sep_token]
                )
    assert (
        len(output) == output_length
    ), "Length of output is not equal to output_length"
    output_id = tokenizer.convert_tokens_to_ids(output)
    return output_id, output_mask


def calculate_sentence_transformer_embedding(text_to_encode, args):
    num = len(text_to_encode)
    embeddings = []
    if args.task_name in ["vulfix", "treevul"]:
        MAX_LENGTH = 512
        ids = []
        masks = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        for i in tqdm(range(len(text_to_encode))):
            mes, diff = text_to_encode[i][0], text_to_encode[i][1]
            id, mask = bert_tokenize_commit(tokenizer, mes, diff, MAX_LENGTH)
            ids.append(id)
            masks.append(mask)
        emb_model = AutoModel.from_pretrained("microsoft/codebert-base")
        emb_model.to(device)
        for i in tqdm(range(0, len(ids), args.emb_batch_size)):
            start = i
            end = min(i + args.emb_batch_size, len(ids))
            ids_batch = torch.tensor(ids[start:end]).to(device)
            masks_batch = torch.tensor(masks[start:end]).to(device)
            assert (
                ids_batch.shape[1] == MAX_LENGTH
            ), "ids_batch shape is not correct: {}".format(ids_batch.shape)
            assert (
                masks_batch.shape[1] == MAX_LENGTH
            ), "masks_batch shape is not correct: {}".format(masks_batch.shape)
            with torch.no_grad():
                output = emb_model(ids_batch, masks_batch)
            embeddings += output[1].detach().cpu().tolist()
    else:
        emb_model = SentenceTransformer(args.embedding_model)
        bar = tqdm(range(0, num, 20), desc="calculate embeddings")
        for i in range(0, num, 20):
            embeddings += emb_model.encode(text_to_encode[i : i + 20]).tolist()
            bar.update(1)

    embeddings = torch.tensor(embeddings)
    mean_embeddings = torch.mean(embeddings, 0, True)
    embeddings = embeddings - mean_embeddings
    return embeddings


def gpt_completion(
    prompt_path, sys_prompt_path, key, output_path, model_name="gpt-4o-mini", logprobs=True
):
    with open(prompt_path, "r") as f:
        prompt = json.load(f)[1]
    
    with open(sys_prompt_path, "r") as f:
        sys_prompt = json.load(f)

    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": " ".join(sys_prompt),
            },
            {"role": "user", "content": prompt},
        ],
        logprobs=logprobs,
    )
    with open(output_path, "w") as f:
        f.write(response.to_json())
    # return response.choices[0].message.content


def codex_execution(key, output_path, prompt_path):
    openai.api_key = key
    with open(prompt_path) as f:
        prompt = json.load(f)[1]
    completion = openai.Completion.create(
        engine="code-davinci-002",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
        logprobs=5,
        stop=["--", "\n\n", ";", "#"],
    )
    with open(output_path, "w") as f:
        json.dump(completion, f)


def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]


PUNCTUATION_SET_TO_EXCLUDE = set("".join(["‘", "’", "´", "`", ".", ",", "-", '"']))


def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
        given_answers = (
            given_answers
            + get_sub_answers(given_answers, begin=1)
            + get_sub_answers(given_answers, end=-1)
        )
    answers = []
    for answer in given_answers:
        alias = answer.replace("_", " ").lower()
        alias = "".join(
            c if c not in PUNCTUATION_SET_TO_EXCLUDE else " " for c in alias
        )
        answers.append(" ".join(alias.split()).strip())
    return set(answers)


table_prompt = """
CREATE TABLE hotel(
  name text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  type text CHECK (type IN (hotel, guest house)),
  parking text CHECK (parking IN (dontcare, yes, no)),
  book_stay int,
  book_day text,
  book_people int,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  stars int CHECK (stars IN (dontcare, 0, 1, 2, 3, 4, 5)),
  internet text CHECK (internet IN (dontcare, yes, no))
)
/*
4 example rows:
SELECT * FROM hotel LIMIT 4;
name  pricerange  type  parking book_stay book_day  book_people area  stars internet
a and b guest house moderate  guest house  dontcare  3 friday  5 east  4 yes
ashley hotel  expensive hotel yes 2 thursday  5 north 5 yes
el shaddia guest house  cheap guest house  yes 5 friday  2 centre  dontcare  no
express by holiday inn cambridge  dontcare  guest house yes 3 monday  2 east  dontcare  no
*/

CREATE TABLE train(
  destination text,
  departure text,
  day text,
  book_people int,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM train LIMIT 3;
destination departure day book_people leaveat arriveby
london kings cross  cambridge monday  6 dontcare 05:51
cambridge stansted airport  dontcare  1 20:24 20:52
peterborough  cambridge saturday  2  12:06  12:56
*/

CREATE TABLE attraction(
  name text,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  type text CHECK (type IN (architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre))
)
/*
4 example rows:
SELECT * FROM attraction LIMIT 4;
name area type
abbey pool and astroturf pitch  centre  swimming pool
adc theatre centre  theatre
all saints church dontcare  architecture
castle galleries  centre  museum
*/

CREATE TABLE restaurant(
  name text,
  food text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  area text CHECK (area IN (centre, east, north, south, west)),
  book_time text,
  book_day text,
  book_people int
)
/*
5 example rows:
SELECT * FROM restaurant LIMIT 5;
name  food  pricerange  area  book_time book_day  book_people
pizza hut city centre italian dontcare centre  13:30 wednesday 7
the missing sock  international moderate  east  dontcare dontcare  2
golden wok chinese moderate north 17:11 friday 4
cambridge chop house  dontcare  expensive  center 08:43 monday  5
darrys cookhouse and wine shop  modern european expensive center  11:20 saturday  8
*/

CREATE TABLE taxi(
  destination text,
  departure text,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM taxi LIMIT 3;
destination departure leaveat arriveby
copper kettle royal spice 14:45 15:30
magdalene college  university arms hotel dontcare  15:45
lovell lodge  da vinci pizzeria 11:45 dontcare
*/

-- Using valid SQLite, answer the following multi-turn conversational questions for the tables provided above.

"""


def slot_values_to_seq_sql(original_slot_values, single_answer=False):
    sql_str = ""
    tables = OrderedDict()
    col_value = dict()

    # add '_' in SQL columns
    slot_values = {}
    for slot, value in original_slot_values.items():
        if " " in slot:
            slot = slot.replace(" ", "_")
        slot_values[slot] = value

    for slot, value in slot_values.items():
        assert len(slot.split("-")) == 2

        if "|" in value:
            value = value.split("|")[0]

        table, col = slot.split("-")  # slot -> table-col

        if table not in tables.keys():
            tables[table] = []
        tables[table].append(col)

        # sometimes the answer is ambiguous
        if single_answer:
            value = value.split("|")[0]
        col_value[slot] = value

    # When there is only one table
    if len(tables.keys()) == 1:
        where_clause = []
        table = list(tables.keys())[0]
        for col in tables[table]:
            where_clause.append(
                "{} = {}".format(col, col_value["{}-{}".format(table, col)])
            )
        sql_str = "SELECT * FROM {} WHERE {}".format(table, " AND ".join(where_clause))
    # When there are more than one table
    else:
        # We observed that Codex has variety in the table short names, here we just use a simple version.
        from_clause = []
        where_clause = []
        for i, table in enumerate(tables.keys()):
            t_i = "t{}".format(i + 1)
            from_clause.append("{} AS {}".format(table, t_i))
            for col in tables[table]:
                where_clause.append(
                    "{}.{} = {}".format(t_i, col, col_value["{}-{}".format(table, col)])
                )
        sql_str = "SELECT * FROM {} WHERE {}".format(
            ", ".join(from_clause), " AND ".join(where_clause)
        )

    return sql_str


class PreviousStateRecorder:

    def __init__(self):
        self.states = {}

    def add_state(self, data_item, slot_values):
        dialog_ID = data_item["dialogue_ID"]
        turn_id = data_item["turn_id"]
        if dialog_ID not in self.states:
            self.states[dialog_ID] = {}
        self.states[dialog_ID][turn_id] = slot_values

    def state_retrieval(self, data_item):
        dialog_ID = data_item["dialogue_ID"]
        turn_id = data_item["turn_id"]
        if turn_id == 0:
            return {}
        else:
            return self.states[dialog_ID][turn_id - 1]


def codex_completion(prompt_text, key, output_path, model_name="code-davinci-002"):
    openai.api_key = key
    result = openai.Completion.create(
        engine=model_name,
        prompt=prompt_text,
        max_tokens=200,
        temperature=0,
        logprobs=5,
        stop=["--", "\n", ";", "#"],
    )
    with open(output_path, "w") as f:
        json.dump(result, f)
    return result["choices"][0]["text"]


def sql_pred_parse(pred):
    import sqlparse

    # parse sql results and fix general errors

    pred = " * FROM" + pred

    # fix for no states
    if pred == " * FROM  WHERE ":
        return {}

    # Here we need to write a parser to convert back to dialogue state
    pred_slot_values = []
    # pred = pred.lower()
    parsed = sqlparse.parse(pred)
    if not parsed:
        return {}
    stmt = parsed[0]
    sql_toks = pred.split()
    operators = [" = ", " LIKE ", " < ", " > ", " >= ", " <= "]

    if "AS" in pred:
        as_indices = [i for i, x in enumerate(sql_toks) if x == "AS"]

        table_name_map_dict = {}
        for indice in as_indices:
            table_name_map_dict[sql_toks[indice + 1].replace(",", "")] = sql_toks[
                indice - 1
            ]

        slot_values_str = (
            str(stmt.tokens[-1])
            .replace("_", " ")
            .replace("""'""", "")
            .replace("WHERE ", "")
        )
        for operator in operators:
            slot_values_str = slot_values_str.replace(operator, "-")
        slot_values = slot_values_str.split(" AND ")

        for sv in slot_values:
            for t_ in table_name_map_dict.keys():
                sv = sv.replace(t_ + ".", table_name_map_dict[t_] + "-")
            pred_slot_values.append(sv)
    else:

        table_name = sql_toks[sql_toks.index("FROM") + 1]

        slot_values_str = (
            str(stmt.tokens[-1])
            .replace("_", " ")
            .replace("""'""", "")
            .replace("WHERE ", "")
        )
        for operator in operators:
            slot_values_str = slot_values_str.replace(operator, "-")
        slot_values = slot_values_str.split(" AND ")

        pred_slot_values.extend(
            [table_name + "-" + sv for sv in slot_values if slot_values != [""]]
        )

    pred_slot_values = {
        "-".join(sv_pair.split("-")[:-1]): sv_pair.split("-")[-1]
        for sv_pair in pred_slot_values
    }

    # remove _ in SQL columns
    pred_slot_values = {
        slot.replace("_", " "): value for slot, value in pred_slot_values.items()
    }

    # fix typos
    # pred_slot_values, _ = typo_fix(pred_slot_values)

    return pred_slot_values


def check_prefix_suffix(value, candidates):
    # add/delete "the" in the front, or the suffix in the end.
    if value in candidates:
        return value
    prefixes = ["the "]
    suffixes = [
        " hotel",
        " restaurant",
        " cinema",
        " guest house",
        " theatre",
        " airport",
        " street",
        " gallery",
        " museum",
    ]
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    for suffix in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    for prefix in [""] + prefixes:
        for suffix in [""] + suffixes:
            possible_value = prefix + value + suffix
            if possible_value in candidates:
                return possible_value
    return ""


def typo_fix(slot_values, ontology, version="2.1"):

    # fix the named entities in these slots
    named_entity_slots = [
        "hotel-name",
        "train-destination",
        "train-departure",
        "attraction-type",
        "attraction-name",
        "restaurant-name",
        "taxi-departure",
        "taxi-destination",
        "restaurant-food",
    ]
    fixed = {}
    for slot, value in slot_values.items():
        # fix 's
        value = value.replace(" s ", "s ")
        if value.endswith(" s"):
            value = value[:-2] + "s"

        # fix typo words
        general_typos = {
            "fen ditton": "fenditton",
            "guesthouse": "guest house",
            "steveage": "stevenage",
            "stantsted": "stansted",
            "storthford": "stortford",
            "shortford": "stortford",
            "weish": "welsh",
            "bringham": "birmingham",
            "liverpoool": "liverpool",
            "petersborough": "peterborough",
            "el shaddai": "el shaddia",
            "wendesday": "wednesday",
            "brazliian": "brazilian",
            "graffton": "grafton",
        }
        for k, v in general_typos.items():
            value = value.replace(k, v)

        # fix whole value
        value_replacement = {
            "center": "centre",
            "caffe uno": "cafe uno",
            "caffee uno": "cafe uno",
            "christs college": "christ college",
            "churchill college": "churchills college",
            "sat": "saturday",
            "saint johns chop shop house": "saint johns chop house",
            "good luck chinese food takeaway": "good luck",
            "asian": "asian oriental",
            "gallery at 12": "gallery at 12 a high street",
        }

        if version == "2.1":
            value_replacement["portuguese"] = "portugese"
            value_replacement["museum of archaeology and anthropology"] = (
                "museum of archaelogy and anthropology"
            )

        if version == "2.4":
            value_replacement["portugese"] = "portuguese"
            value_replacement["museum of archaelogy and anthropology"] = (
                "museum of archaeology and anthropology"
            )

        for k, v in value_replacement.items():
            if value == k:
                value = v

        # time format fix  9:00 -> 09:00
        if ":" in value and len(value) < 5:
            value = "0" + value

        if slot in named_entity_slots:
            value = check_prefix_suffix(value, ontology[slot])

        if value:
            fixed[slot] = value
    return fixed


def compute_acc(gold, pred, n_slot=30):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = n_slot
    ACC = n_slot - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = (
            2 * precision * recall / float(precision + recall)
            if (precision + recall) != 0
            else 0
        )
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


def evaluate(preds: dict, golds: dict):

    gold_slots = list(golds.keys())
    for k in gold_slots:
        if "|" in golds[k]:
            gold_values = golds[k].split("|")
            if k in preds and preds[k] in gold_values:
                golds[k] = preds[k]

    jga, acc, f1 = 0, 0, 0

    if preds == golds:
        jga = 1
    acc = compute_acc(golds, preds)
    f1 = compute_prf(golds, preds)[0]

    return jga, acc, f1


import logging


class Logger:
    def __init__(
        self, log_file, name=None, log_level=logging.INFO, console=True, file_mode="w", time=True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        if time:
            formatter = logging.Formatter(
                "[%(asctime)s] - [%(levelname)s]: %(message)s"
            )
        else:
            formatter = logging.Formatter("[%(levelname)s]: %(message)s")

        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create a console handler
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def debug(self, message):
        self.logger.debug(message)

def match_answer(task_name, answer):
    if task_name == "vulfix":
        # check if "answer: yes" or "answer: no" is in the answer, between "answer" and "yes" or "no" there can be some other character
        # "answer" can be in lower or capital case
        # return "yes" or "no" if found, otherwise return None
        pattern = re.compile(r"answer.*\s+(yes|no)", re.IGNORECASE)
        match = pattern.search(answer)
        if match:
            return match.group(1)
        return None
    elif task_name == "treevul":
        # use regex to match CWE id in the answer
        pattern = re.compile(r"answer: CWE-(?P<id>\d+)", re.IGNORECASE)
        m = pattern.search(answer)
        if m:
            return "CWE-" + m.groupdict()["id"]
        return "N/A"
    raise ValueError(f"Unsupported task name: {task_name}")