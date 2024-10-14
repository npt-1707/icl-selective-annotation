# embedding model class for abstracting bert, codebert, and other models
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from openai import OpenAI
import tiktoken
import time
from tqdm import tqdm

class CommitEmbeddingModel:
    def __init__(self, model_name, use_diff=True):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.use_diff = use_diff

    def calculate_sentence_embedding(self, sentences, **kargs):
        pass


class BERTModel(CommitEmbeddingModel):
    def __init__(self, use_diff=True):
        name = 'bert-base-uncased'
        super().__init__(name, use_diff)
        self.max_length = 512
    
    def load(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def tokenize_sentences(self, sentences):
        ids = []
        masks = []
        for sentence in sentences:
            if self.use_diff:
                message, diff = sentence
                message_tokens = self.tokenizer.tokenize(message)
                diff_tokens = self.tokenizer.tokenize(diff)
                len_message = len(message_tokens)
                len_diff = len(diff_tokens)
                output = []
                cls_token = self.tokenizer.cls_token
                sep_token = self.tokenizer.sep_token
                pad_token = self.tokenizer.pad_token
                if len_message + len_diff + 3 < self.max_length:
                    output += [cls_token] + message_tokens + [sep_token] + diff_tokens + [sep_token] + [pad_token] * (self.max_length - len_message - len_diff - 3)
                    output_mask = [1] * (len_message + len_diff + 3) + [0] * (self.max_length - len_message - len_diff - 3)
                else:
                    output_mask = [1] * (len_message + len_diff + 3)
                    if len_message + len_diff + 3 == self.max_length:
                        output += [cls_token] + message_tokens + [sep_token] + diff_tokens + [sep_token]
                    elif len_message > self.max_length -2:
                        output = [cls_token] + message_tokens[:self.max_length-2] + [sep_token]
                    else:
                        output = [cls_token] + message_tokens + [sep_token] + diff_tokens[:self.max_length-len_message-3] + [sep_token]
            else:
                message = sentence[0]
                message_tokens = self.tokenizer.tokenize(message)
                if len(message_tokens) <= self.max_length - 2:
                    output = [self.tokenizer.cls_token] + message_tokens + [self.tokenizer.sep_token] + [self.tokenizer.pad_token] * (self.max_length - len(message_tokens) - 2)
                    output_mask = [1] * (len(message_tokens) + 2) + [0] * (self.max_length - len(message_tokens) - 2)
                else:
                    output = [self.tokenizer.cls_token] + message_tokens[:self.max_length - 2] + [self.tokenizer.sep_token]
                    output_mask = [1] * (self.max_length)
            assert len(output) == self.max_length, f"output length {len(output)} not equal to {self.max_length}"
            output_ids = self.tokenizer.convert_tokens_to_ids(output)
            ids.append(output_ids)
            masks.append(output_mask)
        return ids, masks

    def calculate_sentence_embedding(self, sentences, args):
        if not self.model and not self.tokenizer:
            self.load()
        self.model.to(args.device)
        num_sentences = len(sentences)
        embeddings = []
        ids, masks = self.tokenize_sentences(sentences)
        for i in range(0, num_sentences, args.emb_batch_size):
            end_idx = min(i+args.emb_batch_size, num_sentences)
            ids_batch = ids[i:end_idx]
            masks_batch = masks[i:end_idx]
            ids_tensor = torch.tensor(ids_batch).to(args.device)
            masks_tensor = torch.tensor(masks_batch).to(args.device)
            with torch.no_grad():
                output = self.model(ids_tensor, attention_mask=masks_tensor)
                embeddings+=output[1].detach().cpu().tolist()
        embeddings = np.array(embeddings)
        mean_embeddings = np.mean(embeddings, axis=0, keepdims=True)
        embeddings = embeddings - mean_embeddings
        return embeddings

class CodeBERTModel(BERTModel):
    def __init__(self, use_diff=True):
        super().__init__(use_diff)
        self.model_name = 'microsoft/codebert-base'

class OpenAIEmbeddingModel(CommitEmbeddingModel):
    def __init__(self, use_diff=True):
        model_name = 'text-embedding-3-small'
        super().__init__(model_name, use_diff)
        self.max_length = 8191
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def tokenize_sentences(self, sentences):
        ids = []
        for sentence in sentences:
            if self.use_diff:
                message, diff = sentence
                text = f"Commit message: {message} Diff: {diff}"
            else:
                message = sentence[0]
                text = f"Commit message: {message}"
            id = self.encoding.encode(text)[:self.max_length]
            ids.append(id)
        assert len(ids) == len(sentences), f"Number of encoded_ids {len(ids)} not equal to number of sentences {len(sentences)}"
        return ids

    def calculate_sentence_embedding(self, sentences, args):
        encoded_tokens = self.tokenize_sentences(sentences)
        client = OpenAI(api_key=args.key)
        num_sentences = len(sentences)
        embeddings = []
        max_num_trials = 3
        for i in tqdm(range(0, num_sentences, args.emb_batch_size), desc="Calculating embeddings"):
            end_idx = min(i + args.emb_batch_size, num_sentences)
            tokens = encoded_tokens[i:end_idx]
            is_success = False
            for _ in range(max_num_trials):
                try:
                    responses = client.embeddings.create(
                        input=tokens,
                        model="text-embedding-3-small",
                        encoding_format="float",
                        # dimensions=512,
                    )
                    is_success = True
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(1)
            assert is_success, f"Failed to get embeddings for batch {i} to {end_idx}"
            for emb_obj in responses.data:
                embeddings.append(emb_obj.embedding)

        embeddings = np.array(embeddings)
        mean_embeddings = np.mean(embeddings, axis=0, keepdims=True)
        embeddings = embeddings - mean_embeddings
        return embeddings

def get_embedding_model(model_name, use_diff=True):
    if model_name == 'bert':
        print("Using BERT model")
        return BERTModel(use_diff)
    elif model_name == 'codebert':
        print("Using CodeBERT model")
        return CodeBERTModel(use_diff)
    elif model_name == 'openai':
        print("Using OpenAI Embedding model")
        return OpenAIEmbeddingModel(use_diff)
    assert False, f"Embedding model name not supported: {model_name}"
