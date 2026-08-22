import json
from pathlib import Path
from typing import Dict, List, Set

from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


class Tokenizer:
    """Tokenizer with chat template supported using jinja2 engine"""

    def __init__(self, tokenizer_path: str):
        super().__init__()
        tokenizer_path_obj = Path(tokenizer_path)
        tokenizer_config_path = tokenizer_path_obj.parent / "tokenizer_config.json"

        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            self.tokenizer_config = json.load(f)
            
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)
        self.chat_template = Environment().from_string(
            self.tokenizer_config["chat_template"]
        )
        
        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

        self.stop_token_ids: Set[int] = {self.eos_token_id}
        
        candidate_stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end|>"]
        for token_str in candidate_stop_tokens:
            tid = self.tokenizer.token_to_id(token_str)
            if tid is not None:
                self.stop_token_ids.add(tid)
        

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
