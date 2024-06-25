from langchain_text_splitters import CharacterTextSplitter
from typing import List, Dict
from tqdm import tqdm


class Tokenizer:
    def __init__(self, raw_texts: List[Dict[str, str]], chunk_size: int = None, chunk_overlap: int = None):
        if chunk_size is None:
            self.chunk_size = 4000
        else:
            self.chunk_size = chunk_size

        if chunk_overlap is None:
            self.chunk_overlap = 0
        else:
            self.chunk_overlap = chunk_overlap

        self.tokens = []
        self.raw_texts = raw_texts

    def generate_tokens(self):
        # Optional: Enforce a specific token size for texts
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=0)

        # Generate tokens per doc
        doc_paths = set([item['doc_path'] for item in self.raw_texts])
        print("Generating tokens...")
        for doc in doc_paths:
            joined_texts = " ".join([text['raw_text'] for text in self.raw_texts if text['doc_path'] == doc])
            tokens = text_splitter.split_text(joined_texts)

            # Make sure tokens do not exceed context length
            if len(tokens) > 3:
                for start_index in range(0, len(tokens), 3):
                    self.tokens.append({'tokens': tokens[start_index:start_index + 3],
                                        'raw_text': joined_texts,
                                        'doc_path': doc
                                        })
            else:
                self.tokens.append({'tokens': tokens,
                                    'raw_text': joined_texts,
                                    'doc_path': doc
                                    })
        for token in self.tokens:
            print(f'Doc: {token["doc_path"]}\nTokens: {len(token["tokens"])}')
