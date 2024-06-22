import base64
import os
from typing import List, Dict
from configparser import ConfigParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.extend(['./src'])

from extract_doc_elements import ElementExtractor
from tokenizer import Tokenizer

input_path = "../data/input"
image_path = "../data/image"
key_path = "../config/keys.ini"
config_path = "../config/model_config.ini"

# Extract elements
element_extractor = ElementExtractor(input_dir=input_path, image_dir=image_path)
element_extractor.get_raw_elements()
element_extractor.categorize_elements()

# Tokenize text chunks
tokenizer = Tokenizer(raw_texts=element_extractor.raw_texts)
tokenizer.generate_tokens()

# Get tables
raw_tables = element_extractor.raw_tables


def get_model(mode: str) -> ChatOpenAI:
    if mode == 'image':
        config_section = 'image-summary'
    else:
        config_section = 'text-summary'

    # Read model config
    config = ConfigParser()

    config.read(key_path)
    api_key = config['multimodal-rag']['API_Key']

    config.read(config_path)
    model = config[config_section]['model']
    temp = int(config[config_section]['temperature'])
    max_tokens = int(config[config_section]['max_tokens'])

    return ChatOpenAI(model=model, max_tokens=max_tokens, api_key=api_key, temperature=temp)


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    llm = get_model(mode='image')

    msg = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                     },
                ]
            )
        ]
    )

    return msg.content


class Summarize:
    def __init__(self, texts: List[Dict[List[str], str, str]],
                 tables: List[Dict[str, str, str]],
                 image_dir: str,
                 summarize_text: bool = True):
        self.texts = texts
        self.tables = tables
        self.summarize_text = summarize_text
        self.image_dir = image_dir

    def get_table_and_text_summaries(self):
        """
        Generate table and text summaries from text tokens and table elements
        """
        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Text summary chain
        llm = get_model(mode='text')
        summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []
        table_summaries = []

        # Apply to text if texts are provided and summarization is requested
        text_tokens = [item['tokens'] for item in self.texts]
        if text_tokens and self.summarize_text:
            text_summaries = summarize_chain.batch(text_tokens, {"max_concurrency": 5})
        elif text_tokens:
            text_summaries = text_tokens

        # Apply to tables if tables are provided
        tables = [item['raw_text'] for item in self.tables]
        if tables:
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

        return text_summaries, table_summaries

    def generate_image_summaries(self):
        """
        Generate summaries and base64 encoded strings for images
        """
        # Store base64 encoded images
        img_base64_list = []

        # Store image summaries
        image_summaries = []

        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""

        # Apply to images
        path = self.image_dir

        for img_file in sorted(os.listdir(path)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                base64_image = encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(image_summarize(base64_image, prompt))

        return img_base64_list, image_summaries
