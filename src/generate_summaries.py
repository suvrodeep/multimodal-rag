import base64
import os

from typing import List, Dict
from configparser import ConfigParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(model, img_base64, prompt):
    """Make image summary"""
    msg = model.invoke(
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
    def __init__(self, texts: List[Dict],
                 tables: List[Dict],
                 image_dir: str,
                 key_path: str,
                 config_path: str,
                 summarize_text: bool = True):
        self.texts = texts
        self.tables = tables
        self.summarize_text = summarize_text
        self.image_dir = image_dir
        self.key_path = key_path
        self.config_path = config_path

        # Store image summaries
        self.image_summaries = []

    def get_model(self, mode: str) -> ChatOpenAI:
        if mode == 'image':
            config_section = 'image-summary'
        else:
            config_section = 'text-summary'

        # Read model config
        config = ConfigParser()

        config.read(self.key_path)
        api_key = config['multimodal-rag']['API_Key']

        config.read(self.config_path)
        model = config[config_section]['model']
        temp = int(config[config_section]['temperature'])
        max_tokens = int(config[config_section]['max_tokens'])

        return ChatOpenAI(model=model, max_tokens=max_tokens, api_key=api_key, temperature=temp)

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
        llm = self.get_model(mode='text')
        summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []
        table_summaries = []

        # Apply to text if texts are provided and summarization is requested
        text_tokens = [item['tokens'] for item in self.texts]
        print(f'Summarizing {len(text_tokens)} tokenized text elements')
        if text_tokens and self.summarize_text:
            text_summaries = summarize_chain.batch(text_tokens, {"max_concurrency": 5})
        elif text_tokens:
            text_summaries = text_tokens

        # Apply to tables if tables are provided
        table_texts = [item['raw_text'] for item in self.tables]
        print(f'Summarizing {len(table_texts)} table elements')
        if table_texts:
            table_summaries = summarize_chain.batch(table_texts, {"max_concurrency": 5})

        # Add table and text metadata to summaries
        for text, summary in zip(self.texts, text_summaries):
            text['summary'] = summary
        for table, summary in zip(self.tables, table_summaries):
            table['summary'] = summary

    def generate_image_summaries(self):
        """
        Generate summaries and base64 encoded strings for images
        """
        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""

        # Initialize model
        summary_model = self.get_model(mode='image')

        # Apply summarization to images
        images = [self.image_dir + '/' + file for file in os.listdir(self.image_dir)
                  if file.split(".")[-1] in ['jpg', 'png']]
        print(f'Summarizing {len(images)} extracted image elements')

        for img_file in tqdm(images, ncols=100):
            base64_image = encode_image(img_file)
            self.image_summaries.append({'img_base64': base64_image,
                                         'summary': image_summarize(model=summary_model,
                                                                    img_base64=base64_image,
                                                                    prompt=prompt)
                                         })
