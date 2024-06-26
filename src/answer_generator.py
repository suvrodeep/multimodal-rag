import io
import re
import base64
import textwrap

from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from typing import List, Dict
from configparser import ConfigParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from PIL import Image

import sys

# sys.path.extend(['./src'])

from extract_doc_elements import ElementExtractor
from extract_doc_elements import set_environ_vars
from generate_summaries import Summarize
from tokenizer import Tokenizer
from vector_store import VectorStore, MultiModalRetriever

input_path = "../data/input"
image_path = "../data/image"
key_path = "../config/keys.ini"
config_path = "../config/model_config.ini"


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, img_format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(800, 600)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if type(doc) is str:
            if looks_like_base64(doc) and is_image_data(doc):
                doc = resize_base64_image(doc, size=(1200, 900))
                b64_images.append(doc)
            else:
                texts.append(doc)
        # If dict then append text element only
        elif type(doc) is dict:
            texts.append(doc[list(doc.keys())[0]])
    return {"images": b64_images, "texts": texts}


def prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "Assume an expert role based on the user provided question.\n"
            f"User provided question: {data_dict['question']}\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts, graphs or flow diagrams.\n"
            "The tables provided will be formatted as HTML. You can output the table if necessary.\n"
            "Use this information to provide answers to the user question. \n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


class Generation:
    def __init__(self):
        # Create RAG chain
        config_parser = ConfigParser()
        config_parser.read(key_path)
        self.key = config_parser['multimodal-rag']['API_Key']

        config_parser.read(config_path)
        self.model = config_parser['generation']['model']
        self.max_tokens = config_parser['generation']['max_tokens']
        self.temp = config_parser['generation']['temperature']

    def multi_modal_rag_chain(self, retriever):
        """
        Multi-modal RAG chain
        """
        # Multi-modal LLM
        model = ChatOpenAI(temperature=self.temp, model=self.model, max_tokens=self.max_tokens, api_key=self.key)

        # RAG pipeline
        chain = (
                {
                    "context": retriever | RunnableLambda(split_image_text_types),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(prompt_func)
                | model
                | StrOutputParser()
        )

        return chain


def generate_summary_composites():
    # Extract elements
    set_environ_vars()
    element_extractor = ElementExtractor(input_dir=input_path, image_dir=image_path)
    element_extractor.get_raw_elements()
    element_extractor.categorize_elements()

    # Tokenize text chunks
    tokenizer = Tokenizer(raw_texts=element_extractor.raw_texts)
    tokenizer.generate_tokens()

    # Get tables and tokenized text
    raw_tables = element_extractor.raw_tables
    tokenized_texts = tokenizer.tokens

    # Get summaries
    summarize = Summarize(texts=tokenized_texts,
                          tables=raw_tables,
                          image_dir=image_path,
                          key_path=key_path,
                          config_path=config_path)

    # Generate text and table summaries
    summarize.get_table_and_text_summaries()
    text_composites, table_composites = summarize.texts, summarize.tables

    # Generate image summaries
    summarize.generate_image_summaries()
    image_composites = summarize.image_summaries

    return text_composites, table_composites, image_composites


def get_retriever():
    vector_store = VectorStore(key_path=key_path, config_path=config_path)
    text_comp, table_comp, img_comp = generate_summary_composites()

    retriever = MultiModalRetriever(vectorstore=vector_store.create_vector_store(),
                                    text_composites=text_comp,
                                    table_composites=table_comp,
                                    image_composites=img_comp
                                    )
    return retriever.create_multi_vector_retriever()


def main():
    print("\n\nInitializing. Please wait...\n\n")
    multimodal_retriever = get_retriever()
    generation = Generation()
    chain_multimodal_rag = generation.multi_modal_rag_chain(multimodal_retriever)

    while True:
        query = input("\n\nPlease enter query: ")
        if query is None or query == "":
            print("\n\nNo input provided. Exiting chat loop. Exiting script.\n\n")
            exit(0)
        else:
            response = chain_multimodal_rag.invoke(query)
            print(f'\n\nResponse:\n{textwrap.fill(response, width=150)}')


if __name__ == "__main__":
    main()
