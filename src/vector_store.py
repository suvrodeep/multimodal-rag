import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
from tqdm import tqdm
from configparser import ConfigParser

import sys

sys.path.extend(['./src'])

from extract_doc_elements import ElementExtractor
from extract_doc_elements import set_environ_vars
from generate_summaries import Summarize
from tokenizer import Tokenizer


input_path = "./data/input"
image_path = "./data/image"
key_path = "./config/keys.ini"
config_path = "./config/model_config.ini"


def create_multi_vector_retriever(vectorstore,
                                  text_summaries: List[str],
                                  texts: List[Dict],
                                  table_summaries: List[str],
                                  tables: List[Dict],
                                  image_summaries: List[str],
                                  images: List[str]):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever: MultiVectorRetriever,
                      doc_summaries: List[str],
                      doc_contents: List[Dict | str]
                      ):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

        summary_docs = []
        for index, summary in enumerate(tqdm(doc_summaries, ncols=100)):
            summary_docs.append(Document(page_content=summary, metadata={id_key: doc_ids[index]}))
        # summary_docs = [
        #     Document(page_content=summary, metadata={id_key: doc_ids[index]})
        #     for index, summary in enumerate(doc_summaries)
        # ]

        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        print("Embedding text summaries")
        add_documents(retriever=retriever, doc_summaries=text_summaries, doc_contents=texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        print("Embedding table summaries")
        add_documents(retriever=retriever, doc_summaries=table_summaries, doc_contents=tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        print("Embedding image summaries")
        add_documents(retriever=retriever, doc_summaries=image_summaries, doc_contents=images)

    return retriever


# The vectorstore to use to index the summaries
config = ConfigParser()
config.read(key_path)
api_key = config['multimodal-rag']['API_Key']
vectorstore = Chroma(collection_name="multimodal-rag",
                     embedding_function=OpenAIEmbeddings(model='text-embedding-3-large',
                                                         openai_api_key=api_key)
                     )

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
texts, text_summaries = [], []
tables, table_summaries = [], []

for item in text_composites:
    texts.append({'raw_text': item['raw_text'],
                  'doc_path': item['doc_path']
                  })
    text_summaries.append(item['summary'])

for item in table_composites:
    tables.append({'text_as_html': item['text_as_html'],
                   'doc_path': item['doc_path']
                   })
    table_summaries.append(item['summary'])

# Generate image summaries
summarize.generate_image_summaries()
image_composites = summarize.image_summaries
images, image_summaries = [], []
for item in image_composites:
    images.append({'img_base64': item['img_base64'],
                   'image_path': item['image_path']
                   })
    image_summaries.append(item['summary'])

img_base64_list = [item['img_base64'] for item in images]

# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore=vectorstore,
    text_summaries=text_summaries,
    texts=texts,
    table_summaries=table_summaries,
    tables=tables,
    image_summaries=image_summaries,
    images=img_base64_list,
)

result = retriever_multi_vector_img.invoke("model comparison")
