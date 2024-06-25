import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
from tqdm import tqdm
from configparser import ConfigParser


class VectorStore:
    def __init__(self, key_path: str, config_path: str):
        self.key_path = key_path
        self.config_path = config_path

    def create_vector_store(self):
        # The vectorstore to use to index the summaries
        config = ConfigParser()

        config.read(self.key_path)
        api_key = config['multimodal-rag']['API_Key']

        config.read(self.config_path)
        embedding_model = config['embedding']['model']

        vectorstore = Chroma(collection_name="multimodal-rag",
                             embedding_function=OpenAIEmbeddings(model=embedding_model,
                                                                 openai_api_key=api_key)
                             )
        return vectorstore


class MultiModalRetriever:
    def __init__(self,
                 vectorstore,
                 text_composites: List[Dict],
                 table_composites: List[Dict],
                 image_composites: List[Dict]
                 ):

        self.vector_store = vectorstore
        self.text_composites = text_composites
        self.table_composites = table_composites
        self.image_composites = image_composites

    def split_composites(self, composite_type: str):
        """
        Break composite elements into their constituent raw texts and summaries
        """
        if composite_type == 'table':
            tables, table_summaries = [], []
            for item in self.table_composites:
                tables.append({'text_as_html': item['text_as_html'],
                               'doc_path': item['doc_path']
                               })
                table_summaries.append(item['summary'])
            return tables, table_summaries
        elif composite_type == 'text':
            texts, text_summaries = [], []
            for item in self.text_composites:
                texts.append({'raw_text': item['raw_text'],
                              'doc_path': item['doc_path']
                              })
                text_summaries.append(item['summary'])
            return texts, text_summaries
        elif composite_type == 'image':
            images, image_summaries = [], []
            for item in self.image_composites:
                images.append(item['img_base64'])
                image_summaries.append(item['summary'])
            return images, image_summaries
        else:
            return [], []

    def create_multi_vector_retriever(self):
        """
        Create retriever that indexes summaries, but returns raw images or texts
        """
        texts, text_summaries = self.split_composites(composite_type='text')
        tables, table_summaries = self.split_composites(composite_type='table')
        images, image_summaries = self.split_composites(composite_type='image')

        # Initialize the storage layer
        store = InMemoryStore()
        id_key = "doc_id"

        # Create the multi-vector retriever
        retriever = MultiVectorRetriever(vectorstore=self.vector_store,
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

