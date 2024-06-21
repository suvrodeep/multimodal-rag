from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, ElementMetadata, ElementType, Text, Image
import os
from typing import List


# Categorize elements by type
def categorize_elements(raw_pdf_elements: List[List[Element]]) -> (List[str], List[dict[str, str]]):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of lists of unstructured.documents.elements
    """
    tables = []
    texts = []
    for pdf_elements in raw_pdf_elements:
        for element in pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append({'raw_text': element.text,
                               'text_as_html': element.metadata.to_dict()['text_as_html']})
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(element.text)
    return texts, tables


# File path
input_path = "../data/input"
image_path = "../data/image"

# Get elements
file_paths = [input_path + '/' + file for file in os.listdir(input_path) if file.split(".")[-1] == 'pdf']
raw_pdf_elements = []
for file_path in file_paths:
    raw_pdf_elements.append(partition_pdf(filename=file_path,
                                          chunking_strategy='by_title',
                                          max_characters=4096,
                                          extract_image_block_types=['Image'],
                                          extract_image_block_output_dir=image_path,
                                          infer_table_structure=True))

# Get text, tables
texts, tables = categorize_elements(raw_pdf_elements)

# Optional: Enforce a specific token size for texts
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4096, chunk_overlap=0)

joined_texts = " ".join(texts)
texts_4k_token = text_splitter.split_text(joined_texts)
