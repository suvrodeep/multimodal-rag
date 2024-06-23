from unstructured.partition.pdf import partition_pdf
from tqdm import tqdm
import os


def set_environ_vars():
    # Set environment variables for image padding
    os.environ['EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD'] = '10'
    os.environ['EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD'] = '10'


class ElementExtractor:
    def __init__(self, input_dir: str, image_dir: str, chunk_size: int = None):
        self.input_dir = input_dir
        self.image_dir = image_dir
        self.raw_elements = []
        self.raw_texts = []
        self.raw_tables = []

        if chunk_size is None:
            self.chunk_size = 4000
        else:
            self.chunk_size = chunk_size

    # Get raw elements
    def get_raw_elements(self):
        """
        Get raw elements from PDF
        """
        # Get elements
        file_paths = [self.input_dir + '/' + file for file in os.listdir(self.input_dir)
                      if file.split(".")[-1] == 'pdf']
        print(f'Identified {len(file_paths)} PDF files for processing')

        max_chunk_size = self.chunk_size
        preferred_chunk_size = self.chunk_size - 200
        text_combination_threshold = int(self.chunk_size / 2)

        for file_path in tqdm(file_paths, ncols=100):
            self.raw_elements.append(partition_pdf(filename=file_path,
                                                   chunking_strategy='by_title',
                                                   max_characters=max_chunk_size,
                                                   new_after_n_chars=preferred_chunk_size,
                                                   combine_text_under_n_chars=text_combination_threshold,
                                                   infer_table_structure=True,
                                                   extract_image_block_types=['Image'],
                                                   extract_image_block_output_dir=self.image_dir
                                                   ))

    # Categorize elements by type
    def categorize_elements(self):
        """
        Categorize extracted elements from a PDF into tables and texts.
        raw_pdf_elements: List of lists of unstructured.documents.elements
        """

        for pdf_elements in self.raw_elements:
            for element in pdf_elements:
                if "unstructured.documents.elements.Table" in str(type(element)):
                    table_metadata = element.metadata.to_dict()
                    self.raw_tables.append({'raw_text': element.text,
                                            'text_as_html': table_metadata['text_as_html'],
                                            'doc_path': table_metadata['file_directory'] + '/' + table_metadata[
                                                'filename']
                                            })
                elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                    text_metadata = element.metadata.to_dict()
                    self.raw_texts.append({'raw_text': element.text,
                                           'doc_path': text_metadata['file_directory'] + '/' + text_metadata['filename']
                                           })

        print(f'Got {len(self.raw_texts)} text elements')
        print(f'Got {len(self.raw_tables)} table elements')

