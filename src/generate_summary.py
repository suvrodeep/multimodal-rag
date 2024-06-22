import sys
sys.path.extend(['./src'])

from extract_doc_elements import ElementExtractor
from tokenizer import Tokenizer

input_path = "../data/input"
image_path = "../data/image"

element_extractor = ElementExtractor(input_dir=input_path, image_dir=image_path)
element_extractor.get_raw_elements()
element_extractor.categorize_elements()

tokenizer = Tokenizer(raw_texts=element_extractor.raw_texts)
tokenizer.generate_tokens()
tokens = tokenizer.tokens

