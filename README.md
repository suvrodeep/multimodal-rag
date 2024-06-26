This is a baseline model application to understand the pros and cons different RAG architectures for multimodal elements. It processes documents and extracts texts, image and table elements from them to embed them in a common embedding space using a MultiVector Store. The original raw elemnets are stored in a doc-store. Currently only LangChain In-Memory doc-store is implemented.

Frameworks used:
* Langchain orchestration framework
* API calls to GPT 3.5 Turbo for text and and table summarization
* GPT-4-o for image summarization and final answer synthesis

How to test:

1. Create a /data directory at root and two sub directories /data/image and /data/input
2. Put input files in /data/input
3. Run answer_generator.py

ToDo:
* A chat interface using Streamlit will be implemented
* Deployment to Google Cloud Run as a service
