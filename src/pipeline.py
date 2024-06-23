import torch
from haystack import Document, Pipeline
from prompt_template import prompt_builder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
import os
from getpass import getpass
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.utils import ComponentDevice
# load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
# generator = OpenAIGenerator(model="gpt-3.5-turbo")

if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = getpass("Enter Hugging Face API key:")

device = ComponentDevice.from_str('mps')
torch.mps.set_per_process_memory_fraction(0.0)
# generation_kwargs = {"max_length": 100, "num_return_sequences": 1, "top_k": 50}
generator = HuggingFaceLocalGenerator(model="meta-llama/Llama-2-7b-chat-hf", device=device)
document_store = InMemoryDocumentStore()

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

# docs example
docs = [
    Document(content="The capital of France is Paris.", meta={"name": "doc1"}),
    Document(content="The capital of Germany is Berlin.", meta={"name": "doc2"}),
    Document(content="The capital of Italy is Rome.", meta={"name": "doc3"}),
    Document(content="The capital of Spain is Madrid.", meta={"name": "doc4"}),
    Document(content="The capital of Portugal is Lisbon.", meta={"name": "doc5"}),
    Document(content="The capital of Switzerland is Bern.", meta={"name": "doc6"}),
]

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

retriever = InMemoryEmbeddingRetriever(document_store)

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")
basic_rag_pipeline.warm_up()

question = "What is the capital of Spain?"


# Run the pipeline
def run_pipeline(question: str):
    response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})
    return response["llm"]["replies"][0]


while True:
    question = input("Ask a question: ")
    print(run_pipeline(question))
    if question == "exit":
        break
