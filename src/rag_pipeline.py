import os
from getpass import getpass

from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Pipeline
from dotenv import load_dotenv

load_dotenv()

answer_builder = AnswerBuilder(
    pattern=None,
    reference_pattern=None,
)

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
generator = OpenAIGenerator(model="gpt-3.5-turbo")

# llm = OpenAIGenerator(
#     model="gpt-3.5-turbo",
#     streaming_callback=None,
#     system_prompt=None,
# )

template = """
Instructions: Write a response to the question below. When question is outside the context of the documents, answer Sorry, I can't answer your question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

document_store = ElasticsearchDocumentStore(
    hosts=f'http://{os.getenv("ELASTICSEARCH_HOST", "localhost")}:9200',
    index="default"
)
retriever = ElasticsearchEmbeddingRetriever(
    document_store=document_store,
    filters={},
    top_k=5
)

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")


rag_pipeline = Pipeline()  # Define the pipeline
# rag_pipeline.add_component("answer_builder", answer_builder)
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

# Now, connect the components to each other
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

rag_pipeline.max_loops_allowed = 100




