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

llm = OpenAIGenerator(
    model="gpt-3.5-turbo",
    streaming_callback=None,
    system_prompt=None,
)
template = """
Given the context please answer the question.
If the context does not contain the answer, just say that you don't know.
Context:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(
    template=template
)

document_store = ElasticsearchDocumentStore(
    host="http://localhost:9200/",
    index="default"
)
retriever = ElasticsearchEmbeddingRetriever(
    document_store=document_store,
    filters={},
    top_k=5
)

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

rag_pipeline = Pipeline()  # Define the pipeline
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("answer_builder", answer_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.add_component("prompt_builder", prompt_builder)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("retriever.documents", "answer_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
# rag_pipeline.connect("llm.metadata", "answer_builder.metadata")

rag_pipeline.max_loops_allowed = 100




