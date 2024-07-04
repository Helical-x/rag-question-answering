from haystack.components.converters.txt import TextFileToDocument
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers.document_writer import DocumentWriter, DuplicatePolicy
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack import Pipeline
indexing_pipeline = Pipeline()  # Define the pipeline

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
    remove_regex=None,
    remove_repeated_substrings=False,
    remove_substrings=None
)

converter = TextFileToDocument()
splitter = DocumentSplitter(
    split_by="word",
    split_overlap=0,
    split_length=500
)

doc_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
document_store = ElasticsearchDocumentStore(
    host="http://localhost:9200/",
    index="default"
)
writer = DocumentWriter(
    document_store=document_store,
    policy=DuplicatePolicy.SKIP
)

# add components to the pipeline
indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("converter", converter)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("doc_embedder", doc_embedder)
indexing_pipeline.add_component("writer", writer)

# connect the components
indexing_pipeline.connect("converter.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "doc_embedder.documents")  # corrected line
indexing_pipeline.connect("doc_embedder.documents", "writer.documents")  # corrected line
indexing_pipeline.max_loops_allowed = 100