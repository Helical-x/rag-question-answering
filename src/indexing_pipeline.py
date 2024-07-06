from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack import Pipeline

document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200/")

file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
pdf_converter = PyPDFToDocument()
document_joiner = DocumentJoiner()

document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)

document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_writer = DocumentWriter(document_store, DuplicatePolicy.SKIP)


indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=file_type_router, name="file_type_router")
indexing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
indexing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
indexing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
indexing_pipeline.add_component(instance=document_joiner, name="document_joiner")
indexing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
indexing_pipeline.add_component(instance=document_splitter, name="document_splitter")
indexing_pipeline.add_component(instance=document_embedder, name="document_embedder")
indexing_pipeline.add_component(instance=document_writer, name="document_writer")

indexing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
indexing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
indexing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
indexing_pipeline.connect("text_file_converter", "document_joiner")
indexing_pipeline.connect("pypdf_converter", "document_joiner")
indexing_pipeline.connect("markdown_converter", "document_joiner")
indexing_pipeline.connect("document_joiner", "document_cleaner")
indexing_pipeline.connect("document_cleaner", "document_splitter")
indexing_pipeline.connect("document_splitter", "document_embedder")
indexing_pipeline.connect("document_embedder", "document_writer")
