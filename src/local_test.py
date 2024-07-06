from src.rag_pipeline import rag_pipeline
from src.indexing_pipeline import indexing_pipeline
from pathlib import Path

output_dir = "../example_data"

indexing_pipeline.run({"file_type_router": {"sources": list(Path(output_dir).glob("**/*"))}})


question = "What's atmosphere?"

response = rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0])
