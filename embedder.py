from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from typing import Dict


class EmbedChunk:
    def __init__(self, model_name):
        if model_name == "text-embedding-ada-002":
            self.embedding_model = OpenAIEmbeddings(model=model_name)
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                # model_kwargs={"device": "cpu"},
                # encode_kwargs={
                #     "device": "cpu",
                #     "batch_size": 8,
                # },
            )

    def __call__(self, batch) -> Dict:
        embeddings = self.embedding_model.embed_documents([batch.page_content])
        return {
            "text": batch.page_content,
            "source": batch.metadata["source"],
            "embeddings": embeddings,
        }
