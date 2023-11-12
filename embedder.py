from langchain.embeddings.huggingface import HuggingFaceEmbeddings


class EmbedChunk:
    def __init__(self, model_name):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            # model_kwargs={"device": "cpu"},
            # encode_kwargs={
            #     "device": "cpu",
            #     "batch_size": 8,
            # },
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents([batch.page_content])
        return {
            "text": batch.page_content,
            "source": batch.metadata["source"],
            "embeddings": embeddings,
        }
