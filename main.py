from loader import Loader
from embedder import EmbedChunk
from langchain.vectorstores.chroma import Chroma


def main():
    loader = Loader()
    embedder = EmbedChunk("distilbert-base-uncased")
    db = Chroma.from_documents(
        documents=loader.documents,
        embedding=embedder.embedding_model,
        persist_directory="./db/",
    )

    query = "How to install ray in python?"
    sim_docs = db.similarity_search(query, k=5)

    for doc in sim_docs:
        print(doc.page_content)
        print(doc.metadata["source"])
        print("----------------------------")


if __name__ == "__main__":
    main()
