from loader import Loader
from embedder import EmbedChunk
from llm import LLM
from langchain.vectorstores.chroma import Chroma
import chromadb


def main():
    embedder = EmbedChunk("distilbert-base-uncased")
    db = chromadb.PersistentClient("./db/")

    # check if collection exists
    if "langchain" in db.list_collections():
        db_langchain = Chroma(
            client=db,
            collection_name="langchain",
            embedding_function=embedder.embedding_model,
        )
    else:
        loader = Loader()
        db_langchain = Chroma.from_documents(
            documents=loader.documents,
            embedding=embedder.embedding_model,
            persis_directory="./db/",
        )

    # TODO esto hay que cambiarlo para que sea un LLM de langchain
    # TODO como funciona la memoria cuando estoy haciendo RAG?
    llm = LLM()
    query = "How to install ray in python?"
    sim_docs = db.similarity_search(query, k=5)
    response = llm.get_response(user_request=query)

    # for doc in sim_docs:
    #     # TODO pienso que el context length es bien importante, muchas veces esto solo trae algunos pedazos de codigo,
    #     # es muy dificil deducir la mejor respuesta con esos chunks tan malos
    #     print(doc.page_content)
    #     print(doc.metadata["source"])
    #     print("----------------------------")

    # TODO luego de implementar un LLM de langchain y hacer varias pruebas, se debe pasar al evaluador del RAG
    # la idea es que los embeddings sean gestionados por un LLM de huggingface pero la respuesta se de con OpenAI


if __name__ == "__main__":
    main()
