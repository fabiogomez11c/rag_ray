from loader import Loader
from embedder import EmbedChunk
from llm import LLM
from langchain.vectorstores.chroma import Chroma
import chromadb


def main():
    embedder = EmbedChunk("text-embedding-ada-002")
    db = chromadb.PersistentClient("./db/")

    # check if collection exists
    collections = [col.name for col in db.list_collections()]
    if "langchain" in collections:
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
            persist_directory="./db2/",
        )

    # TODO como funciona la memoria cuando estoy haciendo RAG?
    llm = LLM()
    query = "Which operating systems does Ray support?"
    sim_docs = db_langchain.similarity_search(query, k=5)
    generated_context = [doc.page_content for doc in sim_docs]
    generated_context = "\n".join(generated_context)
    response = llm.get_response(user_request=query, generated_context=generated_context)

    # TODO luego de implementar un LLM de langchain y hacer varias pruebas, se debe pasar al evaluador del RAG
    # la idea es que los embeddings sean gestionados por un LLM de huggingface pero la respuesta se de con OpenAI
    return response


if __name__ == "__main__":
    main()
