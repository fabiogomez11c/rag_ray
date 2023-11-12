from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from utils import SectionBSHTMLLoader
from functools import reduce
from pathlib import Path


class Loader:
    def __init__(self) -> None:
        # list all html files using glob
        files_path = Path("./docs.ray.io/")
        html_files = list(files_path.rglob("*.html"))
        html_documents = [
            SectionBSHTMLLoader(html_file).load() for html_file in html_files
        ]

        # properties to store in object
        self.html_documents = list(reduce(lambda x, y: y + x, html_documents, []))
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], chunk_size=300, chunk_overlap=100
        )

        self.documents = self.text_splitter.split_documents(self.html_documents)
        # embedder = EmbedChunk("distilbert-base-uncased")

        # db = Chroma.from_documents(documents, embedding=embedder.embedding_model)

        # query = "How to use ray?"
        # docs = db.similarity_search(query, k=5)
