from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from utils import SectionBSHTMLLoader
from functools import reduce
from pathlib import Path

# list all html files using glob
files_path = Path("./docs.ray.io/")
html_files = list(files_path.rglob("*.html"))[0:10]
html_documents = [SectionBSHTMLLoader(html_file).load() for html_file in html_files]
html_documents = list(reduce(lambda x, y: y + x, html_documents, []))

# text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
# documents = text_splitter.split_documents(html_documents)
# html_ = html_files[0]
# document = BSHTMLLoader(html_).load()

print("Done")
