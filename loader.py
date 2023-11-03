from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from pathlib import Path

# list all html files using glob
files_path = Path("./docs.ray.io/")
html_files = list(files_path.rglob("*.html"))[0:10]
html_documents = [BSHTMLLoader(html_file).load()[0] for html_file in html_files]

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
documents = text_splitter.split_documents(html_documents)
# html_ = html_files[0]
# document = BSHTMLLoader(html_).load()

print("Done")
