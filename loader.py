from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores.chroma import Chroma
from utils import SectionBSHTMLLoader
from functools import reduce
from pathlib import Path
import matplotlib.pyplot as plt

# list all html files using glob
files_path = Path("./docs.ray.io/")
html_files = list(files_path.rglob("*.html"))[0:50]
html_documents = [SectionBSHTMLLoader(html_file).load() for html_file in html_files]
html_documents = list(reduce(lambda x, y: y + x, html_documents, []))

lengths = [len(doc.page_content) for doc in html_documents]


"""
There are some sections with a lot of charecters, that could be an issue for the context length of the LLM.
"""
# plt.plot(lengths, marker="o")
# plt.title("Section Lengths")
# plt.ylabel("# of Characters")
# plt.xlabel("Section Number")
# plt.show()

# The original example uses RecursiveCharacterTextSplitter, is is better to use this method to maintain the chunk size
# plain CharacterTextSplitter could lead to chunks to big for the LLM context length, in which cases this method is better?
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], chunk_size=300, chunk_overlap=100
)
documents = text_splitter.split_documents(html_documents)
new_lengths = [len(doc.page_content) for doc in documents]

# plt.plot(new_lengths, marker="o")
# plt.title("Section Lengths")
# plt.ylabel("# of Characters")
# plt.xlabel("Section Number")
# plt.show()

print("Done")
