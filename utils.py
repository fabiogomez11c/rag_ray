from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.docstore.document import Document
from typing import Dict, List, Union


def path_to_uri(path, scheme="https://", domain="docs.ray.io"):
    return scheme + domain + str(path).split(domain)[-1]


def SectionBSHTMLLoader(BSHTMLLoader):
    """Load `HTML` files and parse them with `beautiful soup`, extract their sections only."""

    def __init__(
        self,
        file_path: str,
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        super(SectionBSHTMLLoader, self).__init__(
            file_path,
            open_encoding,
            bs_kwargs,
            get_text_separator,
        )

    def load(self) -> List[Document]:
        from bs4 import BeautifulSoup

        with open(self.file_path, "r", encoding=self.open_encoding) as f:
            soup = BeautifulSoup(f, **self.bs_kwargs)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        # sections part
        sections = soup.find_all("section")
        section_list = []
        for section in sections:
            section_id = section.get("id")
            section_text = ""
            if section_id:
                uri = path_to_uri(path=self.file_path)
                section_list.append(
                    Document(
                        page_content=section_text,
                        metadata={"source": uri, "title": title},
                    )
                )
        return section_list
