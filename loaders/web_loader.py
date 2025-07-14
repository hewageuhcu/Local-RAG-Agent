from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ...
]

def load_documents():
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]
