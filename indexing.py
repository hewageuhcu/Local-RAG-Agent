from langchain_community.document_loaders import WebBaseLoader
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document URLs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
documents = doc_splits

# Setup embeddings
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

# Create Elasticsearch vector store
db = ElasticsearchStore.from_documents(
    documents,
    embeddings,
    es_url="https://local-agent.es.us-central1.gcp.cloud.es.io",
    es_user="elastic",
    es_password="6kT3MqEoS1X1wj8VAqhI58qp",
    index_name="local agent",
)

# Create retriever
retriever = db.as_retriever()