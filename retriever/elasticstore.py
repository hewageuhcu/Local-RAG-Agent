from langchain_elasticsearch import ElasticsearchStore
from config.settings import es_url, es_user, es_password, index_name

def get_vectorstore(documents, embeddings):
    return ElasticsearchStore.from_documents(
        documents, embeddings, es_url, es_user, es_password, index_name
    )
