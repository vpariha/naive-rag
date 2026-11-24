from functools import lru_cache

from langchain_chroma import Chroma

from naive_rag.config.setting import app_config
from naive_rag.model.model import get_embedding_model


@lru_cache(maxsize=1)
def get_vectore_store():
    return Chroma(
        collection_name=app_config.CHROMA_COLLECTION_NAME,
        persist_directory=app_config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=get_embedding_model(),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )  # You can specify an embedding function if needed
