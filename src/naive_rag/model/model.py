from functools import lru_cache

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings import Embeddings, init_embeddings

from naive_rag.config.setting import app_config

        
@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    return init_embeddings(provider=app_config.EMBEDDING_MODEL_PROVIDER,
                                        model=app_config.EMBEDDING_MODEL_NAME,
                                        encode_kwargs={
                                            "normalize_embeddings": True,
                                        },
                                        query_encode_kwargs={
                                            "normalize_embeddings": True,
                                        },)
    
@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    return init_chat_model(model_provider=app_config.CHAT_MODEL_PROVIDER,
                                        model=app_config.CHAT_MODEL_NAME,
                                        temperature=app_config.CHAT_MODEL_TEMPERATURE,
                                        )