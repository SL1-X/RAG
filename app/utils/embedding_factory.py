from app.services.settings_service import settings_service
from app.utils.logger import get_logger
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

logger = get_logger(__name__)


class EmbeddingFactory:
    @staticmethod
    def create_embeddings():
        settings = settings_service.get()
        embedding_provider = settings.get("embedding_provider")
        embedding_model_name = settings.get("embedding_model_name")
        embedding_api_key = settings.get("embedding_api_key")
        embedding_base_url = settings.get("embedding_base_url")
        try:
            if embedding_provider == "huggingface":
                embeddings = HuggingFaceEmbeddings(  # 这是一个本地模型 模型文件是在本地的
                    model_name=embedding_model_name,  # 不需要baseurl，也不需要apikey
                    model_kwargs={"device": "cpu"},
                    # normalize_embeddings指的是将向量转换为单位向量的的过程，也就是使其模长变为1，但是方向不变
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info(f"创建HuggingFaceEmbeddings:{embedding_model_name}")
            elif (
                embedding_provider == "openai"
            ):  # 这个不是本地模型，这是要调用远程OPENAI服务器
                embeddings = OpenAIEmbeddings(  # 不需要baseUrl,但需要apikey
                    model_name=embedding_model_name, openai_api_key=embedding_api_key
                )
                logger.info(f"创建OpenAIEmbeddings:{embedding_model_name}")
            elif (
                embedding_provider == "ollama"
            ):  # 调用的是本地服务 baseURL，但不需要apikey
                embeddings = OllamaEmbeddings(
                    model_name=embedding_model_name, base_url=embedding_base_url
                )
                logger.info(f"创建HuggingFaceEmbeddings:{embedding_model_name}")
            else:
                logger.warning(f"未知的Embedding提供商,默认使用huggingface")
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    # normalize_embeddings指的是将向量转换为单位向量的的过程，也就是使其模长变为1，但是方向不变
                    encode_kwargs={"normalize_embeddings": True},
                )
            return embeddings
        except Exception as e:
            logger.info(f"创建向量模型失败:{e}", exc_info=True)
            return HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"},
                # normalize_embeddings指的是将向量转换为单位向量的的过程，也就是使其模长变为1，但是方向不变
                encode_kwargs={"normalize_embeddings": True},
            )
