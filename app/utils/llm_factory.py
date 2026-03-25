from app.utils.logger import get_logger
from app.config import Config
from app.services.settings_service import settings_service

logger = get_logger(__name__)


class LLMFactory:
    # 注册的LLM提供者，服务提供商，用于存储各个provider的构建函数
    _providers = {}

    @classmethod
    def register_provider(cls, provider_name, provider_factory):
        cls._providers[provider_name.lower()] = provider_factory
        logger.info(f"已经注册了LLM提供商:{provider_name}")

    @classmethod
    def create_llm(
        cls, settings=None, temperature=0.7, max_tokens=1024, streaming=True
    ):
        if settings is None:
            settings = settings_service.get()

        provider = settings.get("llm_provider", "deepseek").lower()
        # provider_factory = cls._providers[provider]
        # if provider_factory:
        #    return provider_factory(settings, temperature, max_tokens, streaming)
        # else:
        #    raise ValueError(f"不支持LLM提供商{provider}")
        if provider == "deepseek":
            return cls._create_deekpseek(settings, temperature, max_tokens, streaming)
        elif provider == "openai":
            return cls._create_openai(settings, temperature, max_tokens, streaming)
        elif provider == "ollama":
            return cls._create_ollama(settings, temperature, max_tokens, streaming)
        else:
            raise ValueError(f"不支持LLM提供商{provider}")

    @classmethod
    def _create_deekpseek(cls, settings, temperature, max_tokens, streaming):
        from langchain_deepseek import ChatDeepSeek

        model_name = settings.get("llm_model_name", Config.DEEPSEEK_CHAT_MODEL)
        api_key = settings.get("llm_api_key", Config.DEEPSEEK_API_KEY)
        base_url = settings.get("llm_base_url", Config.DEEPSEEK_BASE_URL)
        llm = ChatDeepSeek(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
        logger.info("已经创建DeepSeek LLM:{model_name}")
        return llm

    @classmethod
    def _create_openai(cls, settings, temperature, max_tokens, streaming):
        from langchain_openai import ChatOpenAI

        model_name = settings.get("llm_model_name", Config.OPENAI_CHAT_MODEL)
        api_key = settings.get("llm_api_key", Config.OPENAI_API_KEY)
        base_url = settings.get("llm_base_url", Config.OPENAI_BASE_URL)
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
        logger.info("已经创建OpenAI LLM:{model_name}")
        return llm

    @classmethod
    def _create_ollama(cls, settings, temperature, max_tokens, streaming):
        from langchain_community.chat_models import ChatOllama

        model_name = settings.get("llm_model_name", Config.OLLAMA_CHAT_MODEL)
        api_key = settings.get("llm_api_key", Config.OLLAMA_API_KEY)
        base_url = settings.get("llm_base_url", Config.OLLAMA_BASE_URL)
        llm = ChatOllama(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
        logger.info("已经创建Ollama LLM:{model_name}")
        return llm


LLMFactory.register_provider("deepseek", LLMFactory._create_deekpseek)
LLMFactory.register_provider("openai", LLMFactory._create_openai)
LLMFactory.register_provider("ollama", LLMFactory._create_ollama)
