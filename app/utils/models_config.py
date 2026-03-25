# 模型配置
# 定义可用的 Embedding 模型和 LLM 模型列表
"""
模型配置
定义可用的 Embedding 模型和 LLM 模型列表
"""

# 定义向量嵌入模型（Embedding Models）的配置字典
EMBEDDING_MODELS = {
    # HuggingFace 嵌入模型
    "huggingface": {
        # 名称
        "name": "HuggingFace Embeddings",
        # 描述说明
        "description": "本地 HuggingFace 模型",
        # 可用模型列表
        "models": [
            # 第一个模型：all-MiniLM-L6-v2
            {
                # 模型名称
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                # 模型路径
                # "path": "C:/Users/lenovo/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2",
                "path": "sentence-transformers/all-MiniLM-L6-v2",
                # 向量维度
                "dimension": "384",
                # 描述
                "description": "轻量级多语言模型，速度快",
            },
            # 第二个模型：paraphrase-multilingual-MiniLM-L12-v2
            {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "dimension": "384",
                "description": "多语言模型，支持中文",
            },
            # 第三个模型：bge-small-zh-v1.5
            {
                "name": "BAAI/bge-small-zh-v1.5",
                "path": "BAAI/bge-small-zh-v1.5",
                "dimension": "512",
                "description": "中文优化模型",
            },
        ],
        # 是否需要 API Key
        "requires_api_key": True,
        # 是否需要 Base URL
        "requires_base_url": True,
    },
    # OpenAI 嵌入模型
    "openai": {
        "name": "OpenAI Embeddings",
        "description": "OpenAI 官方嵌入模型",
        # OpenAI 可用模型
        "models": [
            {
                "name": "text-embedding-3-small",
                "dimension": "1536",
                "description": "小型模型，速度快",
            },
            {
                "name": "text-embedding-3-large",
                "dimension": "3072",
                "description": "大型模型，精度高",
            },
            {
                "name": "text-embedding-ada-002",
                "dimension": "1536",
                "description": "经典模型",
            },
        ],
        "requires_api_key": True,
        "requires_base_url": True,
    },
    # 本地 Ollama 嵌入模型
    "ollama": {
        "name": "Ollama Embeddings",
        "description": "本地 Ollama 模型",
        # Ollama 可用嵌入模型
        "models": [
            {
                "name": "nomic-embed-text",
                "dimension": "768",
                "description": "通用文本嵌入模型",
            }
        ],
        "requires_api_key": False,
        "requires_base_url": True,
    },
}

# 定义 LLM（大模型，推理/对话模型）配置字典
LLM_MODELS = {
    # DeepSeek 模型配置
    "deepseek": {
        "name": "DeepSeek",
        "description": "DeepSeek API",
        # DeepSeek 可用模型列表
        "models": [
            # DeepSeek 对话模型
            {"name": "deepseek-chat", "description": "对话模型"},
            # DeepSeek 代码模型
            {"name": "deepseek-coder", "description": "代码模型"},
        ],
        "requires_api_key": True,
        "requires_base_url": True,
    },
    # OpenAI 大模型配置
    "openai": {
        "name": "OpenAI",
        "description": "OpenAI API",
        # OpenAI 可用大模型
        "models": [
            {"name": "gpt-4", "description": "GPT-4 模型"},
            {"name": "gpt-4-turbo", "description": "GPT-4 Turbo 模型"},
            {"name": "gpt-3.5-turbo", "description": "GPT-3.5 Turbo 模型"},
        ],
        "requires_api_key": True,
        "requires_base_url": False,
    },
    # 本地 Ollama 大模型配置
    "ollama": {
        "name": "Ollama",
        "description": "本地 Ollama 模型",
        # Ollama 可用大模型
        "models": [
            {"name": "llama2", "description": "Llama 2 模型"},
            {"name": "mistral", "description": "Mistral 模型"},
            {"name": "qwen", "description": "Qwen 模型"},
        ],
        "requires_api_key": False,
        "requires_base_url": True,
    },
}
