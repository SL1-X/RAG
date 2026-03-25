import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# 项目根目录的路径
class Config:
    BASE_DIR = Path(__file__).parent.parent
    # 加载环境变量中配置的密钥
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

    # 应用配置
    # 应用监听的主机地址
    APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
    # 服务器监听的端口号
    APP_PORT = os.environ.get("APP_PORT", 5001)
    # 是否启动调用模式
    APP_DEBUG = os.environ.get("APP_DEBUG", "false").lower() == "true"
    # 上传的文件的最大文件大小
    MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 104857600))  # 100M
    # 允许 上传的文件
    ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
    # 允许 上传的图片的扩展名
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
    # 允许 上传的图片的最大大小，默认为5M
    MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", 5242880))

    # 日志配置
    # 日志存放目录
    LOG_DIR = os.environ.get("LOG_DIR", "./logs")
    # 日志文件
    LOG_FILE = os.environ.get("LOG_FILE", "rag_lite.log")
    # 日志级别
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    # 是否启用文件日志
    LOG_ENABLE_FILE = os.environ.get("LOG_ENABLE_FILE", "true").lower() == "true"
    # 是否启用控制台
    LOG_ENABLE_CONSOLE = os.environ.get("LOG_ENABLE_CONSOLE", "true").lower() == "true"

    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", 3306)
    DB_USER = os.environ.get("DB_USER", "root")
    # DB_PASSWORD = os.environ.get("DB_PASSWORD", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "123456")
    DB_NAME = os.environ.get("DB_NAME", "rag")
    DB_CHARSET = os.environ.get("DB_CHARSET", "utf8mb4")

    # 存储的类型
    STORAGE_TYPE = os.environ.get("STORAGE_TYPE", "local")  # local / minio
    # 本地文件的存储目录
    STORAGE_DIR = os.environ.get("STORAGE_DIR", "./storages")

    # MinIO 配置（当 STORAGE_TYPE='minio' 时使用）
    MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
    MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME", "rag-lite")
    MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    MINIO_REGION = os.environ.get("MINIO_REGION", None)

    DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_API_KEY = os.environ.get(
        "DEEPSEEK_API_KEY", "sk-89cb1b5843ec4b01b3c6cf7cded5ea79"
    )
    DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    OPENAI_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    OPENAI_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    OPENAI_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    OLLAMA_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    OLLAMA_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    OLLAMA_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    # 指定向量数据库的类型
    VECTOR_DB_TYPE = os.environ.get("VECTOR_DB_TYPE", "chroma")  # chroma 或 milvus
    # 指定 chroma向量数据库的本地存储目录
    CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
