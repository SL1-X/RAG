from app.models.settings import Settings
from app.services.base_service import BaseService
from app.config import Config
from pathlib import Path
import json


class SettingsService(BaseService[Settings]):
    _EXTRA_SETTINGS_FILE = (
        Path(__file__).resolve().parents[2] / "storages" / "retrieval_tuning.json"
    )

    def _read_extra_settings(self) -> dict:
        file_path = self._EXTRA_SETTINGS_FILE
        try:
            if not file_path.exists():
                return {}
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_extra_settings(self, data: dict):
        file_path = self._EXTRA_SETTINGS_FILE
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get(self):
        extra = self._read_extra_settings()
        with self.session() as session:
            settings = session.query(Settings).filter_by(id="global").first()
            print("session.query settings", settings)
            if settings:
                merged = settings.to_dict()
                merged.update(
                    {
                        "use_rerank": extra.get("use_rerank", True),
                        "rerank_candidate_k": extra.get("rerank_candidate_k", 24),
                        "rerank_language_mode": extra.get("rerank_language_mode", "auto"),
                    }
                )
                return merged
            else:
                defaults = self._get_default_settings()
                defaults.update(
                    {
                        "use_rerank": extra.get("use_rerank", True),
                        "rerank_candidate_k": extra.get("rerank_candidate_k", 24),
                        "rerank_language_mode": extra.get("rerank_language_mode", "auto"),
                    }
                )
                return defaults

    # 获取默认设置的方法
    def _get_default_settings(self) -> dict:
        """获取默认设置"""
        # 返回包含所有默认字段值的字典
        return {
            "id": "global",  # 设置主键
            "embedding_provider": "huggingface",  # 默认 embedding provider
            # "embedding_model_name": "C:/Users/lenovo/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2",  # 默认 embedding 模型
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",  # 默认 embedding 模型
            "embedding_api_key": "embedding_api_key",  # 默认无 embedding API key
            "embedding_base_url": "embedding_base_url",  # 默认无 embedding base url
            "llm_provider": "deepseek",  # 默认 LLM provider
            "llm_model_name": Config.DEEPSEEK_CHAT_MODEL,  # 默认 LLM 模型
            "llm_api_key": Config.DEEPSEEK_API_KEY,  # 配置里的默认 LLM API key
            "llm_base_url": Config.DEEPSEEK_BASE_URL,  # 配置里的默认 LLM base url
            "llm_temperature": 0.7,  # 默认温度
            "chat_system_prompt": "你是一个专业的AI助手。请友好、准确地回答用户的问题。",  # 聊天系统默认提示词
            "rag_system_prompt": "你是一个专业的AI助手。请基于文档内容回答问题。",  # RAG系统提示词
            "rag_query_prompt": "文档内容：\n{context}\n\n问题：{question}\n\n请基于文档内容回答问题。如果文档中没有相关信息，请明确说明。",  # RAG查询提示词
            # "retrieval_mode": "vector",  # 默认检索模式
            "retrieval_mode": "hybrid",  # 默认检索模式
            "vector_threshold": 0.2,  # 向量检索阈值
            "keyword_threshold": 0.0,  # 关键词检索阈值
            "vector_weight": 0.7,  # 检索混合权重
            "top_k": 5,  # 返回结果数量
            "use_rerank": True,  # 是否启用重排
            "rerank_candidate_k": 24,  # 重排候选数量
            "rerank_language_mode": "auto",  # auto|always_on|always_off
        }

    def update(self, data):
        extra_payload = {}
        if "use_rerank" in data:
            value = data.get("use_rerank")
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "on"}
            extra_payload["use_rerank"] = bool(value)
        if "rerank_candidate_k" in data:
            try:
                candidate_k = int(data.get("rerank_candidate_k"))
            except Exception:
                candidate_k = 24
            extra_payload["rerank_candidate_k"] = max(5, min(candidate_k, 200))
        if "rerank_language_mode" in data:
            mode = str(data.get("rerank_language_mode") or "auto").strip().lower()
            if mode not in {"auto", "always_on", "always_off"}:
                mode = "auto"
            extra_payload["rerank_language_mode"] = mode

        if extra_payload:
            current_extra = self._read_extra_settings()
            current_extra.update(extra_payload)
            self._write_extra_settings(current_extra)

        with self.transaction() as session:
            settings = session.query(Settings).filter_by(id="global").first()
            if not settings:
                settings = Settings(id="global")
                session.add(settings)
            for key, value in data.items():
                if hasattr(settings, key) and value is not None:
                    setattr(settings, key, value)
            session.flush()
            session.refresh(settings)
            result = settings.to_dict()
            result.update(self._read_extra_settings())
            return result


settings_service = SettingsService()
