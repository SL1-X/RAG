from app.services.settings_service import settings_service
from app.utils.llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import get_logger
from app.services.rag_service import PIPELINE_MODE_FULL, rag_service

logger = get_logger(__name__)


class ChatService:
    def __init__(self):
        self.settings = settings_service.get()

    def _normalize_history_messages(self, history):
        """将历史消息转换为 ChatPromptTemplate 可识别的消息格式。"""
        if not history:
            return []
        role_map = {"user": "human", "assistant": "ai", "human": "human", "ai": "ai"}
        normalized = []
        for item in history:
            role = role_map.get((item.get("role") or "").lower())
            content = (item.get("content") or "").strip()
            if role and content:
                normalized.append((role, content))
        return normalized

    def chat_stream(self, question, history=None):
        temperature = float(self.settings.get("llm_temperature", " 0.7"))
        temperature = max(0.0, min(temperature, 2.0))
        chat_system_prompt = self.settings.get("chat_system_prompt")
        if not chat_system_prompt:
            chat_system_prompt = "你是一个专业的AI助手。请友好、准确地回答用户的问题。"
        llm = LLMFactory.create_llm(self.settings, temperature=temperature)
        messages = [("system", chat_system_prompt)]
        messages.extend(self._normalize_history_messages(history))
        messages.append(("human", question))
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        # 服务器准备开始向客户端发送消息
        yield {"type": "start", "content": ""}
        full_answer = ""
        try:
            # 遍历大模型生成的每一段代码
            for chunk in chain.stream({}):
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    full_answer += content
                    yield {"type": "content", "content": content}
        except Exception as e:
            logger.error(f"流式生成时出错:{e}")
            yield {"type": "error", "content": f"流式生成时出错:{e}"}
            return
        yield {"type": "done", "content": "", "metadata": {"question": question}}

    # 流式知识库问答：支持 full / retrieve_only / generate_only
    def ask_stream(
        self,
        kb_id,
        question,
        pipeline_mode: str = PIPELINE_MODE_FULL,
        context: str | None = None,
        history=None,
    ):
        return rag_service.ask_stream(
            kb_id=kb_id,
            question=question,
            pipeline_mode=pipeline_mode,
            context=context,
            history=history,
        )


chat_service = ChatService()
