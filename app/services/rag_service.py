from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from app.utils.logger import get_logger
from app.services.settings_service import settings_service
from app.services.retrieval_service import retrieval_service
from app.utils.llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate

logger = get_logger(__name__)

# 与检索设置中的 vector / keyword / hybrid 不同：这里表示「整条 RAG 流水线」如何跑
PIPELINE_MODE_FULL = "full"
PIPELINE_MODE_RETRIEVE_ONLY = "retrieve_only"
PIPELINE_MODE_GENERATE_ONLY = "generate_only"
PIPELINE_MODE_VECTOR_GENERATE = "vector_generate"
PIPELINE_MODE_KEYWORD_GENERATE = "keyword_generate"
PIPELINE_MODE_HYBRID_GENERATE = "hybrid_generate"
# 三路检索并行 + 三路生成并行，各自上下文独立
PIPELINE_MODE_TRIPLE_PARALLEL = "triple_parallel"
VALID_PIPELINE_MODES = frozenset(
    {
        PIPELINE_MODE_FULL,
        PIPELINE_MODE_RETRIEVE_ONLY,
        PIPELINE_MODE_GENERATE_ONLY,
        PIPELINE_MODE_VECTOR_GENERATE,
        PIPELINE_MODE_KEYWORD_GENERATE,
        PIPELINE_MODE_HYBRID_GENERATE,
        PIPELINE_MODE_TRIPLE_PARALLEL,
    }
)

_TRIPLE_BRANCH_ORDER = ("vector", "keyword", "hybrid")


class RAGService:
    def __init__(self):
        pass

    def _get_rag_prompt(self, settings: dict) -> ChatPromptTemplate:
        rag_system_prompt = settings.get("rag_system_prompt")
        rag_query_prompt = settings.get("rag_query_prompt")
        return ChatPromptTemplate.from_messages(
            [("system", rag_system_prompt), ("human", rag_query_prompt)]
        )

    def _retrieve_documents(
        self,
        kb_id,
        question,
        settings: dict | None = None,
        *,
        retrieval_mode: str | None = None,
    ):
        """retrieval_mode 显式指定时忽略设置里的 retrieval_mode（用于三路并行）。"""
        settings = settings or settings_service.get()
        collection_name = f"kb_{kb_id}"
        retrieval_mode = (
            retrieval_mode
            if retrieval_mode is not None
            else settings.get("retrieval_mode", "vector")
        )
        if retrieval_mode == "vector":
            docs = retrieval_service.vector_search(
                collection_name=collection_name, query=question, rerank=True
            )
        elif retrieval_mode == "keyword":
            docs = retrieval_service.keyword_search(
                collection_name=collection_name, query=question, rerank=True
            )
        elif retrieval_mode == "hybrid":
            docs = retrieval_service.hybrid_search(
                collection_name=collection_name, query=question
            )
        else:
            logger.warning(f"未知的检索模型:{retrieval_mode},转化使用向量检索")
            docs = retrieval_service.vector_search(
                collection_name=collection_name, query=question
            )
        logger.info(f"使用{retrieval_mode}模型检索到{len(docs)}个文档")
        return docs

    @staticmethod
    def build_context_from_documents(docs) -> str:
        return "\n\n".join(
            [
                f"文档{i + 1} ({doc.metadata.get('doc_name', '未知文档')}):\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ]
        )

    @staticmethod
    def build_context_from_history(history) -> str:
        """
        将历史消息格式化为可注入提示词的文本，默认仅保留最近 10 条以控制长度。
        """
        if not history:
            return ""
        lines = []
        for item in history[-10:]:
            role = (item.get("role") or "").strip().lower()
            content = (item.get("content") or "").strip()
            if not content:
                continue
            speaker = "用户" if role in {"user", "human"} else "助手"
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def _merge_context_and_history(self, context: str, history) -> str:
        history_text = self.build_context_from_history(history)
        if not history_text:
            return context or ""
        if not context:
            return f"对话历史：\n{history_text}"
        return f"对话历史：\n{history_text}\n\n文档上下文：\n{context}"

    def _rewrite_query_from_history(
        self, question: str, history, settings: dict
    ) -> str:
        """
        基于历史对话把当前问题改写为“可独立检索”的查询语句。
        失败或无历史时回退到原问题。
        """
        history_text = self.build_context_from_history(history)
        if not history_text:
            return question
        try:
            llm = LLMFactory.create_llm(
                settings,
                temperature=0.0,
                max_tokens=256,
                streaming=False,
            )
            rewrite_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是检索查询改写器。请将用户当前问题改写成可独立检索的一句话，"
                        "补全必要指代，不要回答问题，不要解释，只输出改写后的检索问题本身。",
                    ),
                    (
                        "human",
                        "历史对话：\n{history}\n\n当前问题：\n{question}\n\n输出改写后的检索问题：",
                    ),
                ]
            )
            chain = rewrite_prompt | llm
            out = chain.invoke({"history": history_text, "question": question})
            rewritten = (out.content if getattr(out, "content", None) else str(out)).strip()
            if not rewritten:
                return question
            return rewritten.splitlines()[0].strip() or question
        except Exception as e:
            logger.warning(f"查询改写失败，回退原问题: {e}")
            return question

    def retrieve(self, kb_id, question) -> tuple[list, str]:
        """仅检索：返回 LangChain Document 列表与拼好的上下文字符串。"""
        settings = settings_service.get()
        docs = self._retrieve_documents(kb_id, question, settings)
        context = self.build_context_from_documents(docs)
        return docs, context

    @staticmethod
    def _extract_retrieval_debug(docs) -> dict | None:
        for doc in docs or []:
            metadata = doc.metadata or {}
            debug = metadata.get("retrieval_debug")
            if isinstance(debug, dict):
                return debug
        return None

    def _stream_llm_answer(self, question: str, context: str, settings: dict, history=None):
        llm = LLMFactory.create_llm(settings)
        rag_prompt = self._get_rag_prompt(settings)
        chain = rag_prompt | llm
        merged_context = self._merge_context_and_history(context, history)
        for chunk in chain.stream({"context": merged_context, "question": question}):
            if chunk.content:
                yield chunk.content

    def _invoke_rag_answer(
        self, question: str, context: str, settings: dict, history=None
    ) -> str:
        llm = LLMFactory.create_llm(settings)
        rag_prompt = self._get_rag_prompt(settings)
        chain = rag_prompt | llm
        merged_context = self._merge_context_and_history(context, history)
        out = chain.invoke({"context": merged_context, "question": question})
        return out.content if getattr(out, "content", None) else str(out)

    def _run_triple_retrieval_branch(
        self, kb_id: str, question: str, settings: dict, branch: str
    ):
        """单路检索：返回分支、sources、上下文与耗时。"""
        started = perf_counter()
        try:
            docs = self._retrieve_documents(
                kb_id, question, settings, retrieval_mode=branch
            )
            context = self.build_context_from_documents(docs)
            sources = self._extract_citations(docs)
            elapsed_ms = int((perf_counter() - started) * 1000)
            return {
                "branch": branch,
                "sources": sources,
                "context": context,
                "retrieved_chunks": len(docs),
                "retrieval_debug": self._extract_retrieval_debug(docs),
                "retrieval_elapsed_ms": elapsed_ms,
                "error": None,
            }
        except Exception as e:
            logger.exception("triple_parallel 检索分支 %s 失败", branch)
            elapsed_ms = int((perf_counter() - started) * 1000)
            return {
                "branch": branch,
                "sources": [],
                "context": "",
                "retrieved_chunks": 0,
                "retrieval_debug": None,
                "retrieval_elapsed_ms": elapsed_ms,
                "error": str(e),
            }

    def _run_triple_generation_branch(
        self,
        *,
        question: str,
        settings: dict,
        branch: str,
        context: str,
        retrieval_error: str | None = None,
        history=None,
    ):
        """单路生成：返回分支答案与生成耗时。"""
        started = perf_counter()
        try:
            # 检索失败时给出可读结果，避免空白分支。
            if retrieval_error:
                answer = f"（{branch} 分支检索失败，无法基于该路文档生成答案：{retrieval_error}）"
            else:
                answer = self._invoke_rag_answer(
                    question, context, settings, history=history
                )
            elapsed_ms = int((perf_counter() - started) * 1000)
            return {
                "branch": branch,
                "answer": answer,
                "generation_elapsed_ms": elapsed_ms,
                "error": None,
            }
        except Exception as e:
            logger.exception("triple_parallel 生成分支 %s 失败", branch)
            elapsed_ms = int((perf_counter() - started) * 1000)
            return {
                "branch": branch,
                "answer": f"（本路生成出错：{e}）",
                "generation_elapsed_ms": elapsed_ms,
                "error": str(e),
            }

    def generate_stream(self, question: str, context: str = "", history=None):
        """仅生成：不访问向量库，用传入的 context 与 question 走 RAG 提示词 + LLM。"""
        settings = settings_service.get()
        yield {"type": "start", "content": ""}
        try:
            for text in self._stream_llm_answer(
                question, context, settings, history=history
            ):
                yield {"type": "content", "content": text}
        except Exception as e:
            logger.error(f"RAG 生成阶段出错: {e}")
            yield {"type": "error", "content": str(e)}
            return
        yield {
            "type": "done",
            "content": "",
            "sources": [],
            "metadata": {
                "question": question,
                "pipeline_mode": PIPELINE_MODE_GENERATE_ONLY,
                "context_chars": len(context or ""),
            },
        }

    def retrieve_only_stream(self, kb_id, question, retrieval_query: str | None = None):
        """仅检索：不调用大模型，在 done 中返回 sources 与元数据。"""
        settings = settings_service.get()
        yield {"type": "start", "content": ""}
        query_for_retrieval = retrieval_query or question
        filtered_docs = self._retrieve_documents(kb_id, query_for_retrieval, settings)
        sources = self._extract_citations(filtered_docs)
        retrieval_debug = self._extract_retrieval_debug(filtered_docs)
        yield {
            "type": "done",
            "content": "",
            "sources": sources,
            "metadata": {
                "kb_id": kb_id,
                "question": question,
                "retrieval_query": query_for_retrieval,
                "retrieved_chunks": len(filtered_docs),
                "retrieval_debug": retrieval_debug,
                "pipeline_mode": PIPELINE_MODE_RETRIEVE_ONLY,
                "hint": "仅检索模式：未调用大模型。",
            },
        }

    def full_rag_stream(
        self,
        kb_id,
        question,
        retrieval_mode: str | None = None,
        retrieval_query: str | None = None,
        history=None,
    ):
        """完整 RAG：先检索再生成（原 ask_stream 行为）。"""
        settings = settings_service.get()
        yield {"type": "start", "content": ""}
        query_for_retrieval = retrieval_query or question
        filtered_docs = self._retrieve_documents(
            kb_id, query_for_retrieval, settings, retrieval_mode=retrieval_mode
        )
        context = self.build_context_from_documents(filtered_docs)
        try:
            for text in self._stream_llm_answer(
                question, context, settings, history=history
            ):
                yield {"type": "content", "content": text}
        except Exception as e:
            logger.error(f"RAG 生成阶段出错: {e}")
            yield {"type": "error", "content": str(e)}
            return
        sources = self._extract_citations(filtered_docs)
        retrieval_debug = self._extract_retrieval_debug(filtered_docs)
        yield {
            "type": "done",
            "content": "",
            "sources": sources,
            "metadata": {
                "kb_id": kb_id,
                "question": question,
                "retrieval_query": query_for_retrieval,
                "retrieved_chunks": len(filtered_docs),
                "retrieval_debug": retrieval_debug,
                "pipeline_mode": PIPELINE_MODE_FULL,
                "retrieval_mode": retrieval_mode
                or settings.get("retrieval_mode", "vector"),
            },
        }

    def triple_parallel_stream(
        self, kb_id, question, history=None, retrieval_query: str | None = None
    ):
        """
        两阶段并行：
        1) 向量 / 关键词 / 混合三路检索并发执行；
        2) 三路检索完成后，三路生成并发执行。
        SSE 按“分支完成顺序”输出 branch_start → content → branch_done。
        """
        settings = settings_service.get()
        pipeline_started = perf_counter()
        yield {
            "type": "start",
            "content": "",
            "metadata": {"pipeline_mode": PIPELINE_MODE_TRIPLE_PARALLEL},
        }

        retrieval_results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            retrieval_futures = {
                pool.submit(
                    self._run_triple_retrieval_branch,
                    kb_id,
                    retrieval_query or question,
                    settings,
                    b,
                ): b
                for b in _TRIPLE_BRANCH_ORDER
            }
            for fut in as_completed(retrieval_futures):
                result = fut.result()
                retrieval_results[result["branch"]] = result

        generation_results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            generation_futures = {
                pool.submit(
                    self._run_triple_generation_branch,
                    question=question,
                    settings=settings,
                    branch=branch,
                    context=retrieval_results.get(branch, {}).get("context", ""),
                    retrieval_error=retrieval_results.get(branch, {}).get("error"),
                    history=history,
                ): branch
                for branch in _TRIPLE_BRANCH_ORDER
            }
            for fut in as_completed(generation_futures):
                result = fut.result()
                branch = result["branch"]
                generation_results[branch] = result
                retrieval_result = retrieval_results.get(branch, {})
                sources = retrieval_result.get("sources", [])
                yield {"type": "branch_start", "branch": branch}
                yield {"type": "content", "branch": branch, "content": result["answer"]}
                yield {
                    "type": "branch_done",
                    "branch": branch,
                    "sources": sources,
                    "metadata": {
                        "kb_id": kb_id,
                        "question": question,
                        "retrieval_query": retrieval_query or question,
                        "retrieval_mode": branch,
                        "retrieved_chunks": retrieval_result.get("retrieved_chunks", 0),
                        "retrieval_debug": retrieval_result.get("retrieval_debug"),
                        "retrieval_elapsed_ms": retrieval_result.get(
                            "retrieval_elapsed_ms", 0
                        ),
                        "generation_elapsed_ms": result.get("generation_elapsed_ms", 0),
                    },
                }

        triple_payload = {}
        for branch in _TRIPLE_BRANCH_ORDER:
            retrieval_result = retrieval_results.get(branch, {})
            generation_result = generation_results.get(branch, {})
            triple_payload[branch] = {
                "answer": generation_result.get("answer", ""),
                "sources": retrieval_result.get("sources", []),
                "retrieved_chunks": retrieval_result.get("retrieved_chunks", 0),
                "retrieval_debug": retrieval_result.get("retrieval_debug"),
                "retrieval_elapsed_ms": retrieval_result.get("retrieval_elapsed_ms", 0),
                "generation_elapsed_ms": generation_result.get("generation_elapsed_ms", 0),
                "error": retrieval_result.get("error") or generation_result.get("error"),
            }
        yield {
            "type": "done",
            "content": "",
            "sources": None,
            "metadata": {
                "kb_id": kb_id,
                "question": question,
                "retrieval_query": retrieval_query or question,
                "pipeline_mode": PIPELINE_MODE_TRIPLE_PARALLEL,
                "pipeline_elapsed_ms": int((perf_counter() - pipeline_started) * 1000),
                "triple": triple_payload,
            },
        }

    def ask_stream(
        self,
        kb_id,
        question,
        pipeline_mode: str = PIPELINE_MODE_FULL,
        context: str | None = None,
        history=None,
    ):
        """
        统一入口：按 pipeline_mode 分流。
        - full: 检索 + 生成
        - retrieve_only: 仅检索
        - generate_only: 仅生成（使用请求体中的 context，不访问该知识库向量检索）
        """
        retrieval_query = question
        rewrite_enabled_modes = {
            PIPELINE_MODE_FULL,
            PIPELINE_MODE_RETRIEVE_ONLY,
            PIPELINE_MODE_VECTOR_GENERATE,
            PIPELINE_MODE_KEYWORD_GENERATE,
            PIPELINE_MODE_HYBRID_GENERATE,
            PIPELINE_MODE_TRIPLE_PARALLEL,
        }
        if pipeline_mode in rewrite_enabled_modes:
            settings = settings_service.get()
            retrieval_query = self._rewrite_query_from_history(question, history, settings)
            if retrieval_query != question:
                logger.info(f"查询改写生效: {question} -> {retrieval_query}")

        if pipeline_mode == PIPELINE_MODE_RETRIEVE_ONLY:
            yield from self.retrieve_only_stream(
                kb_id, question, retrieval_query=retrieval_query
            )
        elif pipeline_mode == PIPELINE_MODE_GENERATE_ONLY:
            yield from self.generate_stream(question, context or "", history=history)
        elif pipeline_mode == PIPELINE_MODE_VECTOR_GENERATE:
            yield from self.full_rag_stream(
                kb_id,
                question,
                retrieval_mode="vector",
                retrieval_query=retrieval_query,
                history=history,
            )
        elif pipeline_mode == PIPELINE_MODE_KEYWORD_GENERATE:
            yield from self.full_rag_stream(
                kb_id,
                question,
                retrieval_mode="keyword",
                retrieval_query=retrieval_query,
                history=history,
            )
        elif pipeline_mode == PIPELINE_MODE_HYBRID_GENERATE:
            yield from self.full_rag_stream(
                kb_id,
                question,
                retrieval_mode="hybrid",
                retrieval_query=retrieval_query,
                history=history,
            )
        elif pipeline_mode == PIPELINE_MODE_TRIPLE_PARALLEL:
            yield from self.triple_parallel_stream(
                kb_id,
                question,
                history=history,
                retrieval_query=retrieval_query,
            )
        else:
            yield from self.full_rag_stream(
                kb_id, question, retrieval_query=retrieval_query, history=history
            )

    def _extract_citations(self, docs):
        sources = []
        for doc in docs:
            metadata = doc.metadata
            retrieval_type = metadata.get("retrieval_type")
            rerank_score = metadata.get("rerank_score", 0)
            vector_score = metadata.get("vector_score", 0)
            keyword_score = metadata.get("keyword_score", 0)
            rrf_score = metadata.get("rrf_score", 0)
            chunk_id = metadata.get("chunk_id") or metadata.get("id")
            doc_id = metadata.get("doc_id")
            doc_name = metadata.get("doc_name")
            retrieval_rank = metadata.get("retrieval_rank")
            content = doc.page_content
            sources.append(
                {
                    "retrieval_type": retrieval_type,
                    "rerank_score": round(rerank_score * 100, 2),
                    "vector_score": round(vector_score * 100, 2),
                    "keyword_score": round(keyword_score * 100, 2),
                    "rrf_score": round(rrf_score * 100, 2),
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "retrieval_rank": retrieval_rank,
                    "content": content,
                }
            )
        return sources


rag_service = RAGService()
