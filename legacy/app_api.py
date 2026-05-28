import logging
import traceback
from typing import Dict, Any

from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2

import pdf_assistant

knowledge_base = getattr(pdf_assistant, "knowledge_base", None)
storage = getattr(pdf_assistant, "storage", None)
db_url = getattr(pdf_assistant, "db_url", None)

logger = logging.getLogger(__name__)


def _extract_text(obj: Any) -> str:
    try:
        if obj is None:
            return ""
        # plain string
        if isinstance(obj, str):
            return obj

        # dict-like responses
        if isinstance(obj, dict):
            for k in ("text", "content", "answer", "result"):
                if k in obj and isinstance(obj[k], str):
                    return obj[k]

            if "choices" in obj and isinstance(obj["choices"], (list, tuple)) and len(obj["choices"]) > 0:
                first = obj["choices"][0]
                if isinstance(first, dict):
                    if "text" in first:
                        return first["text"]
                    if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                        return first["message"]["content"]

            if "messages" in obj and isinstance(obj["messages"], (list, tuple)):
                parts = []
                for m in obj["messages"]:
                    if isinstance(m, dict):
                        role = m.get("role")
                        content = m.get("content")
                    else:
                        role = getattr(m, "role", None)
                        content = getattr(m, "content", None)
                    if content and role in ("assistant", "tool"):
                        parts.append(str(content))
                if parts:
                    return "\n\n".join(parts)

        # object-like: common attributes
        content = getattr(obj, "content", None)
        if content:
            return str(content)

        msgs = getattr(obj, "messages", None) or getattr(obj, "events", None)
        if msgs and isinstance(msgs, (list, tuple)):
            parts = []
            for m in msgs:
                role = getattr(m, "role", None)
                content = getattr(m, "content", None)
                if content and role in ("assistant", "tool"):
                    parts.append(str(content))
            if parts:
                return "\n\n".join(parts)

    except Exception:
        logger.exception("_extract_text failed")

    # fallback
    try:
        return str(obj)
    except Exception:
        return ""


def index_url(url: str) -> Dict[str, Any]:
    try:
        kb = knowledge_base
        if kb is None:
            # fallback: create a new knowledge base using the project's db_url
            vector_db = PgVector2(collection="dishes", db_url=db_url)
            kb = PDFUrlKnowledgeBase(urls=[url], vector_db=vector_db)
        else:
            # try to append to existing urls list
            if hasattr(kb, "urls") and isinstance(getattr(kb, "urls"), list):
                if url not in kb.urls:
                    kb.urls.append(url)
            else:
                # create a temporary KB that reuses the vector_db if available
                vector_db = getattr(kb, "vector_db", None)
                if vector_db is None:
                    vector_db = PgVector2(collection="dishes", db_url=db_url)
                kb = PDFUrlKnowledgeBase(urls=[url], vector_db=vector_db)

        # Load (upsert=True will insert/update vectors)
        kb.load(upsert=True)
        return {"status": "ok", "url": url}
    except Exception as e:
        logger.exception("index_url failed")
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


def query_text(prompt: str) -> Dict[str, Any]:
    """Run a single-turn query using the project's Agent.
    """
    try:
        from phi.agent import Agent
        from phi.model.groq import Groq

        agent = Agent(
            model=Groq(id="llama-3.3-70b-versatile"),
            user_id="web",
            knowledge=knowledge_base,
            storage=storage,
            show_tool_calls=False,
            search_knowledge=True,
            read_chat_history=False,
        )

        # Try a list of likely single-turn methods.
        candidate_methods = ("ask", "respond", "run", "chat", "complete", "generate", "call", "invoke")
        for m in candidate_methods:
            if hasattr(agent, m):
                fn = getattr(agent, m)
                try:
                    result = fn(prompt)
                    text = _extract_text(result)
                    return {"status": "ok", "result": text}
                except TypeError:
                    try:
                        result = fn({"text": prompt})
                        text = _extract_text(result)
                        return {"status": "ok", "result": text}
                    except Exception:
                        continue

        if callable(agent):
            try:
                out = agent(prompt)
                text = _extract_text(out)
                return {"status": "ok", "result": text}
            except Exception:
                pass

        return {
            "status": "error",
            "error": "Agent instance does not expose a known single-turn API.\n"
            "Check the phi version and adapt app_api.query_text accordingly.",
        }
    except Exception as e:
        logger.exception("query_text failed")
        err_text = str(e)
        tb = traceback.format_exc()
        failed_generation = None
        try:
            if "failed_generation" in err_text:
                start = err_text.find("failed_generation")
                sub = err_text[start:]
                fg_start = sub.find("<function=")
                fg_end = sub.find("</function>")
                if fg_start != -1 and fg_end != -1:
                    failed_generation = sub[fg_start:fg_end + len("</function>")]
                else:
                    failed_generation = sub[:200]
        except Exception:
            failed_generation = None

        try:
            if failed_generation and "search_knowledge_base" in failed_generation:
                import re

                # Try to extract a quoted `query` value from the failed_generation snippet
                query = None
                m = re.search(r'search_knowledge_base\s*\{\s*"query"\s*:\s*"([^"]+)"', failed_generation)
                if m:
                    query = m.group(1)
                else:
                    m2 = re.search(r'search_knowledge_base\s*\{([^}]*)\}', failed_generation)
                    if m2:
                        body = m2.group(1)
                        m3 = re.search(r'"query"\s*:\s*"([^"]+)"', body)
                        if m3:
                            query = m3.group(1)

                if query:
                    kb = knowledge_base
                    results = None
                    if kb is not None:
                        # Try a few common search method names and signatures
                        for method in (
                            "search",
                            "query",
                            "search_documents",
                            "query_documents",
                            "retrieve",
                            "find",
                        ):
                            if hasattr(kb, method):
                                try:
                                    fn = getattr(kb, method)
                                    try:
                                        results = fn(query)
                                    except TypeError:
                                        try:
                                            results = fn({"query": query})
                                        except Exception:
                                            try:
                                                results = fn(query, top_k=4)
                                            except Exception:
                                                results = None
                                    if results:
                                        break
                                except Exception:
                                    # try the next method
                                    continue

                    # Normalize results to a short context string
                    context_text = ""
                    if results:
                        texts = []
                        try:
                            for r in results:
                                t = _extract_text(r)
                                if t:
                                    texts.append(t)
                                if len(texts) >= 4:
                                    break
                        except Exception:
                            t = _extract_text(results)
                            if t:
                                texts = [t]
                        if texts:
                            context_text = "\n\n---\n\n".join(texts)

                    if context_text:
                        # call the model directly without tool-calls, providing manual context
                        try:
                            from phi.agent import Agent
                            from phi.model.groq import Groq

                            agent_no_tools = Agent(
                                model=Groq(id="llama-3.3-70b-versatile"),
                                user_id="web",
                                knowledge=None,
                                storage=storage,
                                show_tool_calls=False,
                                search_knowledge=False,
                                read_chat_history=False,
                            )

                            composite_prompt = (
                                "Use the following documents as context to answer the user's query.\n\n"
                                f"{context_text}\n\nUser query: {prompt}"
                            )

                            out = None
                            for m in ("ask", "respond", "run", "chat", "complete", "generate"):
                                if hasattr(agent_no_tools, m):
                                    try:
                                        fn = getattr(agent_no_tools, m)
                                        try:
                                            result = fn(composite_prompt)
                                        except TypeError:
                                            result = fn({"text": composite_prompt})
                                        out = _extract_text(result)
                                        break
                                    except Exception:
                                        continue

                            if out:
                                return {"status": "ok", "result": out, "note": "returned via local search fallback"}
                        except Exception:
                            logger.exception("fallback local agent failed")
        except Exception:
            logger.exception("local fallback attempt failed")

        resp = {"status": "error", "error": err_text, "traceback": tb}
        if failed_generation:
            resp["failed_generation"] = failed_generation
            resp["hint"] = (
                "The model attempted to call a tool but the tool invocation failed. "
                "I tried a local fallback (manual search + direct model call). "
                "If that did not work, ensure the knowledge-base search tool is registered and functional."
            )
        return resp
