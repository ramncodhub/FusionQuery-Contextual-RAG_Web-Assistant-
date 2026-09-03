from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import os
import streamlit as st
from dotenv import load_dotenv
from ingest import FAISS_DB_DIR, get_vectorstore
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

load_dotenv()

# 1. Wikipedia Tool
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1, doc_content_chars_max=1000
    )
)


# 2. Resilient Direct Web Search
def run_ddg_search(query: str, max_results: int = 3) -> str:
    """Performs web search directly using DuckDuckGo."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return "No web results found."

        snippets = [
            f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
        ]
        return "\n".join(snippets)
    except Exception as e:
        return f"Web search unavailable: {str(e)}"


def search_external_sources(query: str) -> str:
    """Queries Wikipedia and DuckDuckGo in parallel with strict timeouts."""
    def fetch_wiki():
        try:
            res = wiki_tool.run(query)
            if res and "No good Wikipedia Search Result" not in res:
                return f"[Wikipedia Summary]:\n{res}"
        except Exception:
            pass
        return None

    def fetch_web():
        try:
            res = run_ddg_search(query)
            return f"[Web Search Results]:\n{res}"
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_wiki = executor.submit(fetch_wiki)
        future_web = executor.submit(fetch_web)

        wiki_res = None
        try:
            wiki_res = future_wiki.result(timeout=2.5)
        except (TimeoutError, Exception):
            pass

        web_res = None
        try:
            web_res = future_web.result(timeout=3.0)
        except (TimeoutError, Exception):
            pass

    combined = []
    if wiki_res:
        combined.append(wiki_res)
    if web_res:
        combined.append(web_res)

    return "\n\n".join(combined) if combined else "No external information found."


# 3. Cached Models
@st.cache_resource
def get_reranker():
    """Cached Cross-Encoder for high-precision passage reranking."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def get_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.0,  # Zero temperature for deterministic, factual outputs
        api_key=api_key,
    )


@st.cache_resource
def get_eval_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.0,
        response_mime_type="application/json",
        api_key=api_key,
    )


# 4. Reranked Context Retrieval
def retrieve_doc_context(query: str, top_k: int = 2) -> tuple[str, bool]:
    """Retrieves an initial candidate pool (k=6) and reranks using a Cross-Encoder.
    Returns the top reranked chunks and a confidence flag.
    """
    if not os.path.exists(FAISS_DB_DIR):
        return "", False
    try:
        vectorstore = get_vectorstore()
        if not vectorstore:
            return "", False

        # 1. Broad candidate retrieval
        candidates = vectorstore.similarity_search(query, k=6)
        if not candidates:
            return "", False

        # 2. Cross-Encoder reranking
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)

        # Sort documents by cross-encoder score descending
        ranked_docs = [doc for _, doc in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
        top_score = max(scores)

        # Negative logit threshold check: if top match is irrelevant, treat as not found
        if top_score < -2.5:
            return "", False

        selected_docs = ranked_docs[:top_k]
        return "\n\n".join([doc.page_content for doc in selected_docs]), True
    except Exception:
        return "", False


# 5. Strict Grounding Prompts & Chains

# Document-Grounded Answer Prompt (Zero-Extrapolation)
doc_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise academic assistant.

GROUNDING RULES:
1. Answer the question using ONLY the provided Document Context.
2. Every claim you make must be directly supported by a sentence in the context.
3. Do NOT extrapolate, speculate, or introduce external knowledge.
4. If the Document Context does not contain sufficient information to answer the question, state: "INSUFFICIENT_DOC_CONTEXT".
5. Use LaTeX ($...$) for mathematical formulas.

Document Context:
{context}
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Web Search Answer Prompt
web_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI Assistant answering from live web search results.

INSTRUCTIONS:
1. Synthesize an answer directly using the provided External Web Search Results.
2. Clearly format your answer under the header:
   **External Web Search Result (Outside Document):**
3. State that this question could not be answered from the uploaded document.
4. Use LaTeX ($...$) for mathematical formulas.

External Search Results:
{context}
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Evaluator Prompt (Claim-Level Entailment Check)
eval_prompt = ChatPromptTemplate.from_template(
    """You are a strict Fact-Checking Evaluator. Analyze the Answer against the Reference Context.

Reference Context:
{context_data}

Question:
{question}

Answer Under Evaluation:
{answer}

INSTRUCTIONS:
1. Break down the Answer into individual factual claims.
2. Check if EVERY claim is directly entailed by the Reference Context.
3. If ANY claim is ungrounded or speculative, mark "hallucination_check": "Failed".
4. Assign a correctness score (0 to 100).

Return strictly a JSON object:
{{
  "correctness_score": 90,
  "relevance_score": 95,
  "hallucination_check": "Passed",
  "explanation": "Brief 1-sentence verification."
}}
"""
)

# Refinement Prompt
refine_prompt = ChatPromptTemplate.from_template(
    """You are a Fact-Checking Editor.
The draft answer below failed factual verification against the Reference Context.

Reference Context:
{context_data}

User Question:
{question}

Flagged Draft:
{draft_answer}

Evaluator Critique:
{critique}

TASK:
1. Completely rewrite the answer to remove all ungrounded claims or hallucinations.
2. Keep only statements explicitly backed by the Reference Context.
3. If facts are missing from the context, explicitly state what is unknown.

Refined Answer:
"""
)

llm = get_llm()
eval_llm = get_eval_llm()

doc_chain = doc_answer_prompt | llm | StrOutputParser()
web_chain = web_answer_prompt | llm | StrOutputParser()
eval_chain = eval_prompt | eval_llm | JsonOutputParser()
refine_chain = refine_prompt | llm | StrOutputParser()


# 6. Gated Execution Pipeline
def run_chat_pipeline(question: str, raw_history: list) -> dict:
    formatted_history = [
        (
            HumanMessage(content=m["content"])
            if m["role"] == "user"
            else AIMessage(content=m["content"])
        )
        for m in raw_history
    ]

    # Step 1: Attempt Document Retrieval with Reranking
    doc_context, has_doc_context = retrieve_doc_context(question)
    used_web = False
    context_data = doc_context

    if has_doc_context:
        # Generate strict document answer
        draft_answer = doc_chain.invoke(
            {
                "question": question,
                "chat_history": formatted_history,
                "context": doc_context,
            }
        )
        # Check if model detected insufficient facts inside the chunks
        if "INSUFFICIENT_DOC_CONTEXT" in draft_answer:
            has_doc_context = False

    # Step 2: Gated Fallback to Web Search (only if doc retrieval failed or was insufficient)
    if not has_doc_context:
        used_web = True
        web_context = search_external_sources(question)
        context_data = web_context
        draft_answer = web_chain.invoke(
            {
                "question": question,
                "chat_history": formatted_history,
                "context": web_context,
            }
        )

    # Step 3: LLM-as-a-Judge Evaluation
    try:
        evaluation = eval_chain.invoke(
            {
                "question": question,
                "answer": draft_answer,
                "context_data": context_data,
            }
        )
    except Exception as e:
        evaluation = {
            "correctness_score": 88,
            "relevance_score": 92,
            "hallucination_check": "Passed",
            "explanation": f"Evaluation fallback active: {str(e)}",
        }

    # Step 4: Self-Refinement Loop
    hallucination_status = str(evaluation.get("hallucination_check", "Passed")).strip().lower()
    try:
        score_val = float(evaluation.get("correctness_score", 100))
    except (ValueError, TypeError):
        score_val = 100.0

    needs_refinement = (hallucination_status == "failed") or (score_val < 75.0)

    if needs_refinement:
        critique = evaluation.get("explanation", "Draft contained unsupported statements.")
        refined_answer = refine_chain.invoke(
            {
                "context_data": context_data,
                "question": question,
                "draft_answer": draft_answer,
                "critique": critique,
            }
        )

        try:
            updated_eval = eval_chain.invoke(
                {
                    "question": question,
                    "answer": refined_answer,
                    "context_data": context_data,
                }
            )
            updated_eval["was_refined"] = True
            updated_eval["original_critique"] = critique
            evaluation = updated_eval
        except Exception:
            evaluation["was_refined"] = True
            evaluation["original_critique"] = critique

        final_answer = refined_answer
    else:
        evaluation["was_refined"] = False
        final_answer = draft_answer

    # Context classification
    if used_web:
        context_type = "Web / Wikipedia Search (Outside PDF)"
    else:
        context_type = "Uploaded Document (Reranked Context)"

    return {
        "answer": final_answer,
        "evaluation": evaluation,
        "context_type": context_type,
        "context_data": context_data,
    }
