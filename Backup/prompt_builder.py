from retriever_reranker import rerank_chunks, retrieve_top_k


def build_prompt_with_sources(query, context_chunks, max_chars=4000):
    """
    Builds an efficient prompt with source-attributed context.

    Args:
        query (str): User question
        context_chunks (list): List of dicts with 'text' and 'page'
        max_chars (int): Max allowed characters for context (e.g., 4000 for GPT)

    Returns:
        prompt (str): Final prompt string for LLM
        used_sources (list): Chunks actually used
    """
    intro = "You are an expert assistant. Answer the question using only the provided context.\n"
    intro += "Cite the page number in parentheses like (Page 3). Be accurate.\n\n"
    
    context_header = "### Context:\n"
    prompt_header = "\n### Question:\n"
    
    context_str = ""
    used_sources = []
    total_chars = 0

    for chunk in context_chunks:
        page = chunk.get("page", "?")
        text = chunk["text"].strip().replace("\n", " ")
        source_block = f"(Page {page}) {text}\n\n"
        block_len = len(source_block)

        if total_chars + block_len > max_chars:
            break

        context_str += source_block
        used_sources.append(chunk)
        total_chars += block_len

    prompt = intro + context_header + context_str + prompt_header + query.strip()
    return prompt, used_sources


def build_prompt_without_sources(query, context_chunks, max_chars=6000):
    """
    Builds a clean prompt WITHOUT citations for Gemini.

    Args:
        query (str): User's question
        context_chunks (list): List of dicts with 'text'
        max_chars (int): Max total character limit

    Returns:
        str: Final prompt for LLM
    """
    intro = (
        "You are a helpful assistant. Answer the question as clearly and concisely as possible "
        "using only the provided context. Do not mention any page numbers or sources. Just provide a natural, accurate answer.\n\n"
    )

    context_header = "### Context:\n"
    prompt_header = "\n### Question:\n"

    context_str = ""
    total_chars = 0
    for chunk in context_chunks:
        text = chunk["text"].strip().replace("\n", " ")
        block = f"{text}\n\n"
        if total_chars + len(block) > max_chars:
            break
        context_str += block
        total_chars += len(block)

    return intro + context_header + context_str + prompt_header + query.strip()

# query = "Is icu treatment covered under this policy and are there any limits?"
# reranked_chunks = rerank_chunks(query, retrieve_top_k(query, k=10), top_n=5)

# prompt, sources_used = build_prompt_with_sources(query, reranked_chunks)

# print("📄 Final Prompt:\n")
# print(prompt)

# print("\n🔗 Sources Used:")
# for src in sources_used:
#     print(f"Page {src['page']} | Score: {src['rerank_score']:.4f}")
