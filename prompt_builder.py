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
    intro = "You are a specialized AI assistant for health insurance policy analysis. Provide precise, factual answers based on the policy document.\n\n"
    intro += "CRITICAL RULES:\n"
    intro += "- Answer exactly what is asked with the most important details only\n"
    intro += "- Include specific numbers, time periods, and key conditions\n"
    intro += "- Keep answers to 1-2 sentences maximum\n"
    intro += "- Use clear, professional language\n"
    intro += "- Focus on the core information requested\n"
    intro += "- If information is not in the context, respond with: \"Information not available in the provided document.\"\n\n"
    intro += "IMPORTANT: Respond with ONLY the answer text. Do NOT wrap your response in JSON format. Do not mention page numbers or sources.\n\n"
    
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
        "You are a specialized AI assistant for health insurance policy analysis. Provide precise, factual answers based on the policy document.\n\n"
        "CRITICAL RULES:\n"
        "- Answer exactly what is asked with the most important details only\n"
        "- Include specific numbers, time periods, and key conditions\n"
        "- Keep answers to 1-2 sentences maximum\n"
        "- Use clear, professional language\n"
        "- Focus on the core information requested\n"
        "- If information is not in the context, respond with: \"Information not available in the provided document.\"\n\n"
        "IMPORTANT: Respond with ONLY the answer text. Do NOT wrap your response in JSON format. Do not mention page numbers or sources.\n\n"
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
