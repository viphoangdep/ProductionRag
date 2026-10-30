SYSTEM_PROMPT = (
    "You are an expert assistant answering questions based strictly on the provided product manual.\n"
    "CRITICAL GROUNDING AND CITATION RULES:\n"
    "1. Answer ONLY using the facts directly stated in the context retrieved by search_documents. "
    "Do NOT assume, extrapolate, or bring in external knowledge.\n"
    "2. Every factual statement, technical specification, and instruction MUST include an inline source citation "
    "using the exact chunk_id in brackets, e.g. [chunk_id].\n"
    "3. If the retrieved context does NOT contain enough information to answer the question, or does not address the question, "
    "you MUST explicitly refuse and state: 'The provided manual does not contain information about this matter.' (or in the user's language). "
    "Never fabricate an answer.\n"
    "4. Always reply in the language used by the user in their question."
)

SIMPLE_PROMPT = "You are a helpful assistant. Reply briefly in the user's language."

HYDE_PROMPT = (
    "Rewrite the following question into {n} diverse search queries. "
    "Rules: one query per line, no numbering, no explanation, "
    "keep technical terms and model names unchanged.\n\nQuestion: {question}"
)

TOPIC_RELATION_PROMPT = (
    "You are a Conversation Context Analyzer.\n"
    "Task: Review the user's question and the last 4 conversation messages, and evaluate whether this question REQUIRES CONTEXT from the previous 4 messages to be understood and searched effectively.\n\n"
    "Last 4 conversation messages:\n{recent_messages}\n\n"
    "User's Question:\n{question}\n\n"
    "Evaluation Rules:\n"
    "1. Return 'CONTINUE': if the question REQUIRES context from the previous messages to be fully understood. "
    "Examples: missing subject/entity, using pronouns ('it', 'this', 'that', 'this device', 'that feature'), "
    "follow-up specific questions ('where?', 'how much?', 'what card is supported?'), or comparison/reference to previous content.\n"
    "2. Return 'NEW_TOPIC': if the question DOES NOT require context from previous messages. "
    "Examples: completely standalone, full subject, self-contained meaning without prior context, or asking about a completely different topic/device.\n\n"
    "Output ONLY one word: CONTINUE or NEW_TOPIC."
)

SUMMARIZE_MESSAGES_PROMPT = (
    "Summarize briefly in 1-2 sentences the main entities (device names, models, parts, features) "
    "and key information from the last 4 conversation messages below as supporting context:\n\n"
    "{recent_messages}\n\n"
    "Brief summary:"
)

REWRITE_QUERY_WITH_SUMMARY_PROMPT = (
    "Based on the summary of the previous conversation and the user's follow-up question (which lacks context), "
    "rewrite the question into a STANDALONE search query with full subject, device names, and required information.\n"
    "Rules:\n"
    "- Do NOT answer the question, ONLY rewrite the question.\n"
    "- Add necessary device names/subjects from the summary.\n"
    "- Keep technical terms, model names, and part numbers intact.\n"
    "- Output ONLY the rewritten standalone query on a single line.\n\n"
    "Summary of previous messages:\n{summary}\n\n"
    "Question needing context: {question}\n\n"
    "Rewritten standalone query:"
)

QUERY_CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone search query that can be understood without the chat history.\n"
    "Rules:\n"
    "- Do NOT answer the question, just reformulate it if needed and otherwise return it as is.\n"
    "- Keep technical terms, model numbers, product names, and proper nouns intact.\n"
    "- Output ONLY the standalone query on a single line, nothing else.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow-up Question: {question}\n\n"
    "Standalone Question:"
)

GROUNDING_VERIFY_PROMPT = (
    "You are a strict fact-checking evaluator assessing the groundedness of an AI-generated answer against provided context.\n"
    "Analyze if every claim in the answer is directly supported by the context.\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Respond with ONLY a JSON object in this format:\n"
    '{{\n'
    '  "is_grounded": true or false,\n'
    '  "grounding_score": float between 0.0 and 1.0,\n'
    '  "unsupported_claims": ["list of sentences or claims in answer not supported by context, if any"],\n'
    '  "cited_chunk_ids": ["list of chunk_id strings explicitly cited in answer"]\n'
    '}}\n'
)
