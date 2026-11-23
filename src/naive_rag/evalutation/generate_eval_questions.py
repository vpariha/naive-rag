import time

from langchain_core.prompts import ChatPromptTemplate

from naive_rag.model.model import get_chat_model


def generate_eval_question(context: str, question_count: str):

    system_prompt = ChatPromptTemplate.from_template("""
    SYSTEM PROMPT: RAG Evaluation Dataset Generator

    Role

    Generate high-quality factual Q&A datasets from supplied document chunks for evaluating RAG retrieval (Recall@K, Precision@K, MRR, NDCG@K).

    Example input
    Chunk_id: chunk_757585sdasdas78
    Context: Some context
    Source: Some source

    Instructions

    Generate at least {question_count} unique, diverse, factual questions using only the supplied context.
    Cover different topics and chunks. Include numerical, procedural, policy, eligibility, and comparison questions when supported.
    Provide concise, accurate answers without assumptions or external knowledge.
    Evaluate EVERY chunk independently for EVERY question.
    Relevance Scoring

    3: Directly answers the question.
    2: Provides meaningful supporting information.
    1: Provides limited relevant context.
    0: Irrelevant.

    Rules:

    Every question must have at least one score of 3.
    Score semantic and factual relevance, not keyword overlap.
    relevant_chunks must contain exactly chunks scored 2 or 3.
    relevance_scores must include all chunks scored above 0 and exclude score 0.
    Output

    Return ONLY valid JSON as an array:

    [ {{ "query": "Question", "answer": "Concise factual answer", "relevant_chunks": ["chunk_id_1"], "relevance_scores": {{ "chunk_id_1": 3, "chunk_id_2": 1 }} }} ]

    Validation

    Before responding, ensure:

    Questions are unique, answerable, and factually supported.
    Every question has at least one score of 3.
    Every supplied chunk is evaluated for every question.
    Chunk IDs match the input exactly.
    Relevant chunks and scores follow the scoring rules.
    No information is fabricated.
    Output is valid JSON.
    Accuracy takes priority over quantity. If insufficient information exists, return fewer valid questions rather than fabricate data.

    ##Context:
    {context}

    """)

    prompt = system_prompt.invoke(
        {"context": context, "question_count": question_count}
    )

    model = get_chat_model()

    start_time = time.perf_counter()
    result = model.invoke(prompt)
    elapsed_time = time.perf_counter() - start_time

    print(f"model.invoke took {elapsed_time:.2f} seconds")

    with open("evaluation.json", "a", encoding="utf-8") as file:
        file.write(result.content)
        file.write("\n")
        file.write(
            "===================================================================================\n"
        )
        print("batch finished")
