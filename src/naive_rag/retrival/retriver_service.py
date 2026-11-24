from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from naive_rag.config.setting import AppConfig
from naive_rag.model.model import get_chat_model
from naive_rag.persistance.vector_store import get_vectore_store


class RetriverService:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.vector_store = get_vectore_store()
        self.chat_model = get_chat_model()

    def format_docs_with_metadata(self, documents: list[Document]) -> str:
        context = ""
        for doc in documents:
            source = doc.metadata.get("source", "Unknown source")
            context += f"Content: {doc.page_content} \nSource: {source} \n\n\n"
        return context

    def retrieve_data(self, query: str):

        self.prompt = ChatPromptTemplate.from_template(
            """
                ## Role
                You are a precise, reliable AI assistant. Your task is to answer user questions accurately based strictly on the provided context with citation.

                ## Instructions
                1. Grounding: Answer the question using ONLY the facts and data present in the Context section below. Do not assume, extrapolate, or use outside knowledge.
                2. Handling Missing Info: If the context does not contain enough information to answer the question, explicitly state: "I cannot answer this based on the provided documents." Do not guess or fabricate details.
                3. Citations: Cite the specific source for each factual claim using the format [Source] provided in the context.
                4. Tone & Style: Be concise, objective, and professional.

                ## Context
                {context}

                ## Question
                {query}
            """
        )

        retrieve = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": self.app_config.TOP_K}
        )
        chain = (
            {
                "context": retrieve | self.format_docs_with_metadata,
                "query": RunnablePassthrough(),
            }
            | self.prompt
            | self.chat_model
            | StrOutputParser()
        )
        # for chunk in chain.stream(query):
        #     print(chunk, end="", flush=True)
        return chain.stream(query)
