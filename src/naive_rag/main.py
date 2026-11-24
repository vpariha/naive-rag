import gradio as gr
from dotenv import load_dotenv

from naive_rag.config.setting import app_config
from naive_rag.ingestion.ingestion_service import IngestionService
from naive_rag.retrival.retriver_service import RetriverService

load_dotenv()


def ingest_data():
    ingestion_service = IngestionService()
    # ingestion_service.ingest("src/naive_rag/source")
    ingestion_service.ingest_data_using_dolcing("src/naive_rag/source")


def retrieve_data():
    query = "Which prompting technique was found most effective for achieving correctness in Java code refactoring, and by what percentage did it improve test pass rates compared to Zero-Shot?"
    retrival_service = RetriverService(app_config=app_config)
    retrival_service.retrieve_data(query=query)


def main() -> None:

    # ingest_data()
    # retrieve_data()
    retriever = RetriverService(app_config=app_config)

    def respond(message: str, _history):
        stream = retriever.retrieve_data(query=message)
        response_content = ""
        for chunk in stream:
            response_content += chunk
            yield response_content

    gr.ChatInterface(
        fn=respond,
        title="Naive RAG - Ask a question",
    ).launch()


if __name__ == "__main__":
    main()
