import hashlib
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownTextSplitter

from naive_rag.persistance.vector_store import get_vectore_store


class IngestionService:
    def __init__(self):
        self.vector_store = get_vectore_store()

    def ingest_data_using_dolcing(self, path: str):

        documents: list[Document] = []

        for pdf_path in sorted(Path(path).rglob("*.pdf")):
            document_id, document_hash = self.generate_source_document_id(pdf_path)
            loader = DoclingLoader(
                file_path=[str(pdf_path)],
                export_type=ExportType.MARKDOWN,
            )
            loaded_documents = loader.load()

            for document in loaded_documents:
                document.metadata.update(
                    {
                        "document_id": document_id,
                        "document_hash": document_hash,
                    }
                )

        documents.extend(loaded_documents)

        chunks = self.convert_documents_to_chunks(documents)
        self.save_chunks_in_batch(chunks)

    def ingest(self, path: str):
        documents = self.generate_document_ids(self.load_directory(path))
        chunks = self.convert_documents_to_chunks(documents)
        self.save_chunks_in_batch(chunks)

    def save_chunks_in_batch(self, chunks: list[Document]) -> None:
        for start in range(0, len(chunks), 10):
            batch = chunks[start : start + 10]
            ids = [chunk.metadata["chunk_id"] for chunk in batch]
            self.vector_store.add_documents(batch, ids=ids)
            # context = ""
            # for chunk in chunks[start : start + 10]:
            #     context += f"Chunk_id:{chunk.metadata['chunk_id']} \nContent: {chunk.page_content} \nSource: {chunk.metadata['source']} \n\n"

            # generate_eval_question(context=context, question_count=8)
        print("Batch chunking finished successfully")

    @staticmethod
    def load_directory(path: str) -> list[Document]:
        if not Path(path).exists():
            raise ValueError(f"Path {path} does not exist.")

        dir_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        documents = dir_loader.load()
        return documents

    @staticmethod
    def convert_documents_to_chunks(documents: list[Document]) -> list[Document]:
        text_splitter = MarkdownTextSplitter()
        chunks = text_splitter.split_documents(documents)
        positions: dict[str, int] = {}

        for chunk in chunks:
            document_id = chunk.metadata.get("document_id")
            chunk_index = positions.get(document_id, 0)
            positions[document_id] = chunk_index + 1

            chunk_id, chunk_hash = IngestionService.generate_chunk_ids(
                document_id, chunk.page_content, chunk_index
            )
            chunk.metadata.update({"chunk_id": chunk_id, "chunk_hash": chunk_hash})

        return chunks

    @staticmethod
    def generate_chunk_ids(
        document_id: str,
        chunk_text: str,
        chunk_index: int,
    ) -> tuple[str, str]:
        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        raw_id = "\0".join(
            (
                document_id,
                "markdown-splitter-v1",
                str(chunk_index),
                chunk_hash,
            )
        )
        chunk_id = "chunk_" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return chunk_id, chunk_hash

    @staticmethod
    def generate_document_ids(documents: list[Document]) -> list[Document]:
        for document in documents:
            source = str(document.metadata.get("source", ""))
            page = str(document.metadata.get("page", ""))

            document_hash = hashlib.sha256(
                document.page_content.encode("utf-8")
            ).hexdigest()

            identity = f"{source}\0{page}\0{document_hash}".encode()
            document_id = "document_" + hashlib.sha256(identity).hexdigest()

            document.metadata.update(
                {
                    "document_id": document_id,
                    "document_hash": document_hash,
                }
            )

        return documents

    @staticmethod
    def generate_source_document_id(pdf_path: Path) -> tuple[str, str]:
        digest = hashlib.sha256()
        with pdf_path.open("rb") as pdf:
            for block in iter(lambda: pdf.read(1024 * 1024), b""):
                digest.update(block)

        file_hash = digest.hexdigest()
        source_key = pdf_path.resolve().as_posix()
        identity = f"{source_key}\0{file_hash}".encode()
        document_id = "document_" + hashlib.sha256(identity).hexdigest()

        return document_id, file_hash
