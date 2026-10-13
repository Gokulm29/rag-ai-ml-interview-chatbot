
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "interview_notes.pdf"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K = 3


# ==========================================
# RAG ENGINE
# ==========================================

class RAGEngine:

    def __init__(self):
        print("\n[1/4] Loading embedding model...", flush=True)

        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = []
        self.index = None

        print("Embedding model loaded successfully!", flush=True)

    # --------------------------------------
    # STEP 1: EXTRACT PDF TEXT
    # --------------------------------------

    def load_pdf(self, pdf_path):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"\nPDF not found: {pdf_path}\n"
                "Place your PDF inside the backend/data folder "
                "and name it interview_notes.pdf."
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("The input file must be a PDF.")

        print(f"\n[2/4] Reading PDF: {pdf_path.name}", flush=True)

        reader = PdfReader(str(pdf_path))
        documents = []

        for page_number, page in enumerate(
            reader.pages, start=1
        ):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()

            if text:
                documents.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                })

        if not documents:
            raise ValueError(
                "No readable text found in this PDF. "
                "It may be scanned or image-only."
            )

        print(f"PDF pages: {len(reader.pages)}", flush=True)
        print(f"Pages containing text: {len(documents)}", flush=True)

        return documents

    # --------------------------------------
    # STEP 2: SPLIT TEXT INTO CHUNKS
    # --------------------------------------

    def create_chunks(self, documents):

        print("\n[3/4] Creating text chunks...", flush=True)

        chunks = []

        for document in documents:
            words = document["text"].split()
            start = 0

            while start < len(words):

                end = min(start + CHUNK_SIZE, len(words))

                chunk_text = " ".join(words[start:end])

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "source": document["source"],
                        "page": document["page"],
                    })

                if end >= len(words):
                    break

                start = end - CHUNK_OVERLAP

        if not chunks:
            raise ValueError("No text chunks could be created.")

        print(f"Total chunks created: {len(chunks)}", flush=True)

        return chunks

    # --------------------------------------
    # STEP 3: CREATE EMBEDDINGS AND FAISS INDEX
    # --------------------------------------

    def build_index(self, pdf_path):

        documents = self.load_pdf(pdf_path)
        self.chunks = self.create_chunks(documents)

        print("\nGenerating embeddings...", flush=True)

        texts = [
            chunk["text"] for chunk in self.chunks
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings, dtype=np.float32
        )

        dimension = embeddings.shape[1]

        # Inner product on normalized vectors
        # corresponds to cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print("\nFAISS index created successfully!", flush=True)
        print(f"Embedding dimensions: {dimension}", flush=True)
        print(f"Indexed chunks: {self.index.ntotal}", flush=True)

    # --------------------------------------
    # STEP 4: RETRIEVE RELEVANT CHUNKS
    # --------------------------------------

    def retrieve(self, question, top_k=TOP_K):

        if self.index is None or not self.chunks:
            raise RuntimeError(
                "Index is empty. Build the index first."
            )

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding, dtype=np.float32
        )

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            query_embedding, k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            chunk = self.chunks[int(idx)]

            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score),
            })

        return results


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("=" * 55, flush=True)
    print("       AI/ML INTERVIEW RAG ENGINE", flush=True)
    print("=" * 55, flush=True)

    try:
        engine = RAGEngine()

        engine.build_index(PDF_PATH)

        print("\n" + "=" * 55, flush=True)
        print("RAG ENGINE READY!", flush=True)
        print("Type 'exit' to stop.", flush=True)
        print("=" * 55, flush=True)

        while True:

            question = input(
                "\nEnter your interview question: "
            ).strip()

            if question.lower() in {"exit", "quit"}:
                print("RAG engine stopped. Goodbye!", flush=True)
                break

            if not question:
                print("Please enter a question.", flush=True)
                continue

            try:
                results = engine.retrieve(question)

                print("\n--- RETRIEVED CONTEXT ---", flush=True)

                for number, result in enumerate(
                    results, start=1
                ):
                    print(f"\nResult {number}", flush=True)
                    print(
                        f"Source: {result['source']}",
                        flush=True,
                    )
                    print(
                        f"Page: {result['page']}",
                        flush=True,
                    )
                    print(
                        f"Similarity: {result['score']:.4f}",
                        flush=True,
                    )
                    print(
                        f"Text: {result['text']}",
                        flush=True,
                    )
                    print("-" * 55, flush=True)

            except Exception as error:
                print(f"Retrieval error: {error}", flush=True)

    except KeyboardInterrupt:
        print("\nProgram interrupted.", flush=True)

    except Exception as error:
        print(
            f"\nERROR: {type(error).__name__}: {error}",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
