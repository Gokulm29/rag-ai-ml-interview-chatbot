
from rag_engine import RAGEngine, PDF_PATH
from llm import generate_answer


def main():
    print("Initializing RAG engine...")

    engine = RAGEngine()
    engine.build_index(PDF_PATH)

    print("\nRAG + LLM chatbot is ready!")
    print("Type 'exit' to stop.")

    while True:
        question = input(
            "\nEnter your interview question: "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            chunks = engine.retrieve(question)

            print("\nGenerating answer...\n")

            answer = generate_answer(question, chunks)

            print("=" * 60)
            print("INTERVIEW ANSWER")
            print("=" * 60)
            print(answer)

            print("\nSOURCES")

            for chunk in chunks:
                print(
                    f"- {chunk['source']}, "
                    f"page {chunk['page']}"
                )

            print("=" * 60)

        except Exception as error:
            print(f"Error: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
