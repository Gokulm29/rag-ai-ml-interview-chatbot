# 🤖 RAG-Based AI/ML Interview Assistant

An AI-powered interview preparation chatbot that helps candidates prepare for **Python, Machine Learning, Deep Learning, Generative AI, NLP, RAG, and AI Agents** through interactive conversations and mock interviews.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from an interview preparation PDF and generate contextual answers using a Large Language Model (LLM).

## 🚀 Live Demo

🔗 **Try the Application:** https://rag-ai-ml-interview-chatbot-27ffo56czmxcgwtezpunrx.streamlit.app/

🔗 **GitHub Repository:** https://github.com/Gokulm29/rag-ai-ml-interview-chatbot

## ✨ Features

- 💬 **AI Chat Mode:** Ask technical questions and receive detailed explanations.
- 🎯 **Mock Interview Mode:** Practice interview questions interactively.
- 📚 **PDF-Based Knowledge Retrieval:** Retrieves relevant information from the interview preparation document.
- 🧠 **RAG Architecture:** Combines document retrieval with LLM-based answer generation.
- 🔎 **Retrieved Sources:** Option to view relevant retrieved content.
- 📝 **Suggested Questions:** Quickly explore common AI/ML interview topics.
- 🌙 **Dark Theme Interface:** Clean, ChatGPT-style conversational interface.
- ☁️ **Cloud Deployment:** Hosted using Streamlit Community Cloud.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| LLM API | Groq |
| Language Model | `openai/gpt-oss-120b` |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Document Processing | PyPDF |
| Environment Configuration | python-dotenv |
| Deployment | Streamlit Community Cloud |
| Version Control | Git and GitHub |

## 🏗️ System Architecture

<img width="920" height="1260" alt="System architecture@1x" src="https://github.com/user-attachments/assets/8075e18d-59bf-4bd8-9b6e-2c29d4d72fa8" />


## 🔄 How It Works

1. **Document ingestion:** Extracts text from the interview preparation PDF.
2. **Text chunking:** Splits extracted text into smaller chunks for efficient retrieval.
3. **Embedding generation:** Converts text chunks into vector representations using sentence embeddings.
4. **Vector indexing:** Stores and searches embeddings using FAISS.
5. **Question processing:** Accepts a user's technical question through the chat interface.
6. **Context retrieval:** Retrieves relevant document chunks based on semantic similarity.
7. **Answer generation:** Sends the question and retrieved context to the Groq-hosted LLM.
8. **Response display:** Displays the generated answer in the Streamlit interface.

## 📂 Project Structure

```text
rag-ai-ml-interview-chatbot/
│
├── data/
│   └── interview_notes.pdf
│
├── app.py
├── rag_engine.py
├── llm.py
├── requirements.txt
├── test_app.py
├── test_llm.py
├── test_rag.py
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Gokulm29/rag-ai-ml-interview-chatbot.git
cd rag-ai-ml-interview-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Replace `your_groq_api_key` with your own API key. **Never commit your API key to GitHub.**

### 5. Run the application

```bash
streamlit run app.py
```

The application will be available at the local URL displayed in your terminal, typically `http://localhost:8501`.

## 🔐 Deployment Configuration

For Streamlit Community Cloud:

1. Connect your GitHub repository to Streamlit Community Cloud.
2. Configure the application secrets in the Streamlit dashboard.
3. Add the required Groq API key and model configuration.
4. Deploy the application and verify the chat and mock interview features.

Example secrets configuration:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Keep your actual credentials private.

## 🧪 Testing

The repository includes test files for the application, LLM integration, and RAG functionality.

Run the tests using:

```bash
pytest
```

## 🎯 Use Cases

- Preparing for AI/ML technical interviews.
- Revising Python and machine learning fundamentals.
- Understanding Generative AI and RAG concepts.
- Practicing technical interview questions.
- Reviewing concepts using a document-based AI assistant.

## 🔮 Future Enhancements

- Support for multiple PDFs and document formats.
- Resume-based personalized interview questions.
- Difficulty levels: Beginner, Intermediate, and Advanced.
- Interview performance scoring and feedback.
- Conversation history and progress tracking.
- Voice-based mock interview practice.
- More detailed source citations and answer evaluation.

## 👨‍💻 Author

**Gokul M**

GitHub: https://github.com/Gokulm29

---

⭐ If you find this project useful, consider giving the repository a star!
