# 🤖 RAG-Based AI/ML Interview Assistant

An AI-powered interview preparation chatbot that helps candidates prepare for **Python, Machine Learning, Deep Learning, Generative AI, NLP, RAG, and AI Agents** through interactive conversations and mock interviews.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from an interview preparation PDF and generate contextual answers using a Large Language Model (LLM).

## 🚀 Live Demo

🔗 **Try the Application:** https://rag-ai-ml-interview-chatbot-27ffo56czmxcgwtezpunrx.streamlit.app/

🔗 **GitHub Repository:** https://github.com/Gokulm29/rag-ai-ml-interview-chatbot

## ✨ Features

- 💬 **AI Chat Mode:** Ask technical questions and receive detailed explanations.
- 🎯 **Mock Interview Mode:** Practice interview questions interactively.
- 🔐 **User Authentication:** Secure login using email/password and Google OAuth.
- 🔑 **Google Login:** Sign in easily using a Google account.
- 📚 **PDF-Based Knowledge Retrieval:** Retrieves relevant information from the interview preparation document.
- 🧠 **RAG Architecture:** Combines document retrieval with LLM-based answer generation.
- 🔎 **Retrieved Sources:** Option to view relevant retrieved content.
- 📝 **Suggested Questions:** Quickly explore common AI/ML interview topics.
- 💾 **Chat History:** Stores user conversations and allows users to access previous chat history.
- 👤 **User Management:** Stores user details securely in MongoDB Atlas.
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
| Database | MongoDB Atlas |
| Authentication | Email/Password, Google OAuth |
| Environment Configuration | python-dotenv |
| Deployment | Streamlit Community Cloud |
| Version Control | Git and GitHub |

## 🏗️ System Architecture

<img width="920" height="1260" alt="System architecture@1x" src="https://github.com/user-attachments/assets/8075e18d-59bf-4bd8-9b6e-2c29d4d72fa8" />

## 🔄 How It Works

1. **User Authentication:** Users log in using email/password or Google OAuth.
2. **Document Ingestion:** Extracts text from the interview preparation PDF.
3. **Text Chunking:** Splits extracted text into smaller chunks for efficient retrieval.
4. **Embedding Generation:** Converts text chunks into vector representations using sentence embeddings.
5. **Vector Indexing:** Stores and searches embeddings using FAISS.
6. **Question Processing:** Accepts a user's technical question through the chat interface.
7. **Context Retrieval:** Retrieves relevant document chunks based on semantic similarity.
8. **Answer Generation:** Sends the question and retrieved context to the Groq-hosted LLM.
9. **Chat History Storage:** Saves user conversations and chat history in MongoDB Atlas.
10. **Response Display:** Displays the generated answer in the Streamlit interface.

## 🗄️ Database Integration

The application uses **MongoDB Atlas** as its cloud database.

Key functionalities:

- Stores registered user details.
- Maintains individual user chat history.
- Stores conversation data for future access.
- Supports persistent chat history across sessions.

## 🔐 Authentication

The application provides two authentication methods:

- **Email and Password Authentication:** Users can register and log in using their email address and password.
- **Google OAuth Authentication:** Users can authenticate using their Google accounts.

Authentication helps provide personalized access to the chatbot and user-specific conversation history.

## 📂 Project Structure

```text
## 📂 Project Structure
rag-ai-ml-interview-chatbot/
│
├── .devcontainer/          # Development container configuration
│
├── .streamlit/             # Streamlit configuration
│
├── data/                   # Interview preparation PDF
│
├── .gitignore              # Git ignored files
│
├── README.md               # Project documentation
│
├── app.py                  # Main Streamlit application
├── app_backup.py           # Backup application file
│
├── database.py             # MongoDB Atlas integration
├── database_backup.py      # Backup database file
│
├── llm.py                  # LLM integration using Groq
│
├── rag_engine.py           # RAG pipeline and FAISS retrieval
│
├── requirements.txt        # Project dependencies
│
├── test_app.py             # Application tests
├── test_database.py        # Database tests
├── test_llm.py             # LLM tests
└── test_rag.py             # RAG functionality tests
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
MONGODB_URI=your_mongodb_connection_string
```

Configure the required Google OAuth credentials according to your authentication implementation.

Replace the placeholder values with your actual credentials.

**Never commit API keys, database credentials, or OAuth secrets to GitHub.**

### 5. Run the application

```bash
streamlit run app.py
```

The application will be available at the local URL displayed in your terminal, typically `http://localhost:8501`.

## 🔐 Deployment Configuration

For Streamlit Community Cloud:

1. Connect your GitHub repository to Streamlit Community Cloud.
2. Configure application secrets in the Streamlit dashboard.
3. Add the required Groq API key and model configuration.
4. Configure the MongoDB Atlas connection string.
5. Configure the required Google OAuth credentials.
6. Deploy the application and verify login, Google authentication, chat, and mock interview features.

Example secrets configuration:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-120b"
MONGODB_URI = "your_mongodb_connection_string"
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
- Accessing personalized conversations and previous chat history.

## 🔮 Future Enhancements

- Support for multiple PDFs and document formats.
- Resume-based personalized interview questions.
- Difficulty levels: Beginner, Intermediate, and Advanced.
- Interview performance scoring and feedback.
- Interview progress tracking.
- Voice-based mock interview practice.
- More detailed source citations and answer evaluation.

## 👨‍💻 Author

**Gokul M**

GitHub: https://github.com/Gokulm29

---

⭐ If you find this project useful, consider giving the repository a star!
