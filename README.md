# AI-Powered-SQL-ChatBot

Here’s a detailed **README.md** file for your Streamlit SQL Chatbot project:

---

# 💬 SQL Chatbot with Voice & Multilingual Support

A powerful **Streamlit-based AI SQL Assistant** that allows you to **query databases in natural language** — using **text or voice input** — and automatically generates SQL queries, executes them, and visualizes results interactively.

This chatbot integrates **Azure OpenAI**, **LangChain**, and **ChromaDB** to enable intelligent query generation and context-aware reasoning over database schemas.

---

## 🚀 Features

### 🧠 AI-Powered Query Generation

* Converts **natural language questions** into optimized SQL queries using **Azure OpenAI GPT models**.
* Understands schema context automatically via **ChromaDB embeddings**.

### 🎤 Voice Input

* Ask questions via your **microphone** — powered by **SpeechRecognition** and **Google Speech API**.

### 🌍 Multilingual Support

* Ask questions in any language (e.g., Hindi, French, Spanish).
* The bot automatically detects the language and responds in the **same language** using **Google Translate**.

### 📊 Interactive Data Visualization

* Automatically visualize query results with **Plotly** or Streamlit’s chart components.
* Supports multiple chart types: Bar, Line, Pie, Histogram, Scatter, Box, Violin, Heatmap, and Density Contour.

### ⏹️ Real-time Streaming & Stop Button

* Responses and tables appear **streamingly**, character by character.
* You can **stop generation anytime** using the sidebar **🛑 Stop Generation** button.

### 💾 Schema-Aware Embeddings

* Automatically extracts schema metadata and stores embeddings using **Sentence Transformers (`all-MiniLM-L6-v2`)** in a **persistent ChromaDB** collection.

---

## 🧩 Tech Stack

| Component                  | Technology                                     |
| -------------------------- | ---------------------------------------------- |
| **Frontend**               | Streamlit                                      |
| **AI/LLM**                 | Azure OpenAI (`AzureChatOpenAI` via LangChain) |
| **Embeddings**             | Sentence Transformers (`all-MiniLM-L6-v2`)     |
| **Database ORM**           | SQLAlchemy                                     |
| **Vector Storage**         | ChromaDB                                       |
| **Visualization**          | Plotly, Streamlit Charts                       |
| **Voice Input**            | SpeechRecognition, sounddevice                 |
| **Translation**            | Googletrans                                    |
| **Environment Management** | python-dotenv                                  |

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sql-chatbot.git
cd sql-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` File

Add your Azure OpenAI credentials inside `.env`:

```ini
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-01
```

---

## 🗄️ Supported Databases

| Database       | Connection Method                                                                  |
| -------------- | ---------------------------------------------------------------------------------- |
| **PostgreSQL** | SQLAlchemy URI: `postgresql://username:password@host:port/database`                |
| **SQL Server** | Uses `pyodbc` driver. Supports both SQL Authentication and Windows Authentication. |

Example connection string for SQL Server:

```bash
mssql+pyodbc://username:password@SERVER_NAME/DATABASE_NAME?driver=ODBC+Driver+17+for+SQL+Server
```

---

## 🧠 How It Works

1. **Schema Extraction** – The app connects to your database and extracts schema info (tables, columns, and sample rows).
2. **Embedding Creation** – Schema descriptions are encoded and stored in **ChromaDB** for contextual understanding.
3. **Query Understanding** – When you ask a question, the app retrieves the most relevant schema context using vector similarity search.
4. **SQL Generation** – The AI model generates an accurate SQL query in the same language as the question.
5. **Query Execution** – SQL is executed via SQLAlchemy, and results are shown in a streamed markdown table.
6. **Natural Language Summary** – Results are summarized in plain language.
7. **Visualization** – Optionally, visualize data through dynamic charts.

---

## 🎤 Voice Query Example

1. Click **“Start Voice Input”** from the sidebar.
2. Ask a question like:

   > “Show me the total sales by product category.”
3. The chatbot detects and transcribes your question.
4. It then generates the corresponding SQL query and shows the results.

---

## 🌍 Multilingual Example

* Ask in Hindi:

  > "हर ग्राहक की कुल खरीद दिखाओ"
* The bot detects Hindi, generates SQL, executes it, and replies in Hindi.

---

## 📊 Visualization Options

After query execution, click:

* **📊 Show Graph** – to visualize the result.
* Choose among chart types:

  * Bar, Line, Area, Scatter, Histogram, Pie, Box, Violin, Heatmap, Density Contour.
* **❌ Hide Graph** – to close the visualization.

---

## 🛑 Stop Button

If AI generation or streaming takes too long, use the sidebar **🛑 Stop Generation** button to immediately halt output generation.

---

## 📦 Requirements

Create a `requirements.txt` file like this:

```txt
streamlit
sqlalchemy
langchain
langchain-openai
chromadb
sentence-transformers
pandas
python-dotenv
plotly
speechrecognition
sounddevice
googletrans==4.0.0-rc1
pyodbc
```

---

## 🧑‍💻 Run the App

```bash
streamlit run app.py
```

Once launched, open your browser at:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧾 Example Workflow

1. Connect to your database (SQL Server or PostgreSQL).
2. Extract schema automatically.
3. Ask a question like:

   > “List the top 5 products by revenue.”
4. The bot generates SQL, executes it, summarizes results, and optionally visualizes them.

---

## ⚙️ Folder Structure

```
sql-chatbot/
│
├── app.py                 # Main Streamlit application
├── chroma_db/             # Persistent vector database storage
├── requirements.txt
├── .env                   # Azure OpenAI environment variables
└── README.md
```

---

## 🧩 Future Enhancements

* ✅ Add MySQL and Oracle support
* ✅ Integrate caching for repeated queries
* ✅ Improve multilingual translation accuracy
* ✅ Enable downloadable query result reports (CSV/XLSX)

---

## 🧑‍🏫 Author

**Aaditya Gupta**
AI/ML Engineer | Cloud Developer | Streamlit Enthusiast
📧 [[aaditya200805@gmail.com](mailto:aaditya200805@gmail.com)]
🌐 [(https://www.linkedin.com/in/aaditya-gupta200802/overlay/about-this-profile/) / https://github.com/AADITYAXX/]

