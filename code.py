import streamlit as st
from sqlalchemy import create_engine, inspect, MetaData, text
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from dotenv import load_dotenv
import re
import time
import plotly.express as px
import speech_recognition as sr   # 🔥 Voice input
import sounddevice as sd
from googletrans import Translator # 🔥 Multilingual support

translator = Translator()
sr.Microphone.list_microphone_names()

# --- Initialize Environment ---
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
# --- Constants ---
SCHEMA_HINTS = {
    "error": "Errors or failures are recorded in the 'log_data_information' table, likely in a column named 'error_desc' or 'log_message'.",
    "plc": "To find events related to PLCs, query the 'log_data_information' table.",
}
 
# --- Initialize AI Components ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("schema_descriptions")

# --- Global Stop Flag ---
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False
 
# --- Streaming Utilities ---
def stream_text(text, speed=0.03):
    text_placeholder = st.empty()
    current_text = ""
    for char in text:
        if st.session_state.stop_generation:  # 🔥 Stop button check
            break
        current_text += char
        text_placeholder.markdown(current_text + "▌")
        time.sleep(speed)
    text_placeholder.markdown(current_text)
    return current_text
 
def stream_dataframe(df, speed=0.02):
    container = st.empty()
    if df.empty:
        container.markdown("No data found")
        return df
    col_widths = [max(df[col].astype(str).str.len().max(), len(col)) for col in df.columns]
    header = "|" + "|".join(f" {str(col).ljust(width)} " for col, width in zip(df.columns, col_widths)) + "|"
    separator = "|" + "|".join(f":{'-'*(width)}:" for width in col_widths) + "|"
    rows = [
        "|" + "|".join(f" {str(row[col]).ljust(width)} " for col, width in zip(df.columns, col_widths)) + "|"
        for _, row in df.iterrows()
    ]
    displayed_table = ""
    for line in [header, separator] + rows:
        for char in line:
            if st.session_state.stop_generation:  # 🔥 Stop button check
                break
            displayed_table += char
            container.markdown(displayed_table + "▌")
            time.sleep(speed)
        displayed_table += "\n"
        container.markdown(displayed_table)
    return df
 
# --- Database Functions ---
def get_engine(db_type, connection_string):
    if db_type == "postgresql":
        return create_engine(connection_string)
    elif db_type == "sqlserver":
        try:
            import pyodbc
            return create_engine(connection_string)
        except ImportError:
            st.error("pyodbc not installed. Run: pip install pyodbc")
            raise
    else:
        raise ValueError("Unsupported database type")
 
def extract_schema(engine):
    inspector = inspect(engine)
    schema_info = []
    for table_name in inspector.get_table_names():
        table_info = {
            "table_name": table_name,
            "columns": [{"name": col["name"], "type": str(col["type"])}
                        for col in inspector.get_columns(table_name)],
            "sample_rows": []
        }
        try:
            with engine.connect() as conn:
                quote_char = '"' if engine.dialect.name == 'postgresql' else ']'
                open_quote = '[' if quote_char == ']' else '"'
                sample_query = text(f'SELECT TOP 3 * FROM {open_quote}{table_name}{quote_char}') if engine.dialect.name == 'mssql' else text(f'SELECT * FROM "{table_name}" LIMIT 3')
                sample_result = conn.execute(sample_query)
                table_info["sample_rows"] = pd.DataFrame(sample_result.fetchall(), columns=sample_result.keys()).to_dict(orient='records')
        except Exception:
            pass
        schema_info.append(table_info)
    return schema_info
 
def schema_to_nl(schema_info):
    nl_descriptions = []
    for table in schema_info:
        columns_desc = [f"{col['name']} ({col['type']})" for col in table["columns"]]
        description = f"The table '{table['table_name']}' has columns: {', '.join(columns_desc)}."
        if table["sample_rows"]:
            sample_data_str = ', '.join([str(row) for row in table["sample_rows"]])
            description += f" Sample rows: [{sample_data_str}]."
        nl_descriptions.append({"table_name": table["table_name"], "description": description})
    return nl_descriptions
 
def store_embeddings(nl_descriptions):
    try:
        collection.delete(where={"table_name": {"$ne": "dummy"}})
    except Exception:
        pass
    texts = [desc["description"] for desc in nl_descriptions]
    metadatas = [{"table_name": desc["table_name"]} for desc in nl_descriptions]
    ids = [f"id_{i}" for i in range(len(nl_descriptions))]
    embeddings = embedding_model.encode(texts)
    collection.add(embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas, ids=ids)
 
def get_relevant_context(query, n_results=5):
    query_embedding = embedding_model.encode(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
    if not results.get("ids") or not results["ids"][0]:
        return []
    return [
        {"table_name": meta["table_name"], "description": doc}
        for meta, doc in zip(results["metadatas"][0], results["documents"][0])
    ]
 
def generate_sql_from_ai(question, context, db_type, hints):
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    context_str = "\n".join([ctx['description'] for ctx in context])
    hints_str = "\n".join(f"- {hint}" for hint in hints)
    system_prompt = f"""
    You are a {db_type} SQL expert. Generate a SQL query based on the user's question.
    Only return the SQL query without any additional explanation or formatting. Understand the question in which language it is asked and then reply in that language only.
    {f"\nIMPORTANT HINTS:\n{hints_str}" if hints else ""}
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Schema Context with Sample Data:\n{context_str}\n\nQuestion: {question}")
    ]
    response = llm.invoke(messages)
    return response.content.strip()
 
def clean_sql_aggressively(raw_sql):
    sql_keywords = ['select', 'insert', 'update', 'delete', 'with', 'create', 'alter', 'drop', 'truncate']
    pattern = re.compile(r'\b(' + '|'.join(sql_keywords) + r')\b', re.IGNORECASE)
    match = pattern.search(raw_sql)
    cleaned_sql = raw_sql[match.start():] if match else raw_sql
    cleaned_sql = cleaned_sql.strip()
    if cleaned_sql.endswith("```"):
        cleaned_sql = cleaned_sql[:-3].strip()
    if cleaned_sql.endswith(";"):
        cleaned_sql = cleaned_sql[:-1].strip()
    return cleaned_sql
 
def execute_query(engine, query):
    with engine.connect() as conn:
        try:
            result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            st.error(f"Query execution failed:\n\nError: {str(e)}")
            return pd.DataFrame()
 
def summarize_results_in_nl(user_question, df):
    if df.empty:
        return "I found no results for your query."
    data_str = df.to_string(index=False)
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    system_prompt = """
    You are a data analyst. Summarize the query results in a concise, natural language response.
    Focus on key insights and important patterns. Do not show raw data or SQL.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {user_question}\n\nQuery Results:\n{data_str}")
    ]
    response = llm.invoke(messages)
    return response.content
 
# --- Voice Input Function ---
def get_voice_input(status_placeholder):
    recognizer = sr.Recognizer()
    translator = Translator()
    with sr.Microphone() as source:
        status_placeholder.write("🎤 Listening... Speak your question.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        status_placeholder.write(f"✅ Recognized Speech: {text}")
        
        if text.strip():
            try:
                detected_lang = translator.detect(text).lang
            except Exception:
                detected_lang = "Unknown"
        else:
            detected_lang = "Unknown"
        return text, detected_lang
    except sr.UnknownValueError:
        status_placeholder.write("⚠ Could not understand audio.")
        return "", "Unknown"
    except sr.RequestError as e:
        status_placeholder.write(f"❌ Speech Recognition error: {e}")
    return "", "Unknown"
 
# --- Main Application ---
def main():
    st.set_page_config(page_title="SQL Chatbot", page_icon="💬", layout="wide")
    st.title("SQL Chatbot 💬")
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph_states" not in st.session_state:
        st.session_state.graph_states = {}
 
    with st.sidebar:
        st.header("Database Connection")
        db_type = st.selectbox("Database Type", ["postgresql", "sqlserver"])
 
        if db_type == "sqlserver":
            st.subheader("SQL Server Connection")
            auth_method = st.radio("Authentication", ("SQL Server Authentication", "Windows Authentication"))
            server = st.text_input("Server")
            database = st.text_input("Database")
            if auth_method == "SQL Server Authentication":
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
            else:
                conn_str = f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        else:
            st.subheader("PostgreSQL Connection")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            host = st.text_input("Host", "localhost")
            port = st.text_input("Port", "5432")
            database = st.text_input("Database")
            conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database}"
 
        if st.button("Connect to Database") and "engine" not in st.session_state:
            with st.spinner("Connecting to database..."):
                try:
                    engine = get_engine(db_type, conn_str)
                    schema_info = extract_schema(engine)
                    nl_descriptions = schema_to_nl(schema_info)
                    store_embeddings(nl_descriptions)
                    st.session_state.engine = engine
                    st.session_state.db_type = db_type
                    st.session_state.messages = []
                    st.session_state.graph_states = {}
                    st.success("Database connected successfully!")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
 
    # --- Voice Input Section (after connection) ---
    if "engine" in st.session_state:
        st.sidebar.subheader("🎤 Voice Input")
        status_placeholder = st.sidebar.empty()
        if st.sidebar.button("Start Voice Input"):
            voice_text, detected_lang = get_voice_input(status_placeholder)
            if voice_text:
                st.session_state.voice_prompt = voice_text
                st.session_state.input_lang = detected_lang

        # --- Stop Button ---
        if st.sidebar.button("🛑 Stop Generation"):
            st.session_state.stop_generation = True
 
    # Main chat interface
    if "engine" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "table" in message and not message["table"].empty:
                    st.dataframe(message["table"])
                    if i not in st.session_state.graph_states:
                        st.session_state.graph_states[i] = False
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📊 Show Graph", key=f"show_{i}"):
                            st.session_state.graph_states[i] = True
                            st.rerun()
                    with col2:
                        if st.button(f"❌ Hide Graph", key=f"hide_{i}"):
                            st.session_state.graph_states[i] = False
                            st.rerun()
                    if st.session_state.graph_states[i]:
                        st.subheader("Data Visualization")
                        chart_type = st.selectbox(
                            "Chart Type",
                            ["Bar", "Line", "Area", "Scatter", "Histogram", "Pie", "Box", "Violin", "Heatmap", "Density Contour"],
                            key=f"chart_type_{i}"
                        )
                        numeric_df = message["table"].select_dtypes(include='number')
                        if not numeric_df.empty:
                            chart_key = f"chart_rendered_{i}"
                            if chart_key not in st.session_state:
                                st.session_state[chart_key] = None
                            if st.session_state[chart_key] != chart_type:
                                st.session_state[chart_key] = chart_type
                            if chart_type == "Bar":
                                st.bar_chart(numeric_df)
                            elif chart_type == "Line":
                                st.line_chart(numeric_df)
                            elif chart_type == "Area":
                                st.area_chart(numeric_df)
                            elif chart_type == "Scatter":
                                if len(numeric_df.columns) >= 2:
                                    fig = px.scatter(message["table"], x=numeric_df.columns[0], y=numeric_df.columns[1])
                                    st.plotly_chart(fig)
                                else:
                                    st.warning("Need at least two numeric columns for scatter plot")
                            elif chart_type == "Histogram":
                                fig = px.histogram(message["table"], x=numeric_df.columns[0])
                                st.plotly_chart(fig)
                            elif chart_type == "Pie":
                                if len(message["table"].columns) >= 2:
                                    fig = px.pie(message["table"], values=numeric_df.columns[0], names=message["table"].columns[1])
                                    st.plotly_chart(fig)
                                else:
                                    st.warning("Need a value column and a names column for pie chart")
                            elif chart_type == "Box":
                                fig = px.box(message["table"], y=numeric_df.columns)
                                st.plotly_chart(fig)
                            elif chart_type == "Violin":
                                fig = px.violin(message["table"], y=numeric_df.columns)
                                st.plotly_chart(fig)
                            elif chart_type == "Heatmap":
                                fig = px.imshow(numeric_df.corr())
                                st.plotly_chart(fig)
                            elif chart_type == "Density Contour":
                                if len(numeric_df.columns) >= 2:
                                    fig = px.density_contour(message["table"], x=numeric_df.columns[0], y=numeric_df.columns[1])
                                    st.plotly_chart(fig)
                                else:
                                    st.warning("Need at least two numeric columns for density contour plot")
                        else:
                            st.warning("No numeric data available for visualization.")
                if "sql" in message:
                    with st.expander("View SQL Query"):
                        st.code(message["sql"], language="sql")
 
        # --- Text or Voice Chat Input ---
        prompt = st.chat_input("Ask about your data...") or st.session_state.get("voice_prompt", None)
        if prompt:
            if "voice_prompt" in st.session_state:
                st.session_state.voice_prompt = None
                
            try:
                detected_lang = translator.detect(prompt).lang
            except Exception:
                detected_lang = "en"
            st.session_state.input_lang = detected_lang
 
            st.session_state.messages.append({"role": "user", "content": prompt})
            assistant_message = {"role": "assistant","content": "","streamed_content": "","streaming": True}
            st.session_state.messages.append(assistant_message)
            
            with st.spinner("Processing your question..."):
                try:
                    st.session_state.stop_generation = False  # Reset stop flag
                    active_hints = {hint for keyword, hint in SCHEMA_HINTS.items() if keyword in prompt.lower()}
                    context = get_relevant_context(prompt)
                    raw_sql = generate_sql_from_ai(prompt, context, st.session_state.db_type, list(active_hints))
                    cleaned_sql = clean_sql_aggressively(raw_sql)
                    df = execute_query(st.session_state.engine, cleaned_sql)
                    
                    with st.chat_message("assistant"):
                        summary = summarize_results_in_nl(prompt, df)
                        
                        target_lang = st.session_state.get("input_lang", "en")
                        if target_lang != "en":
                            try:
                                summary = translator.translate(summary, dest=target_lang).text
                            except Exception as e:
                                st.warning(f"Translation failed: {str(e)}")
                                
                        streamed_text = stream_text(summary)
                        streamed_df = stream_dataframe(df)
                        assistant_response = {
                            "role": "assistant",
                            "content": summary,
                            "streamed_content": streamed_text,
                            "table": df,
                            "streamed_table": streamed_df,
                            "sql": cleaned_sql
                        }
                        st.session_state.messages[-1] = assistant_response
                        message_index = len(st.session_state.messages) - 1
                        st.session_state.graph_states[message_index] = False
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.session_state.messages[-1] = {
                        "role": "assistant",
                        "content": f"❌ Error: {str(e)}",
                        "streaming": False
                    }
            
if __name__ == "__main__":
    main()
