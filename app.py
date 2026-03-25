import streamlit as st
import psycopg2
from groq import Groq
import pandas as pd
from psycopg2.extras import execute_values
import os

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": 5432,
    "sslmode": "require"
}

client = Groq(api_key=GROQ_API_KEY)


# --- DB CONNECTION ---
def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# --- FETCH SCHEMA FROM DB ---
def fetch_schema_from_db():
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)

        tables = cur.fetchall()
        schema_text = ""

        for table in tables:
            table_name = table[0]

            cur.execute(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table_name}';
            """)

            columns = cur.fetchall()
            col_names = [col[0] for col in columns]

            schema_text += f"{table_name}({', '.join(col_names)})\n"

        return schema_text.strip()

    except:
        return ""
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

# --- CREATE TABLE FROM CSV ---
def create_table_from_csv(df, table_name):
    conn = get_connection()
    cur = conn.cursor()

    cols = ", ".join([f"{col} TEXT" for col in df.columns])
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    cur.execute(f"CREATE TABLE {table_name} ({cols});")

    values = [tuple(map(str, row)) for row in df.values]
    query = f"INSERT INTO {table_name} VALUES %s"
    execute_values(cur, query, values)

    conn.commit()
    cur.close()
    conn.close()

# --- AI FUNCTION ---
def generate_sql(question, schema):
    prompt = f"""
    You are a SQL expert.

    Convert the user question into a PostgreSQL query.

    Rules:
    - Only generate SELECT queries
    - Do not use DELETE, UPDATE, DROP
    - Return ONLY SQL query (no explanation)

    Schema:
    {schema}

    Question: {question}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    sql = response.choices[0].message.content.strip()

    if "```" in sql:
        sql = sql.split("```")[1].replace("sql", "").strip()

    return sql

# --- RUN QUERY ---
def run_query(query):
    if "select" not in query.lower():
        return "❌ Only SELECT queries allowed"

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(query)

        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

        return columns, rows

    except Exception as e:
        return str(e)

    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

# --- UI ---
st.set_page_config(page_title="AI SQL Chat", page_icon="💬")

st.title("💬 AI SQL Chat with CSV Upload")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "schema" not in st.session_state:
    st.session_state.schema = ""

# --- CSV UPLOAD ---
st.subheader("📂 Upload CSV to Create Table")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
table_name = st.text_input("Enter table name")

if st.button("Create Table"):
    if uploaded_file and table_name:
        df = pd.read_csv(uploaded_file)
        create_table_from_csv(df, table_name)

        schema_text = f"{table_name}({', '.join(df.columns)})"
        st.session_state.schema = (st.session_state.schema + "\n" + schema_text).strip()

        st.success(f"✅ Table '{table_name}' created!")
    else:
        st.warning("Please upload CSV and enter table name")

# --- SCHEMA DISPLAY ---
st.subheader("📦 Current Schema")

schema_display = st.session_state.schema if st.session_state.schema else fetch_schema_from_db()
st.code(schema_display if schema_display else "No tables found")

# --- CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            st.code(msg["sql"], language="sql")

            if isinstance(msg["result"], tuple):
                columns, rows = msg["result"]
                st.dataframe([dict(zip(columns, r)) for r in rows])
            else:
                st.error(msg["result"])

# --- CHAT INPUT ---
question = st.chat_input("Ask your question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    # 🔥 SMART SCHEMA SELECTION
    schema_to_use = st.session_state.schema.strip()

    if not schema_to_use:
        schema_to_use = fetch_schema_from_db()

    sql = generate_sql(question, schema_to_use)
    result = run_query(sql)

    st.session_state.messages.append({
        "role": "assistant",
        "sql": sql,
        "result": result
    })

    with st.chat_message("assistant"):
        st.write("🧠 SQL:")
        st.code(sql, language="sql")

        if isinstance(result, tuple):
            columns, rows = result
            st.dataframe([dict(zip(columns, r)) for r in rows])
        else:
            st.error(result)

# --- CLEAR CHAT ---
st.markdown("---")
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
