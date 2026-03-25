import streamlit as st
import psycopg2
import pandas as pd
import re
import hashlib
from psycopg2.extras import execute_values
from groq import Groq
import os

# --- CONFIG ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "options": "-c statement_timeout=30000"  # ✅ 30 second timeout
}

client = Groq(api_key=GROQ_API_KEY)

# --- DB CONNECTION ---
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

# --- CLEAN NAMES ---
def clean_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    return name[:30]

# --- USER PREFIX FROM EMAIL ---
def get_user_prefix(user_email):
    return "u" + hashlib.md5(user_email.encode()).hexdigest()[:6]

# --- DATATYPE MAPPING ---
def map_dtype(dtype):
    dtype_str = str(dtype).lower()
    if "int" in dtype_str:
        return "INTEGER"
    elif "float" in dtype_str:
        return "FLOAT"
    elif "datetime" in dtype_str or "timestamp" in dtype_str:
        return "TIMESTAMP"
    else:
        return "TEXT"

# --- ENSURE REGISTRY TABLE EXISTS ---
def ensure_registry_table():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.user_tables (
                user_email TEXT,
                table_label TEXT,
                table_name  TEXT,
                created_at  TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
    finally:
        cur.close()
        conn.close()

# --- CREATE TABLE FROM CSV ---
def create_table_from_csv(df, table_label, user_email):
    df = df.head(5000)
    conn = get_connection()
    cur = conn.cursor()
    try:
        prefix     = get_user_prefix(user_email)
        table_name = f"{prefix}_{clean_name(table_label)}"

        # Limit tables per user
        cur.execute("""
            SELECT COUNT(*) FROM public.user_tables
            WHERE user_email = %s
        """, (user_email,))
        if cur.fetchone()[0] >= 20:
            raise Exception("❌ Max 20 tables allowed per user")

        # Register table (avoid duplicates)
        cur.execute("""
            INSERT INTO public.user_tables (user_email, table_label, table_name)
            SELECT %s, %s, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM public.user_tables
                WHERE user_email = %s AND table_name = %s
            );
        """, (user_email, table_label, table_name, user_email, table_name))

        # Build columns
        cols = ", ".join([
            f'"{col}" {map_dtype(df[col].dtype)}'
            for col in df.columns
        ])

        # Create table in public schema
        cur.execute(f'DROP TABLE IF EXISTS public."{table_name}";')
        cur.execute(f'CREATE TABLE public."{table_name}" ({cols});')

        # Insert data
        values = [tuple(row) for row in df.values]
        execute_values(
            cur,
            f'INSERT INTO public."{table_name}" VALUES %s',
            values,
            page_size=1000
        )

        conn.commit()
        return table_name

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

# --- GET USER TABLES ---
def get_user_tables(user_email):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT table_label, table_name
            FROM public.user_tables
            WHERE user_email = %s
            ORDER BY created_at DESC
        """, (user_email,))
        return cur.fetchall()  # [(label, actual_table_name), ...]
    finally:
        cur.close()
        conn.close()

# --- DELETE A USER TABLE ---
def delete_user_table(user_email, table_name):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(f'DROP TABLE IF EXISTS public."{table_name}";')
        cur.execute("""
            DELETE FROM public.user_tables
            WHERE user_email = %s AND table_name = %s
        """, (user_email, table_name))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

# --- FETCH SCHEMA STRUCTURE ---
def fetch_schema_structure(user_email):
    tables = get_user_tables(user_email)
    if not tables:
        return "", []
    conn = get_connection()
    cur = conn.cursor()
    try:
        schema_text = ""
        table_list  = []
        for label, tname in tables:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                ORDER BY ordinal_position
            """, (tname,))
            cols = [row[0] for row in cur.fetchall()]
            schema_text += f"{label} [table: {tname}] ({', '.join(cols)})\n"
            table_list.append((label, tname, cols))
        return schema_text.strip(), table_list
    finally:
        cur.close()
        conn.close()

# --- GENERATE SQL ---
def generate_sql(question, schema_text):
    prompt = f"""
You are a PostgreSQL expert.

Convert the user's question into a valid PostgreSQL SELECT query.

Rules:
- Only generate SELECT queries
- Never use DELETE, UPDATE, DROP, INSERT, ALTER, TRUNCATE
- Use the exact table name shown as [table: <name>] in the schema, always prefixed with public e.g. public."<table_name>"
- Column names must be double-quoted if they contain spaces or special characters
- Return ONLY the raw SQL query with no explanation and no markdown formatting

Schema:
{schema_text}

Question: {question}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    sql = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if "```" in sql:
        sql = re.sub(r"```(?:sql)?", "", sql).strip().strip("`").strip()
    return sql

# --- RUN QUERY ---
def run_query(query):
    if "select" not in query.lower():
        return "❌ Only SELECT queries are allowed"
    forbidden = ["drop", "delete", "update", "insert", "alter", "truncate"]
    if any(w in query.lower() for w in forbidden):
        return "❌ Forbidden operation detected in query"
    conn = None
    cur  = None
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(query)
        rows    = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return columns, rows
    except Exception as e:
        return f"❌ Query error: {str(e)}"
    finally:
        if cur:  cur.close()
        if conn: conn.close()

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="AI SQL Chat", page_icon="💬", layout="wide")
st.title("💬 AI SQL Chat")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- INIT REGISTRY TABLE ---
try:
    ensure_registry_table()
except Exception as e:
    st.error(f"DB init error: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🔐 Login")
user_email = st.sidebar.text_input("Enter your email")

if user_email:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Your Tables")
    user_tables = get_user_tables(user_email)
    if user_tables:
        for label, tname in user_tables:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.markdown(f"📄 **{label}**")
            if col2.button("🗑️", key=f"del_{tname}"):
                try:
                    delete_user_table(user_email, tname)
                    st.sidebar.success(f"Deleted '{label}'")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(str(e))
    else:
        st.sidebar.info("No tables yet. Upload a CSV to get started.")

# --- MAIN AREA: TWO COLUMNS ---
left, right = st.columns([1, 2])

with left:
    st.subheader("📂 Upload CSV Files")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if st.button("⬆️ Create Tables", use_container_width=True):
        if not user_email:
            st.warning("⚠️ Enter your email in the sidebar first")
        elif not uploaded_files:
            st.warning("⚠️ Upload at least one CSV file")
        else:
            for file in uploaded_files:
                try:
                    df         = pd.read_csv(file)
                    table_label = file.name.replace(".csv", "")
                    create_table_from_csv(df, table_label, user_email)
                    st.success(f"✅ '{table_label}' uploaded successfully")
                except Exception as e:
                    st.error(f"Error with {file.name}: {e}")
            st.rerun()

    # --- SCHEMA PREVIEW ---
    if user_email:
        st.markdown("---")
        st.subheader("📦 Schema Structure")
        try:
            schema_text, table_list = fetch_schema_structure(user_email)
            if schema_text:
                st.code(schema_text)
            else:
                st.info("No tables found. Upload a CSV to begin.")
        except Exception as e:
            st.error(f"Error fetching schema: {e}")

with right:
    st.subheader("💬 Chat with your Data")

    # --- CHAT HISTORY ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    if msg.get("sql"):
                        st.code(msg["sql"], language="sql")
                    if isinstance(msg.get("result"), tuple):
                        columns, rows = msg["result"]
                        if rows:
                            st.dataframe(
                                [dict(zip(columns, r)) for r in rows],
                                use_container_width=True
                            )
                        else:
                            st.info("Query returned no results.")
                    elif isinstance(msg.get("result"), str):
                        st.error(msg["result"])

    # --- CHAT INPUT ---
    question = st.chat_input("Ask a question about your data...")
    if question:
        if not user_email:
            st.warning("⚠️ Enter your email in the sidebar first")
        else:
            st.session_state.messages.append({"role": "user", "content": question})

            try:
                schema_text, _ = fetch_schema_structure(user_email)
                if not schema_text:
                    raise Exception("No tables found. Please upload a CSV first.")
                sql    = generate_sql(question, schema_text)
                result = run_query(sql)
            except Exception as e:
                sql    = ""
                result = f"❌ {str(e)}"

            st.session_state.messages.append({
                "role":   "assistant",
                "sql":    sql,
                "result": result
            })
            st.rerun()

    # --- CLEAR CHAT ---
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
