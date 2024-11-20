from flask import Flask, request, jsonify
from hdbcli import dbapi
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores.hanavector import HanaDB
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
import os

app = Flask(__name__)

# Initialize global variables
hana_conn = None
hana_vectordb = None
retrieval_chain = None
total_records= None

# Define environment variables for SAP GenAI
os.environ['AICORE_CLIENT_ID'] = "sb-42a29a03-b2f4-47de-9a41-e0936be9aaf5!b256749|aicore!b164"
os.environ['AICORE_AUTH_URL'] = "https://gen-ai.authentication.us10.hana.ondemand.com"
os.environ['AICORE_CLIENT_SECRET'] = "b5e6caee-15aa-493a-a6ac-1fef0ab6e9fe$Satg7UGYPLsz5YYeXefHpbwTfEqqCkQEbasMDPGHAgU="
os.environ['AICORE_RESOURCE_GROUP'] = "default"
os.environ['AICORE_BASE_URL'] = "https://api.ai.prod.us-east-1.aws.ml.hana.ondemand.com/v2"

EMBEDDING_DEPLOYMENT_ID = "dc34e8c0d316a34b"
LLM_DEPLOYMENT_ID = "dd52e3661060696b"

# Initialize the proxy client
proxy_client = get_proxy_client("gen-ai-hub")
embeddings = OpenAIEmbeddings(deployment_id=EMBEDDING_DEPLOYMENT_ID)
llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)


@app.route("/",methods=['GET'])
def welcome():
    return jsonify({"status": "welcome to the API is running"}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

@app.route('/configure', methods=['POST'])
def configure_database():
    global hana_conn, hana_vectordb, retrieval_chain , total_records

    # Get the configuration payload
    payload = request.json
    host = payload.get("host")
    port = payload.get("port")
    user = payload.get("user")
    password = payload.get("password")

    try:
        # Connect to SAP HANA
        hana_conn = dbapi.connect(
            address=host,
            port=port,
            user=user,
            password=password
        )
        cur = hana_conn.cursor()
        cur.execute("TRUNCATE TABLE VECTORTABLE")

        # Fetch data from SAP HANA
        fetch_sql = "SELECT * FROM DBADMIN.DIX_TABLE1"
        cur.execute(fetch_sql)
        rows = cur.fetchall()

        # Convert rows to LangChain Documents
        documents = [
            Document(
                page_content=",".join(str(value) if value else 'NULL' for value in row).strip(),
                metadata={}
            )
            for row in rows
        ]

        # Initialize the HanaDB vector store and store embeddings
        hana_vectordb = HanaDB(embedding=embeddings, connection=hana_conn, table_name="VECTORTABLE")
        hana_vectordb.add_documents(documents)

        # Define the prompt and retrieval chain
        prompt = ChatPromptTemplate.from_template("""
        You are an AI chatbot assistant. Answer the user's question based on the provided SAP HANA database data.
        For initial messages like "hi" or "hello", respond with a welcoming message.
        Answer in a polite tone using the relevant retrieved information.

        {context} 
        Question: {input}
        """)
        fetch_count_sql = "SELECT COUNT(*) FROM DBADMIN.DIX_TABLE1"
        cur.execute(fetch_count_sql)
        total_records = cur.fetchone()[0]
        print(f"Total records in the database: {total_records}")

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = hana_vectordb.as_retriever(search_type="similarity", search_kwargs={"k": total_records})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return jsonify({"message": "Database configured and embeddings stored successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/invoke', methods=['POST'])
def invoke_query():
    global retrieval_chain, hana_vectordb, total_records

    if not retrieval_chain or not hana_vectordb:
        return jsonify({"error": "Database not configured. Please configure it first."}), 400

    # Get the user query
    payload = request.json
    user_query = payload.get("input", "")

    try:
        # Search documents and combine context
        docs = hana_vectordb.similarity_search(user_query, k=total_records)
        combined_context = "\n\n".join([doc.page_content for doc in docs])

        # Run the query through the retrieval chain
        response = retrieval_chain.invoke({"input": user_query, "context": combined_context})

        return jsonify({"answer": response.get("answer", "No answer found")}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
