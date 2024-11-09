from flask import Flask, jsonify, request
from hdbcli import dbapi
from pydantic import BaseModel
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

# Global variables
connection = None
vector_db = None
retrieval_chain = None

class ConfigPayload(BaseModel):
    appInfo: dict
    DbConfig: dict
    selctedModel: dict

class QueryPayload(BaseModel):
    input: str

@app.route("/", methods=["GET"])
def welcome():
    return jsonify({"message": "Welcome to the SAP HANA-based AI Chatbot API!"})

@app.route("/configure", methods=["POST"])
def configure():
    global connection, vector_db, retrieval_chain

    try:
        config_payload = ConfigPayload(**request.json)
        db_config = config_payload.DbConfig
        selected_model = config_payload.selctedModel

        # Connect to SAP HANA
        connection = dbapi.connect(
            address=db_config['hana_host'],
            port=int(db_config['hana_port']),
            user=db_config['hana_user'],
            password=db_config['hana_password']
        )
        
        # Initialize vector DB and embeddings
        embedding_model = selected_model.get("embedding", "intfloat/multilingual-e5-small")
        embed = SentenceTransformerEmbeddings(model_name=embedding_model)
        vector_db = HanaDB(embedding=embed, connection=connection, table_name="VECTORTABLE")

        # Fetch data from tables in batches to optimize memory
        cursor = connection.cursor()
        cursor.execute("SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = 'DBADMIN'")
        tables = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 100')
            for row in cursor.fetchall():
                combined_text = ",".join(str(value) or 'NULL' for value in row)
                vector_db.add_documents([{"page_content": combined_text, "metadata": {"table": table_name}}])

        # Set up Hugging Face model for response generation
        llm = HuggingFaceEndpoint(
            repo_id=selected_model.get("textGeneration", "mistralai/Mistral-7B-Instruct-v0.3"),
            huggingfacehub_api_token="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"  # Replace with actual token
        )

        # Configure retrieval chain
        prompt_template = ChatPromptTemplate.from_template("""
            You are an AI assistant. Respond based on provided SAP HANA DB data.
            {context}
            Question: {input}
        """)
        retrieval_chain = create_retrieval_chain(vector_db.as_retriever(), prompt_template, llm)

        return jsonify({"message": "Configuration completed successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    try:
        user_query = QueryPayload(**request.json).input

        # Retrieve relevant documents and process query
        docs = vector_db.similarity_search(user_query, k=2)
        combined_context = "\n\n".join(doc.page_content for doc in docs)
        response = retrieval_chain.invoke({"input": user_query, "context": combined_context})

        return jsonify({"answer": response['answer']})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
