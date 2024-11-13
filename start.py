from flask import Flask, jsonify, request
from hdbcli import dbapi
from pydantic import BaseModel
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)

# Global variables for model instances and configurations
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

@app.route("/getHome",methods=["GET"])
def home():
    return jsonify({"message": "Welcome to my home"})

@app.route("/configure", methods=["POST"])
def configure():
    global connection, vector_db, retrieval_chain

    try:
        config_payload = ConfigPayload(**request.json)
        app_info = config_payload.appInfo
        db_config = config_payload.DbConfig
        selected_model = config_payload.selctedModel

        # Connect to SAP HANA
        try:
            connection = dbapi.connect(
                address=db_config['hana_host'],
                port=int(db_config['hana_port']),
                user=db_config['hana_user'],
                password=db_config['hana_password']
            )
            print("Successfully connected to HANA database")
        except Exception as e:
            return jsonify({"error": "Failed to connect to SAP HANA DB"}), 500

        # Initialize vector DB and embedding model
        embedding_model = selected_model.get("embedding") or "intfloat/multilingual-e5-small"
        embed = SentenceTransformerEmbeddings(model_name=embedding_model)
        vector_db = HanaDB(
            embedding=embed,
            connection=connection,
            table_name="VECTORTABLE"
        )

        # Process and add documents to vector DB
        cursor = connection.cursor()
        cursor.execute("SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = 'DBADMIN'")
        tables = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 100')  # Fetching only 100 rows at a time
            rows = cursor.fetchall()
            for row in rows:
                combined_text = ",".join(str(value) if value else 'NULL' for value in row)
                document = Document(page_content=combined_text.strip(), metadata={"table": table_name})
                vector_db.add_documents([document])  # Directly add to vector DB

        # Configure the Hugging Face model
        llm = HuggingFaceEndpoint(
            repo_id=selected_model.get("textGeneration") or "mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token="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"  # Replace with actual token
        )

        # Define prompt template
        prompt_template = ChatPromptTemplate.from_template("""
            You are an expert AI assistant designed to support a vendor onboarding process. Your role is to assist with clear, professional, and helpful responses based on the SAP HANA database data for vendor onboarding statuses.
            when you get the response use those reponse and give a human form of answer.
            Give the response in as a customer support person
            Here is the relevant data for context:
            based on the relevent data answer according to that
            {context}
            
            Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = vector_db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return jsonify({"message": "Configuration completed successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    global retrieval_chain, vector_db

    try:
        query_payload = QueryPayload(**request.json)
        user_query = query_payload.input

        # Perform similarity search in vector DB
        docs = vector_db.similarity_search(user_query, k=2)
        combined_context = "\n\n".join([doc.page_content for doc in docs])

        # Debugging: Print combined context to verify retrieved chunks
        print("Combined Context:\n", combined_context)

        # Run the query through the retrieval chain
        response = retrieval_chain.invoke({"input": user_query, "context": combined_context})

        # Debugging: Print response to verify the output format
        print("Response from retrieval chain:", response)

        # Attempt to retrieve the 'answer' from the response, or use a fallback if missing
        answer_text = response.get('answer', 'No answer available')
        
        return jsonify({
            "answer": answer_text,
            "details": {
                "input": user_query,
                "context": combined_context,
                "answer": answer_text,
            }
        })

    except Exception as e:
        # Debugging: Print exception for troubleshooting
        print("Error during query processing:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
