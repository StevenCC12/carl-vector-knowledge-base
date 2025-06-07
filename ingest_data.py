# ingest_data.py

import json
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MONGO_CONNECTION_STRING = "mongodb+srv://<user>:<password>@...." # PASTE YOUR CONNECTION STRING HERE
DB_NAME = "knowledge_base_db"
COLLECTION_NAME = "entries"
JSON_FILE_PATH = "webinar_qas.json"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def main():
    """
    Main function to read data, generate embeddings, and insert into MongoDB.
    """
    print("Starting data ingestion process...")

    # 1. Connect to MongoDB
    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Optional: Clear existing data to avoid duplicates on re-run
        collection.delete_many({})
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return

    # 2. Load the embedding model
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Read the JSON data
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            qas = json.load(f)
        print(f"Loaded {len(qas)} Q&A pairs from {JSON_FILE_PATH}.")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # 4. Process and prepare documents for insertion
    documents_to_insert = []
    print("Generating embeddings and preparing documents...")
    for item in qas:
        question = item.get("question")
        if not question:
            print(f"Skipping item due to missing question: {item}")
            continue

        # Generate the vector embedding for the question
        question_vector = model.encode(question).tolist()

        # Create the document for MongoDB
        document = {
            "source": "webinar",
            "questionText": question,
            "answerText": item.get("answer"),
            "questionVector": question_vector,
            "sourceDetails": {
                "webinarTitle": item.get("webinar_title"),
                "webinarDate": item.get("webinar_date")
            }
        }
        documents_to_insert.append(document)
    
    # 5. Batch insert documents into MongoDB
    if documents_to_insert:
        try:
            collection.insert_many(documents_to_insert)
            print(f"Successfully inserted {len(documents_to_insert)} documents into MongoDB.")
        except Exception as e:
            print(f"Error inserting documents into MongoDB: {e}")
    else:
        print("No documents were prepared for insertion.")

if __name__ == "__main__":
    main()