import json, os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient # Updated import
from pymongo.server_api import ServerApi      # New import
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# --- Configuration (loaded from environment) ---
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
JSON_FILE_PATH = "webinars.json"

def main():
    """
    Main function to read data, generate embeddings, and insert into MongoDB.
    """
    print("Starting data ingestion process...")

    # 1. Connect to MongoDB
    try:
        client = MongoClient(MONGO_CONNECTION_STRING, server_api=ServerApi('1')) # Use ServerApi
        # The ping test is not strictly necessary here, as the first operation will confirm the connection.
        # But we can add it for a good initial check.
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # ...
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