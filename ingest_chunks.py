import os
import glob
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# --- Configuration (loaded from environment) ---
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = os.getenv("DB_NAME")
KNOWLEDGE_CHUNKS_COLLECTION_NAME = os.getenv("KNOWLEDGE_CHUNKS_COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
TRANSCRIPTS_PATH = "transcripts/*.txt" # Path to find all .txt files in the transcripts folder

def main():
    """
    Main function to find transcripts, chunk them, generate embeddings,
    and insert them into the 'knowledge_chunks' collection.
    """
    if not all([MONGO_CONNECTION_STRING, DB_NAME, KNOWLEDGE_CHUNKS_COLLECTION_NAME, MODEL_NAME]):
        raise ValueError("One or more required environment variables are not set.")

    print("Starting knowledge chunk ingestion process...")

    # --- 1. Initialize Models and Database Connection ---
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_CONNECTION_STRING, server_api=ServerApi('1'))
        client.admin.command('ping')
        db = client[DB_NAME]
        collection = db[KNOWLEDGE_CHUNKS_COLLECTION_NAME]
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return

    # --- 2. Initialize the Text Splitter ---
    # We use RecursiveCharacterTextSplitter for effective, semantic chunking.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Max characters per chunk
        chunk_overlap=100  # Characters to overlap between chunks
    )

    # --- 3. Find and Process Each Transcript File ---
    transcript_files = glob.glob(TRANSCRIPTS_PATH)
    if not transcript_files:
        print(f"No transcript files found at '{TRANSCRIPTS_PATH}'. Please check the path.")
        return
        
    print(f"Found {len(transcript_files)} transcript files to process.")

    for filepath in transcript_files:
        source_name = os.path.basename(filepath)
        print(f"\n--- Processing file: {source_name} ---")

        # To make this script re-runnable, we first delete existing chunks from this specific source.
        print(f"Deleting existing chunks for source: {source_name}...")
        collection.delete_many({"source_name": source_name})

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            continue # Skip to the next file

        # --- 4. Chunk the Text ---
        print("Splitting text into chunks...")
        chunks = text_splitter.split_text(full_text)
        print(f"Text split into {len(chunks)} chunks.")

        if not chunks:
            print("No chunks were generated for this file.")
            continue

        # --- 5. Generate Embeddings and Prepare Documents ---
        print("Generating embeddings for each chunk...")
        documents_to_insert = []
        for i, chunk_text in enumerate(chunks):
            chunk_vector = model.encode(chunk_text).tolist()
            
            document = {
                "source_type": "transcript",
                "source_name": source_name,
                "content": chunk_text,
                "content_vector": chunk_vector,
                "chunk_number": i + 1,
            }
            documents_to_insert.append(document)

        # --- 6. Batch Insert into MongoDB ---
        if documents_to_insert:
            print(f"Inserting {len(documents_to_insert)} new chunks into MongoDB...")
            collection.insert_many(documents_to_insert)
            print("Insertion complete for this source.")
        
    print("\nAll transcript files processed successfully.")

if __name__ == "__main__":
    main()