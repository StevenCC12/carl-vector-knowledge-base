import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file at the very top
load_dotenv()

# --- Configuration (loaded from environment) ---
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
VECTOR_SEARCH_INDEX = os.getenv("VECTOR_SEARCH_INDEX")

# --- Lifespan Event Handler ---
# This context manager will handle startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application lifespan. Runs on startup and shutdown.
    """
    # === Code to run on STARTUP ===
    print("Application startup: Loading resources...")
    if not MONGO_CONNECTION_STRING:
        raise RuntimeError("MONGO_CONNECTION_STRING not found in .env file")

    # Store resources in the app state to be accessible by endpoints
    app.state.model = SentenceTransformer(MODEL_NAME)
    print("SentenceTransformer model loaded.")
    
    app.state.db_client = MongoClient(MONGO_CONNECTION_STRING, server_api=ServerApi('1'))
    app.state.db_client.admin.command('ping')
    print("Successfully connected to MongoDB.")
    
    app.state.db_collection = app.state.db_client[DB_NAME][COLLECTION_NAME]
    
    yield # The application is now running and serving requests

    # === Code to run on SHUTDOWN ===
    print("Application shutdown: Closing resources...")
    app.state.db_client.close()
    print("MongoDB connection closed.")


# --- Initialize FastAPI Application ---
# Pass the lifespan handler to the FastAPI constructor
app = FastAPI(lifespan=lifespan)


# --- Pydantic Models for API Data Structure ---
class QuestionRequest(BaseModel):
    question: str

class APIResponse(BaseModel):
    action: str
    message: str
    score: float
    retrieved_question: str | None = None


# --- API Endpoints ---
@app.post("/find-similar-question", response_model=APIResponse)
def find_similar_question(request: QuestionRequest):
    """
    Accepts a question, finds the most similar one in the knowledge base,
    and returns an action based on the similarity score.
    """
    # 1. Generate embedding for the incoming question
    try:
        incoming_vector = app.state.model.encode(request.question).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode question: {e}")

    # 2. Build the MongoDB Vector Search query
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_SEARCH_INDEX,
                "path": "questionVector",
                "queryVector": incoming_vector,
                "numCandidates": 10,
                "limit": 1
            }
        },
        {
            "$project": {
                "score": { "$meta": "vectorSearchScore" },
                "questionText": 1,
                "answerText": 1,
                "_id": 0
            }
        }
    ]

    # 3. Execute the query
    try:
        results = list(app.state.db_collection.aggregate(pipeline))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

    # 4. Apply threshold logic to the result
    if not results:
        return APIResponse(
            action="escalate-to-human",
            message="No similar question found in knowledge base.",
            score=0.0
        )
    
    top_match = results[0]
    similarity_score = top_match.get("score")
    retrieved_answer = top_match.get("answerText")
    retrieved_question = top_match.get("questionText")

    # These thresholds are a starting point. You'll need to test and tune them.
    if similarity_score >= 0.95: # High confidence
        action = "auto-reply"
        message = retrieved_answer
    elif similarity_score > 0.80: # Medium confidence
        action = "draft-for-review"
        message = retrieved_answer
    else: # Low confidence
        action = "escalate-to-human"
        message = "No highly similar answer found in knowledge base."

    return APIResponse(
        action=action,
        message=message,
        score=similarity_score,
        retrieved_question=retrieved_question
    )

@app.get("/health")
def health_check():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok"}