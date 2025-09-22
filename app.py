from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
import torch

# --- 1. SETUP ---
# Create the FastAPI application with metadata
app = FastAPI(
    title="Email Reply Classifier API",
    description="An API to classify email replies as positive, negative, or neutral using a fine-tuned DistilBERT model.",
    version="1.0.0",
)

# Define the local path to your saved model
MODEL_PATH = "./reply_classifier_model"

# Configure Jinja2 to find HTML templates in the "templates" directory
templates = Jinja2Templates(directory="templates")

# --- 2. LOAD THE ML MODEL ---
# Load the fine-tuned model and tokenizer from the local directory.
# The 'pipeline' is a high-level helper from Hugging Face that simplifies inference.
print("Loading model...")
# Use GPU if available (device=0), otherwise use CPU (device=-1)
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=device
)
print(f"✅ Model loaded successfully on device: {'cuda:0' if device == 0 else 'cpu'}")


# --- 3. Pydantic Models for Data Validation ---
# Define the structure of the incoming request body
class PredictRequest(BaseModel):
    text: str # e.g., "Looking forward to the demo!"

# Define the structure of the outgoing response body
class PredictResponse(BaseModel):
    label: str      # e.g., "positive"
    confidence: float # e.g., 0.9876


# --- 4. API ENDPOINTS ---
@app.get("/", response_class=HTMLResponse, summary="User Interface")
async def serve_ui(request: Request):
    """
    Serves the main user-friendly HTML interface.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictResponse, summary="Classify Email Reply")
def predict(request: PredictRequest):
    """
    Accepts a text string and returns the predicted label and confidence score.
    """
    # Basic validation for empty input
    if not request.text or not request.text.strip():
        # In a real app, you might raise an HTTPException here
        return {"error": "Input text cannot be empty."}
        
    # The classifier returns a list of dictionaries, e.g., [{'label': 'positive', 'score': 0.9876}]
    # We take the first result [0] as we only pass one text string.
    prediction = classifier(request.text)[0]

    # Extract the label and score and return them in the defined Pydantic response format
    return PredictResponse(
        label=prediction['label'],
        confidence=round(prediction['score'], 4)
    )