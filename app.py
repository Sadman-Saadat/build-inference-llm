import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

import wandb
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Configuration
PROJECT_NAME = "bengali-gemma3-finetune"
ARTIFACT_NAME = "bengali-gemma3-lora-model"
VERSION = "v0"
BASE_MODEL_NAME = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"

class GenerateRequest(BaseModel):
    text: str
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerateResponse(BaseModel):
    generated_text: str
    input_text: str

async def load_model():
    """Load model and tokenizer from Wandb artifact"""
    global model, tokenizer
    
    logger.info("=== Starting Model Loading from Wandb ===")
    
    # Initialize wandb and download model
    run = wandb.init(project=PROJECT_NAME, job_type="inference")
    artifact = run.use_artifact(f"{ARTIFACT_NAME}:{VERSION}")
    adapter_dir = artifact.download()
    
    logger.info(f"LoRA adapter downloaded to: {adapter_dir}")
    
    # Load tokenizer from base model
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Apply LoRA adapter using PEFT
    logger.info("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    logger.info("✅ Model with adapter loaded!")
    wandb.finish()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Bengali Gemma3 Inference Server",
    description="API server for Bengali Gemma3 model inference",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the loaded model"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"Generating response for: {request.text}")
    
    # Prepare input
    messages = [{"role": "user", "content": request.text}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Handle list return
    if isinstance(text, list):
        text = text[0] if text else ""
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract generated part
    if "<start_of_turn>model\n" in response:
        generated_part = response.split("<start_of_turn>model\n")[-1].replace("<end_of_turn>", "").strip()
    else:
        # Remove input part
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        generated_part = response.replace(input_text, "").strip()
    
    logger.info(f"Generated response: {generated_part}")
    
    return GenerateResponse(
        generated_text=generated_part,
        input_text=request.text
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
