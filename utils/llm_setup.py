import os
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from accelerate import Accelerator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Load .env file
load_dotenv()

# Get model path from environment variable
model_path = os.getenv("MODEL_PATH")
if not model_path:
    raise ValueError("❌ MODEL_PATH is not set! Please check your .env file.")

# Configure the LLM Model
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    n_ctx=2048,
    n_gpu_layers=20,  # Use all layers on GPU
    n_batch=512,
    callback_manager=callback_manager,
    f16_kv=True,  # Enable mixed precision 
    verbose=True
)

# Accelerator for faster processing
accelerator = Accelerator()
llm = accelerator.prepare(llm)

# Configure embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Global settings
Settings.chunk_size = 1024
Settings.llm = llm
Settings.embed_model = embed_model