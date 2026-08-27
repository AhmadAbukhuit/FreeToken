import asyncio
import json
import logging
import os
import socket
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State to keep track of running models
# Format: {"model_name": {"port": int, "process": subprocess.Popen, "status": str}}
RUNNING_MODELS: Dict[str, Dict[str, Any]] = {}

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup on shutdown
    for model_name, info in list(RUNNING_MODELS.items()):
        logger.info(f"Terminating {model_name}...")
        info["process"].terminate()
    
app = FastAPI(lifespan=lifespan)
client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

# Serve the UI
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("gateway_ui/index.html", "r") as f:
        return f.read()

# Management APIs
@app.get("/api/models")
async def list_models():
    # Return status of all running models
    result = {}
    for name, info in RUNNING_MODELS.items():
        # Check if process is still alive if it's running
        if info.get("process") is not None and info["process"].poll() is not None:
            info["status"] = "crashed"
            info["progress"] = "Process died unexpectedly."
        result[name] = {
            "port": info["port"],
            "status": info["status"],
            "progress": info.get("progress", "Running")
        }
    return result

@app.post("/api/start")
async def start_model(request: Request):
    data = await request.json()
    model_name = data.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")
        
    if model_name in RUNNING_MODELS and RUNNING_MODELS[model_name]["status"] not in ["crashed", "failed"]:
        return {"status": "already running or downloading"}
        
    port = get_free_port()
    
    # Initialize the model state immediately so the UI responds
    RUNNING_MODELS[model_name] = {
        "port": port,
        "process": None, # Will hold the ft serve process later
        "status": "downloading",
        "progress": "Initializing..."
    }
    
    # Run the download and start process in the background
    asyncio.create_task(download_and_start_model(model_name, port))
    
    return {"status": "started", "port": port}

async def download_and_start_model(model_name: str, port: int):
    try:
        actual_path = model_name
        
        # If not an existing local path, resolve it via Hugging Face Hub using the new 'hf' CLI
        if not os.path.exists(model_name):
            logger.info(f"Downloading {model_name} natively via huggingface_hub...")
            
            from huggingface_hub import snapshot_download
            from tqdm.auto import tqdm
            
            # Custom tqdm class to intercept progress updates
            class APITqdm(tqdm):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.is_main = kwargs.get('desc', '').startswith('Fetching')
                    
                def update(self, n=1):
                    super().update(n)
                    if self.is_main and model_name in RUNNING_MODELS:
                        if self.total and self.total > 0:
                            percent = int((self.n / self.total) * 100)
                            RUNNING_MODELS[model_name]["progress"] = f"Downloading: {percent}%"
                        else:
                            RUNNING_MODELS[model_name]["progress"] = f"Downloading: {self.n} files"

            # Run in a thread so we don't block the asyncio loop
            actual_path = await asyncio.to_thread(
                snapshot_download, 
                repo_id=model_name, 
                tqdm_class=APITqdm
            )

        if model_name not in RUNNING_MODELS:
            return # Was stopped during download
            
        RUNNING_MODELS[model_name]["status"] = "starting"
        RUNNING_MODELS[model_name]["progress"] = "Booting engine..."
        logger.info(f"Starting model {model_name} (path: {actual_path}) on port {port}")
        
        # Launch ft serve
        cmd = [
            "ft", "serve", 
            "--host", "127.0.0.1", 
            "--port", str(port), 
            "--model", actual_path,
            "--served-model-name", model_name
        ]
        
        # We use subprocess.Popen here so it matches the original design, 
        # but in an async function.
        ft_process = subprocess.Popen(cmd)
        
        if model_name in RUNNING_MODELS:
            RUNNING_MODELS[model_name]["process"] = ft_process
            RUNNING_MODELS[model_name]["progress"] = "Running"
            
    except Exception as e:
        logger.error(f"Error in background task for {model_name}: {e}")
        if model_name in RUNNING_MODELS:
            RUNNING_MODELS[model_name]["status"] = "failed"
            RUNNING_MODELS[model_name]["progress"] = f"Error: {str(e)}"

@app.post("/api/stop")
async def stop_model(request: Request):
    data = await request.json()
    model_name = data.get("model")
    
    if model_name in RUNNING_MODELS:
        logger.info(f"Stopping model {model_name}")
        RUNNING_MODELS[model_name]["process"].terminate()
        del RUNNING_MODELS[model_name]
        return {"status": "stopped"}
    raise HTTPException(status_code=404, detail="Model not running")

# OpenAI API Reverse Proxy
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy_openai(path: str, request: Request):
    body = b""
    model_name = None
    
    if request.method in ["POST", "PUT"]:
        body = await request.body()
        try:
            json_body = json.loads(body)
            model_name = json_body.get("model")
        except json.JSONDecodeError:
            pass
            
    # If the endpoint doesn't specify a model (like /v1/models), 
    # we can either query all running models and aggregate, or just pick the first one.
    if not model_name and RUNNING_MODELS:
        model_name = next(iter(RUNNING_MODELS))
        
    if not model_name or model_name not in RUNNING_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not running or not specified.")
        
    port = RUNNING_MODELS[model_name]["port"]
    target_url = f"http://127.0.0.1:{port}/v1/{path}"
    
    # Forward the request
    url = httpx.URL(target_url, query=request.url.query.encode("utf-8"))
    req = client.build_request(
        request.method,
        url,
        headers=request.headers.raw,
        content=body
    )
    
    response = await client.send(req, stream=True)
    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=response.headers,
        background=response.aclose
    )
