# AI
import tiktoken
from openai import AsyncOpenAI

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Form, Request
from fastapi.middleware.cors import CORSMiddleware

# Pydantic and Formating
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, Dict, Any, AsyncGenerator

# Other
import base64
import os
from contextlib import asynccontextmanager
import aiofiles
from io import BytesIO
import logging
import time
from datetime import datetime, timezone
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="LLM Router API",
    version="1.0.0",
    description="Smart routing for LLM requests based on query complexity and requirements"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RoutingDecision(BaseModel):
    """Decision about which LLM to use for the query."""

    model_config = ConfigDict(str_strip_whitespace=True)
    recommended_llm: Literal["gpt-4.1-nano", "gpt-4.1", "o4-mini"]

class EvaluationResponse(BaseModel):
    """Complete evaluation response containing analysis and routing decision."""

    routing_decision: RoutingDecision

class RouteRequest(BaseModel):
    """Request model for routing endpoint"""

    prompt: str = Field(..., description="The prompt to route")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key (can also be passed in header)")

class RouteResponse(BaseModel):
    """Response model for routing endpoint"""

    response: str
    selected_model: str
    file_processed: Optional[bool] = None
    file_processing_error: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for the FastAPI app
    """

    logger.info("Starting LLM Router API")
    yield

    logger.info("Cleaning up temporary files")
    for file in os.listdir(UPLOAD_DIR):
        try:
            os.remove(os.path.join(UPLOAD_DIR, file))
        except:
            pass

class RouteLLM:

    def __init__(self, query: str, temperature: float, client: AsyncOpenAI, file_path: Optional[str] = None):
        self.query = query
        self.temperature = temperature
        self.client = client
        self.file_path = file_path
        self.file_added = bool(file_path)
        self.file_type = self._get_file_type() if self.file_added else None
        self.xml_true = "<" in query and ">" in query
        self.uploaded_file_id = None
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.file_processed = False
        self.file_processing_error = None

    def _get_file_type(self) -> str:
        """
        Checks the file type and returns the type - only supports images and PDFs
        """

        if not self.file_path:
            return ""
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return "image"
        elif ext == '.pdf':
            return "pdf"
        else:
            return "unsupported"

    async def _encode_file(self) -> Optional[str]:
        """
        Encodes the file and returns the base64 string
        """

        if not self.file_path or not os.path.exists(self.file_path):
            return None
        
        async with aiofiles.open(self.file_path, "rb") as file:
            content = await file.read()
            return base64.b64encode(content).decode('utf-8')
    
    async def _upload_file(self) -> Optional[str]:
        """
        Upload file using Files API and return file ID
        """
        
        async with aiofiles.open(self.file_path, "rb") as f:
            content = await f.read()
            
        file_obj = BytesIO(content)
        file_obj.name = os.path.basename(self.file_path)
        
        file_upload = await self.client.files.create(
            file=file_obj,
            purpose="assistants"
        )
        
        return file_upload.id

    def format_evaluation_prompt(self) -> str:
       """
       Format the evaluation prompt
       """

       num_tokens = len(self.encoding.encode(self.query))

       system_prompt = f"""
        ### Context
        Evaluate the user query by emphasizing key parameters: task complexity, creativity level (`{self.temperature}`), required token count (`{num_tokens}`), file details (`{self.file_added}`, `{self.file_type}`)
        and complex queries (`{self.xml_true}`). These factors and cost are essential in guiding the model selection process effectively.

        ### Task
        Assign the Suitable AI Model for the Query:
        - **gpt-4.1-nano**: Ideal for tasks requiring minimal reasoning and creativity, perfect for straightforward, concise responses and summaries. Extremely cost-effective at $0.2/1M tokens. Never use with files.
        - **gpt-4.1**: Best for moderately complex tasks with a need for balanced creativity and reasoning. Handles coding, basic file interactions, and image processing efficiently. Priced at $2.00/1M tokens.
        - **o4-mini**: Suited for highly complex tasks involving deep reasoning and significant creativity. Excels in nuanced, structured analysis and comprehensive file and image processing. Available at $1.10/1M tokens.
       """

       return system_prompt

    async def get_evaluation_response(self) -> str:
        """
        Evaluate the user query and return the recommended model
        """

        response = await self.client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.format_evaluation_prompt()},
                {"role": "user", "content": self.query}
            ],
            temperature=0.2,
            response_format=EvaluationResponse
        )

        validated_response = response.choices[0].message.parsed
        return validated_response.routing_decision.recommended_llm

    async def _prepare_messages(self) -> list:
        """Prepare messages based on file type and availability"""

        if not self.file_path or not os.path.exists(self.file_path):
            if self.file_path:
                self.file_processing_error = "File not found or inaccessible"
            return [{"role": "user", "content": self.query}]

        if self.file_type == "image":
            base64_image = await self._encode_file()
            if base64_image:
                ext = os.path.splitext(self.file_path)[1][1:]
                self.file_processed = True
                return [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{ext};base64,{base64_image}"
                            }
                        }
                    ]
                }]
            else:
                self.file_processing_error = "Failed to encode image file"
                
        elif self.file_type == "pdf":
            if not self.uploaded_file_id:
                self.uploaded_file_id = await self._upload_file()
            
            self.file_processed = True
            return [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.query},
                    {
                        "type": "file",
                        "file": {
                            "file_id": self.uploaded_file_id
                        }
                    }
                ]
            }]
            
        elif self.file_type == "unsupported":
            self.file_processing_error = f"Unsupported file type: {os.path.splitext(self.file_path)[1]}"
        else:
            self.file_processing_error = "Unknown file type"
            
        return [{"role": "user", "content": self.query}]

    async def get_gpt_response(self, model_name: str) -> str:
        """Gets response from the routed model"""

        messages = await self._prepare_messages()
        
        params = {
            "model": model_name,
            "messages": messages,
        }
        
        if not model_name.startswith("o4"):
            params["temperature"] = self.temperature
        
        response = await self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    async def orchestrate_response(self) -> Dict[str, Any]:
        """
        Based on the evaluation response, get the response from the routed model
        """

        start_time = time.time()
        
        recommended_llm = await self.get_evaluation_response()
        routing_time = time.time() - start_time

        response_start = time.time()
        response = await self.get_gpt_response(recommended_llm)
        response_time = time.time() - response_start

        return {
            "response": response,
            "selected_model": recommended_llm,
            "file_processed": self.file_processed,
            "file_processing_error": self.file_processing_error,
            "routing_metadata": {
                "routing_time_ms": routing_time * 1000,
                "response_time_ms": response_time * 1000,
                "temperature": self.temperature,
                "has_file": self.file_added,
                "file_type": self.file_type,
                "has_xml": self.xml_true,
                "num_tokens": len(self.encoding.encode(self.query))
            },
            "execution_times_ms": (time.time() - start_time) * 1000
        }

def get_api_key(request_key: Optional[str], header_key: Optional[str]) -> str:
    """Extract API key from request or header"""
    
    api_key = request_key or header_key
    if not api_key:
        raise HTTPException(status_code=400, 
                            detail="OpenAI API key must be provided in request or x-openai-api-key header")
    return api_key

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""

    return {
        "name": "LLM Router API",
        "version": "1.0.0",
        "description": "Smart routing for LLM requests",
        "endpoints": {
            "/route": "POST - Route a text prompt to the optimal LLM",
            "/route/with-file": "POST - Route a prompt with an attached file",
            "/health": "GET - Health check endpoint"
        }
    }

@app.post("/route", response_model=RouteResponse)
async def route_prompt(request: RouteRequest, x_openai_api_key: Optional[str] = Header(default=None, description="OpenAI API key for authentication")) -> RouteResponse:
    """Route a text prompt to the optimal LLM"""

    api_key = get_api_key(request.openai_api_key, x_openai_api_key)
    
    try:
        client = AsyncOpenAI(api_key=api_key)
        router = RouteLLM(request.prompt, request.temperature, client)
        result = await router.orchestrate_response()
        return RouteResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in route_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/route/with-file", response_model=RouteResponse)
async def route_with_file(
    prompt: str = Form(..., description="The prompt to route"),
    temperature: float = Form(default=0.7, description="Temperature for generation"),
    openai_api_key: Optional[str] = Form(default=None, description="OpenAI API key"),
    file: UploadFile = File(..., description="File to process with the prompt"),
    x_openai_api_key: Optional[str] = Header(default=None, description="OpenAI API key")
) -> RouteResponse:
    """Route a prompt with an attached file to the optimal LLM"""

    api_key = get_api_key(openai_api_key, x_openai_api_key)
    
    if file.size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, 
                            detail="File size must be less than 2MB")
    
    file_path = os.path.join(UPLOAD_DIR, f"{time.time()}_{file.filename}")

    try:
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        
        client = AsyncOpenAI(api_key=api_key)
        router = RouteLLM(prompt, temperature, client, file_path)
        result = await router.orchestrate_response()

        return RouteResponse(**result)

    except Exception as e:
        logger.error(f"Error in route_with_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

@app.get("/models")
async def get_models() -> Dict[str, Any]:
    """Get information about available models"""

    return {
        "models": {
            "gpt-4.1-nano": {
                "description": "Fast and cost-effective for simple tasks",
                "cost_per_million_tokens": 0.2,
                "supports_files": False,
                "supports_temperature": True,
                "best_for": ["simple queries", "definitions", "summaries", "basic Q&A"]
            },
            "gpt-4.1": {
                "description": "Balanced model for most tasks",
                "cost_per_million_tokens": 2.0,
                "supports_files": True,
                "supported_file_types": ["images", "pdf"],
                "supports_temperature": True,
                "best_for": ["coding", "analysis", "file processing", "image understanding"]
            },
            "o4-mini": {
                "description": "Best for complex reasoning and analysis",
                "cost_per_million_tokens": 1.1,
                "supports_files": True,
                "supported_file_types": ["images", "pdf"],
                "supports_temperature": False,
                "best_for": ["complex reasoning", "deep analysis", "nuanced tasks"]
            }
        },
        "file_support": {
            "supported_extensions": {
                "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
                "pdf": [".pdf"]
            },
            "max_file_size": "20MB",
            "processing_notes": [
                "Images are base64 encoded and sent to vision-capable models",
                "PDFs are uploaded using OpenAI Files API",
                "Only images and PDFs are supported - all other file types will be rejected"
            ]
        }
    }

@app.middleware("http")
async def log_requests(request: Request, call_next) -> Any:
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

if __name__ == "__main__":
    
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info",
        "access_log": True,
        "reload": False,  
        "workers": 1,
    }
    
    logger.info("Starting LLM Router API with enhanced configuration")
    uvicorn.run(app, **config)