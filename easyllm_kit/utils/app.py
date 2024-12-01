from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
import requests
from easyllm_kit.utils import get_logger
from easyllm_kit.models import LLM
from easyllm_kit.configs import Config
import uvicorn

logger = get_logger('easyllm_kit')

class _GenerateRequest(BaseModel):
    """Internal request model for text generation."""
    prompts: Union[str, List[str]]
    image_dir: Optional[Union[str, List[str]]] = None
    image_format: Optional[str] = "base64"

class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: Union[str, List[str]]
    status: str = "success"
    error: Optional[str] = None

def create_app(config_dir: str, context_prompt: Optional[str] = None) -> FastAPI:
    """
    Create a FastAPI application that serves the LLM model.

    Args:
        config_dir (str): The directory containing the model configuration files.

    Returns:
        FastAPI: The FastAPI application.
    """
    app = FastAPI(
        title="EasyLLM_kit API",
        description="API for text generation using LLM models supported by EasyLLM_kit",
        version="1.0.0"
    )

    config = Config.build_from_yaml_file(config_dir)
    model = LLM.build_from_config(config)

    @app.get("/")
    async def root():
        """Root endpoint that returns basic information about the service."""
        return {
            "status": "active",
            "model_name": model.model_name,
            "description": "EasyLLM_kit text generation service"
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: _GenerateRequest) -> GenerateResponse:
        """Generate text based on the input prompts."""
        try:
            generation_params = {}
            if request.image_dir is not None:
                generation_params['image_dir'] = request.image_dir
                generation_params['image_format'] = request.image_format
            
            generated_text = model.generate(
                prompts=request.prompts,
                **generation_params
            )

            return GenerateResponse(
                generated_text=generated_text,
                status="success"
            )

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}"
            )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app

def make_request(url: str, request_data: Dict) -> Dict:
    """
    Make a request to the LLM service.

    Args:
        url (str): The base URL of the LLM service (e.g., "http://localhost:8000")
        request_data (Dict): Dictionary containing the request parameters:
            - prompts (str or List[str]): The text prompt(s) for generation
            - image_dir (str or List[str], optional): Path(s) to image(s) or base64 strings
            - image_format (str, optional): Format of the images ('base64' or 'path')

    Returns:
        Dict: The response containing:
            - generated_text (str or List[str]): The generated text
            - status (str): Status of the request
            - error (str, optional): Error message if any

    Example:
        >>> request_data = {
        ...     "prompts": "Tell me a story",
        ...     "image_dir": None
        ... }
        >>> response = make_request("http://localhost:8000", request_data)
        >>> print(response["generated_text"])
    """
    try:
        # Ensure the URL ends with /generate
        if not url.endswith('/generate'):
            url = url.rstrip('/') + '/generate'

        # Handle image encoding if present
        if request_data.get('image_dir'):
            import base64
            from easyllm_kit.utils import read_image_as_bytes
            
            # Handle single image or list of images
            if isinstance(request_data['image_dir'], str):
                image_data = read_image_as_bytes(request_data['image_dir'])
                request_data['image_dir'] = base64.b64encode(image_data).decode('utf-8')
            elif isinstance(request_data['image_dir'], list):
                encoded_images = []
                for img_path in request_data['image_dir']:
                    image_data = read_image_as_bytes(img_path)
                    encoded_images.append(base64.b64encode(image_data).decode('utf-8'))
                request_data['image_dir'] = encoded_images

            request_data['image_format'] = 'base64'

        # Make the POST request
        response = requests.post(
            url,
            json=request_data,
            timeout=30
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        return response.json()
            
    except requests.RequestException as e:
        error_msg = f"Request to {url} failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

def run_app(app: FastAPI, host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI application using uvicorn."""
    uvicorn.run(app, host=host, port=port)

# Example usage:
"""
# Making a request
request_data = {
    "prompts": "Tell me a story",
    "image_dir": None
}

try:
    response = make_request("http://localhost:8000", request_data)
    print(f"Generated text: {response['generated_text']}")
except HTTPException as e:
    print(f"Error: {e.detail}")
"""