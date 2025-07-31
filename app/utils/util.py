
import re
import json

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def load_config():
    """
    Load configuration from config.json.
    """
    with open('config/config.json', 'r') as file:
        return json.load(file)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verify the API key from Bearer token
    """
    config = load_config()
    if credentials.credentials != config.get("VALID_API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return credentials.credentials


def sanitize_filename(file_link: str) -> str:
    """
    Sanitize file link to create a safe filename.
    """
    return re.sub(r'[^a-zA-Z0-9_-]', '_', file_link)