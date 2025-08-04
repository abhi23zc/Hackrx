# MongoDB Integration Setup

This document explains how to set up and use the MongoDB integration for saving question-answer pairs.

## Overview

The MongoDB integration allows you to save question-answer pairs to a MongoDB database with the file URL hash as the document ID. This enables persistent storage and retrieval of QA data.

## Prerequisites

1. **MongoDB Server**: You need a MongoDB server running locally or remotely
2. **Python Dependencies**: The `pymongo` package is already added to `requirements.txt`

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure MongoDB Connection

Set up environment variables for MongoDB connection. You can either:

**Option A: Use environment variables**
Create a `.env` file in your project root:

```env
MONGO_USERNAME=your_mongodb_username
MONGO_PASSWORD=your_mongodb_password
MONGO_DB_NAME=hackrx_qa
```

**Option B: Modify the code directly**
Update the MongoDB connection parameters in `fastapi_app.py`:

```python
mongo_username = os.getenv("MONGO_USERNAME", "your_username")
mongo_password = os.getenv("MONGO_PASSWORD", "your_password")
mongo_db_name = os.getenv("MONGO_DB_NAME", "hackrx_qa")
```

### 3. Start MongoDB Server

If running locally, start MongoDB:

```bash
# On Windows
mongod

# On macOS/Linux
sudo systemctl start mongod
```

## API Endpoint

### Save QA Pairs Endpoint

**URL**: `POST /api/v1/hackrx/save-qa`

**Headers**:
```
Authorization: Bearer 2931609bd36ec1a45cb577b3b831dc711c76ae157b3c6250c564284c93b062ff
Content-Type: application/json
```

**Request Body**:
```json
{
  "url": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic?",
    "What are the key findings?"
  ],
  "answers": [
    "The main topic is artificial intelligence.",
    "The key findings include improved accuracy."
  ]
}
```

**Response**:
```json
{
  "success": true,
  "message": "Question-answer pairs saved successfully",
  "document_id": "a1b2c3d4e5f6..."
}
```

## Usage Examples

### 1. Using the Example Script

Run the provided example script:

```bash
python example_qa_save.py
```

### 2. Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/save-qa" \
  -H "Authorization: Bearer 2931609bd36ec1a45cb577b3b831dc711c76ae157b3c6250c564284c93b062ff" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "questions": ["What is the topic?"],
    "answers": ["AI and healthcare"]
  }'
```

### 3. Using Python Requests

```python
import requests

payload = {
    "url": "https://example.com/document.pdf",
    "questions": ["What is the topic?"],
    "answers": ["AI and healthcare"]
}

headers = {
    "Authorization": "Bearer 2931609bd36ec1a45cb577b3b831dc711c76ae157b3c6250c564284c93b062ff",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/api/v1/hackrx/save-qa",
    json=payload,
    headers=headers
)

print(response.json())
```

## Database Schema

The QA pairs are stored in MongoDB with the following structure:

```json
{
  "_id": "hash_of_file_url",
  "url": "https://example.com/document.pdf",
  "questions": ["Question 1", "Question 2"],
  "answers": ["Answer 1", "Answer 2"],
  "created_at": 1640995200.0,
  "question_count": 2
}
```

### Field Descriptions

- `_id`: SHA256 hash of the file URL (used as document ID)
- `url`: Original file URL
- `questions`: Array of questions
- `answers`: Array of corresponding answers
- `created_at`: Unix timestamp when the document was created/updated
- `question_count`: Number of questions in the document

## Error Handling

The endpoint includes comprehensive error handling:

- **400 Bad Request**: When questions and answers arrays have different lengths
- **401 Unauthorized**: Invalid API key
- **500 Internal Server Error**: MongoDB connection issues or other server errors

## Security Considerations

1. **API Key**: Always use the provided API key for authentication
2. **MongoDB Authentication**: Use strong passwords for MongoDB users
3. **Network Security**: Consider using MongoDB with SSL/TLS for production
4. **Environment Variables**: Store sensitive credentials in environment variables

## Troubleshooting

### Common Issues

1. **Connection Failed**: Check if MongoDB server is running and credentials are correct
2. **Authentication Error**: Verify MongoDB username and password
3. **Database Not Found**: Ensure the database name is correct and the user has access

### Debug Mode

Enable debug logging by setting the log level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Existing Pipeline

The new endpoint uses the same hash function (`hash_filelink`) as the main RAG pipeline, ensuring consistency in document identification across the system.

You can now:
1. Process documents through the main RAG pipeline
2. Save the Q&A results to MongoDB for future reference
3. Retrieve stored Q&A pairs using the document hash 