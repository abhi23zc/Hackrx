# 🚀 HackRx PDF RAG API

A production-ready FastAPI service that processes PDF documents and answers questions using Retrieval-Augmented Generation (RAG) with Google's Gemini AI.

## ✨ Features

- 🔍 **PDF Processing**: Automatic extraction and chunking of PDF documents
- 🧠 **AI-Powered Q&A**: Answers questions using Google's Gemini Pro 2.5
- 📊 **Vector Search**: Fast similarity search using FAISS with sentence embeddings
- 🔐 **Authentication**: Secure Bearer token authentication
- ⚡ **Fast Response**: Sub-30 second response times
- 🌐 **Public API**: HTTPS-ready for production deployment

## 📋 API Endpoints

### POST /hackrx/run
Process a PDF document and answer questions.

**Authentication**: Bearer Token Required

**Request Format**:
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

**Response Format**:
```json
{
  "answers": [
    "A grace period of thirty days is provided...",
    "There is a waiting period of thirty-six months..."
  ]
}
```

### GET /health
Health check endpoint for monitoring.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- 4GB+ RAM recommended

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd hackrx-pdf-rag-api
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\\Scripts\\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file:
```bash
# API Configuration
AUTH_TOKEN=2931609bd36ec1a45cb577b3b831dc711c76ae157b3c6250c564284c93b062ff


# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

### 5. Run the Application

#### Local Development
```bash
# Run with auto-reload
python fastapi_app.py

# Or with uvicorn
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

#### Production with HTTPS
```bash
# With SSL certificates
uvicorn fastapi_app:app --host 0.0.0.0 --port 443 \
  --ssl-keyfile=path/to/key.pem \
  --ssl-certfile=path/to/cert.pem
```

## 🚀 Deployment Options

### Option 1: Render (Recommended)
1. Create account at [render.com](https://render.com)
2. Connect GitHub repository
3. Use `render.yaml` configuration
4. Deploy automatically

### Option 2: Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway up
```

### Option 3: Heroku
```bash
# Install Heroku CLI
heroku create hackrx-pdf-rag-api
git push heroku main
```

### Option 4: Docker
```bash
# Build image
docker build -t hackrx-pdf-rag .

# Run container
docker run -p 8000:8000 hackrx-pdf-rag
```

## 📖 Usage Examples

### 1. Testing with curl
```bash
curl -X POST "http://localhost:8000/hackrx/run" \\
  -H "Authorization: Bearer hackrx-2024-secure-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
```

### 2. Python Client
```python
import requests

API_URL = "https://your-domain.com/hackrx/run"
API_KEY = "hackrx-2024-secure-key"

response = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "documents": "https://example.com/document.pdf",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered?",
            "What is the No Claim Discount (NCD) offered?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges?"
        ]
    }
)

print(response.json())
```

### 3. JavaScript/Node.js
```javascript
const axios = require('axios');

async function processDocument() {
  const response = await axios.post('https://your-domain.com/hackrx/run', {
    documents: 'https://example.com/document.pdf',
    questions: [
      'What is the grace period for premium payment?',
      'What is the waiting period for pre-existing diseases?'
    ]
  }, {
    headers: {
      'Authorization': 'Bearer hackrx-2024-secure-key',
      'Content-Type': 'application/json'
    }
  });
  
  console.log(response.data);
}

processDocument();
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | Bearer token for authentication | `hackrx-2024-secure-key` |
| `GEMINI_API_KEY` | Google Gemini API key | Provided |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Debug mode | `False` |

### Performance Tuning
- **Memory**: Ensure 4GB+ RAM for large PDFs
- **Timeout**: Set to 30 seconds for optimal performance
- **Concurrency**: Supports multiple simultaneous requests

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest test_api.py -v
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load test
locust -f load_test.py --host=http://localhost:8000
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Logging
- Logs are written to stdout
- Use `docker logs` for containerized deployments
- Integrate with monitoring tools like Datadog or New Relic

## 🔍 Troubleshooting

### Common Issues

1. **PDF Download Fails**
   - Check URL accessibility
   - Verify file format (PDF only)
   - Ensure HTTPS URLs for production

2. **Slow Response Times**
   - Increase server resources
   - Optimize PDF size (< 10MB recommended)
   - Check network connectivity

3. **Authentication Errors**
   - Verify Bearer token format
   - Check API key in environment variables

4. **Memory Issues**
   - Monitor RAM usage during processing
   - Consider PDF size limits
   - Use streaming for large files

### Debug Mode
```bash
# Enable debug logging
DEBUG=True python fastapi_app.py
```

## 🏗️ Architecture