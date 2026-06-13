FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Bring in the whole repo (models, CSVs, vector stores, src/, api/)
COPY . .

ENV PORT=8000
EXPOSE 8000

# Render/most hosts inject $PORT at runtime
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
