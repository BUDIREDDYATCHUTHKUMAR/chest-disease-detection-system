# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (Minimal)
# Removed libgl1-mesa-glx and libglib2.0-0 as we removed opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Expects build context to be project root
# Install ML dependencies (Heavy - Cached)
COPY backend/requirements-torch.txt /app/requirements-torch.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=2000 --retries 10 -r requirements-torch.txt

# Install App dependencies (Light - Frequent Changes)
COPY backend/requirements-app.txt /app/requirements-app.txt
RUN pip install --no-cache-dir --default-timeout=2000 -r requirements-app.txt

# Copy project files
COPY backend /app/backend
# Validation: Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
