# Use a lightweight Python image
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Set workdir
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 7860

# Run FastAPI on Spaces' default port (7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
