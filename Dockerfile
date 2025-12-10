# Use official Python image
FROM python:3.11-slim


# Install build tools and curl for uv installation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY . /app
COPY ./src /utils
# Copy only the needed folders from data
COPY data/cache ./data/cache
COPY data/encoders ./data/encoders
COPY data/models ./data/models

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
ENV PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
