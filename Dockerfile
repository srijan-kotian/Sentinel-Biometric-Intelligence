FROM python:3.11-slim

# Install system dependencies + math libraries for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Senior Debugger Move: Set Single-Threaded mode to prevent 8GB RAM spike
ENV MAKEFLAGS="-j1"

# Install dlib separately first to ensure it succeeds under the RAM limit
RUN pip install --no-cache-dir dlib==19.24.1

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]