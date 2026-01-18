FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
        portaudio19-dev \
            && rm -rf /var/lib/apt/lists/*

            # Copy requirements
            COPY requirements.txt .

            # Install Python dependencies
            RUN pip install --no-cache-dir -r requirements.txt

            # Copy source code
            COPY src/ ./src/
            COPY config/ ./config/

            # Create trade logs directory
            RUN mkdir -p trade_logs

            # Expose metrics port
            EXPOSE 8080

            # Run the application
            CMD ["python", "-m", "src.main_controller"]
