FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SPECTER2 model and adapter into the image
COPY download_model.py .
RUN python download_model.py

# Copy app code
COPY . .

EXPOSE 7860

# Cloud Run sets PORT=8080, HF Spaces expects 7860 (the default here)
ENV PORT=7860

CMD streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
