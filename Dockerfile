FROM ghcr.io/jpbascur/tsm-base:latest

WORKDIR /app

# Copy app code only — deps and model are in the base image
COPY . .

EXPOSE 7860

# Cloud Run sets PORT=8080, HF Spaces expects 7860 (the default here)
ENV PORT=7860

CMD streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
