FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY app.py .
COPY ui.py .
COPY octmnist_cnn_weights.pth .

# Exponer los puertos correspondientes (8000 para FastAPI, 8501 para Streamlit)
EXPOSE 8000
EXPOSE 8501

# Crear un script de arranque para iniciar ambos servicios en el mismo contenedor
RUN echo '#!/bin/bash\nuvicorn app:app --host 0.0.0.0 --port 8000 & \nstreamlit run ui.py --server.port 8501 --server.address 0.0.0.0\n' > start.sh
RUN chmod +x start.sh

CMD ["./start.sh"]