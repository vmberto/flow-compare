# Usar uma imagem base oficial do Python
FROM python:3.9-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Instalar dependências de sistema necessárias para compilar numpy e h5py
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean

COPY . .

# Atualizar pip, instalar cython primeiro e depois as dependências
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install cython && \
    python -m pip install -r requirements.txt

# Copiar o código da aplicação para dentro do container
COPY . .

# Definir o comando para rodar a aplicação
CMD ["python", "main.py"]