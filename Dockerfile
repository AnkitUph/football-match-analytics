FROM ubuntu:22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    gpg-agent \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Internal venv, lives inside the image only — not on your host, not in
# your project folder. Adding it to PATH means every `python`/`pip`
# command below (and at container runtime) transparently uses it.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Intel's official GPU compute runtime repo.
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
    tee /etc/apt/sources.list.d/intel-gpu-jammy.list && \
    apt-get update && \
    apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

#RUN pip install --no-cache-dir --no-build-isolation \
#    git+https://github.com/KaiyangZhou/deep-person-reid.git

RUN pip uninstall -y opencv-python opencv-python-headless || true

RUN pip install --no-cache-dir opencv-python-headless

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]