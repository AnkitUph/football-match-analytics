FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Several CV-related packages (ultralytics, supervision, easyocr, and
# likely any future ones) list plain `opencv-python` as a dependency,
# which silently overwrites opencv-python-headless if it installs
# afterward - opencv-python needs GUI libraries (libxcb, etc.) that
# don't exist in this slim/headless container, breaking `cv2` entirely.
# Forcing headless back in AFTER all of requirements.txt is installed
# guarantees it wins regardless of what pulled in the GUI version, so
# this doesn't need to be manually re-run every time a new dependency
# is added.
RUN pip uninstall -y opencv-python && pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]