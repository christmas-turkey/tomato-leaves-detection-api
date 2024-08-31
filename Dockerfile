FROM python:3.11

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

# Install YOLO weights
RUN gdown --fuzzy https://drive.google.com/file/d/1cOClJvYkBxURCa3o4FB83fY-QjknNZAR/view?usp=drive_link

EXPOSE 5000

CMD ["python", "-m", "src.server"]