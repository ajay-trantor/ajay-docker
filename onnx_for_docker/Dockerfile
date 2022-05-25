FROM python:3.8

ADD app.py .

ADD 1.png .

ADD klite.onnx .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

ADD requirements.txt .

RUN pip install -r requirements.txt

CMD ["python","./app.py"]
