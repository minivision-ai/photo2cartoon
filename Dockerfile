FROM python:3.6
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["gunicorn", "app:app", "-c", "./gunicorn.conf.py"]