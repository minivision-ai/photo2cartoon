FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
FROM python:3.6
WORKDIR /app

COPY requirements.txt ./
# RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip \ && apt-get install -y --fix-missing\ && pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["gunicorn", "app:app", "-c", "./gunicorn.conf.py"]