FROM python:3.10

WORKDIR /app

COPY requirements2.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "inference.py"]