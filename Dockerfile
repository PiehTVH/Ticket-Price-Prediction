FROM python:3-slim-bullseye

WORKDIR /app

COPY requirements.txt ./

RUN pip install --default-timeout=350 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000 

CMD ["flask", "run", "--host", "0.0.0.0"]