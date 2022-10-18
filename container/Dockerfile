FROM python:3.9

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./app/ /app/

COPY ./app/data /app/data/

RUN chmod +x /app/train.py
RUN chmod +x /app/predict.py

ENTRYPOINT ["python3", "-m", "app.predict"]
