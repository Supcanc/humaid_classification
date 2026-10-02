FROM python:3.14.4-slim

WORKDIR /app/

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY config.py /app/config.py

EXPOSE 8501

CMD [ "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501" ]