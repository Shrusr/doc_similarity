FROM python:3.7-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt 
ENV PYTHONPATH=/app
EXPOSE 5000 
CMD ["python","app.py"]
