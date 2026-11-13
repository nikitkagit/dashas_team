FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY pyproject.toml .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .


COPY . .

RUN rm -rf /app/logs && ln -s /tmp /app/logs
ENV PYTHONPATH=/app/src

CMD ["python", "main.py", "--tasks-dir", "/data/tasks"]