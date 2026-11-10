FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir openai

RUN pip install --no-cache-dir openjiuwen

COPY run_agent.py /app/run_agent.py

ENTRYPOINT ["python", "/app/run_agent.py"]