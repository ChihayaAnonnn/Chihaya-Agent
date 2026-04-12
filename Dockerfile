FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ agent/
COPY api/ api/
COPY bus/ bus/
COPY cli/ cli/
COPY cron/ cron/
COPY eval/ eval/
COPY heartbeat/ heartbeat/
COPY providers/ providers/
COPY session/ session/
COPY utils.py .
COPY context_file_template/ context_file_template/

# Make source packages importable without an editable install
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
