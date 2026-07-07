# TokenOptimizer proxy + cockpit (CPU container).
# The local model runs in a SEPARATE ROCm container (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web ./web
COPY run.py .

ENV TOKENOPT_MODE=live PORT=4321 HOST=0.0.0.0
EXPOSE 4321
CMD ["python", "run.py"]
