FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY constraints.txt visualize_app.py ./

# Only the dependencies of the visualization app, pinned to known-good versions
RUN pip install --no-cache-dir -c constraints.txt \
  "streamlit>=1.28.0" "plotly>=5.14.0" "pandas>=1.3.0" "universal-ml-utils>=0.1.5"

# Mount the WDQL data dir at /app/data, so the app's default
# --dataset-dir (data/<release>/organic-generations-dataset) resolves;
# pass another release via `docker run ... -- --dataset-dir data/<release>/...`
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "visualize_app.py", \
  "--server.port", "8501", \
  "--server.headless", "true", \
  "--server.enableCORS", "false", \
  "--server.enableXsrfProtection", "false", \
  "--browser.gatherUsageStats", "false"]
