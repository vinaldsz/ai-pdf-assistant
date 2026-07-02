FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.45.1 httpx==0.27.2

COPY streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
