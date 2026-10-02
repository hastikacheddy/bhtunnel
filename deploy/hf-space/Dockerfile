# Hugging Face Space (Docker SDK) for the bhtunnel Streamlit demo.
# Builds straight from the GitHub repo, so redeploying = "Factory rebuild".
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

ARG REPO=https://github.com/hastikacheddy/bhtunnel.git
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO} . \
    && pip install --no-cache-dir --user -e ".[quantum,app]"

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
