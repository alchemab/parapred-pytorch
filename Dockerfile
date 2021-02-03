FROM python:3.8-slim-buster

# Get pip and get build tools
RUN apt-get update -y && apt-get install -y \
    python3-pip

# Copy and install
COPY . /data
RUN cd /data && pip install -r requirements.txt && make install

ENTRYPOINT ["python", "/data/cli.py"]