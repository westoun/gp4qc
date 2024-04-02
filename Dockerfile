FROM python:3.11.3-slim

WORKDIR /app 

COPY requirements.txt ./

RUN buildDeps='gcc ' \
    && apt-get update \
    && apt-get -y install git \
    && apt-get install -y $buildDeps\
    && pip3 install -r requirements.txt --no-cache-dir \
    && apt-get purge -y --auto-remove $buildDeps

COPY . . 

ENTRYPOINT [ "python3", "main.py" ]