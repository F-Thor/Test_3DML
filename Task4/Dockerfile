FROM python:3.6-stretch
MAINTAINER Pavel Kolichko <fthorror@gmail.com>

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY . .

CMD python3 src/main.py
