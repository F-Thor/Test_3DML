FROM python:3.8
MAINTAINER Pavel Kolichko <fthorror@gmail.com>

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY . .

CMD python3 src/main.py