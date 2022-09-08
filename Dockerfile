# lightweight python

FROM python:3.7-slim
RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "python", "app.py" , "--host=0.0.0.0", "--port=5000"]