FROM python:3.10
WORKDIR /model
COPY lab3/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python", "lab3/main.py"]
