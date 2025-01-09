# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Copy local code to the container image.
WORKDIR /app
COPY requirements.txt .

# Install production dependencies.
# RUN apt update && apt install -y git
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development 

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]