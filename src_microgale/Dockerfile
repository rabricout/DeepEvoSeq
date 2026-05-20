# Choose a base image 
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

# Set working directory inside container
WORKDIR /home

# Copy your application code to the container
COPY . /home

# Install dependencies (assuming you have requirements.txt)
RUN apt-get update && apt-get install -y python3 python3-pip
#RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run your app
#CMD ["python3", "test.py"]
