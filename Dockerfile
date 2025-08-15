# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY . .

# Set environment variable for Python to run in unbuffered mode
ENV PYTHONUNBUFFERED 1

# Command to run the application when the container starts
# This will execute your predict.py script
CMD ["python", "predict.py"]