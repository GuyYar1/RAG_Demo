# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libatlas-base-dev gfortran

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Copy the requirements file
COPY requirements.txt ./
COPY app.py ./
COPY model_service.py ./
COPY data_loader.py ./
COPY text_generator.py ./
COPY TMDB_tv_dataset_v3.csv ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Optionally, add a command to keep the container running for debugging
#CMD ["bash"]
# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]

