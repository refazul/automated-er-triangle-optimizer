# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Fix for Debian slim images: Create manual directories so the Java installer doesn't crash
RUN mkdir -p /usr/share/man/man1 /usr/share/man/man2

# Install the default JRE (This handles the missing installation candidate error)
RUN apt-get update && \
    apt-get install -y default-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Environment Variables (Updated to point to the default Java path)
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PYSPARK_PYTHON=python3

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dataset CSVs and the Python script into the container
COPY . /app

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the web app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]