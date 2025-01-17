FROM registry.redhat.io/ubi9/python-311:9.5-1736353526

# listen on port 8080
EXPOSE 8080/tcp

# Set the working directory in the container
WORKDIR /projects

# Copy the dependencies file to the working directory
COPY requirements.txt.docker ./requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY ./src/app.py .

# Copy AI Models into container
COPY ./whisper_onnx_tiny_en_fp32_e2e.onnx .

# Environment variable for number of workers
ENV NUM_WORKERS "10"

# Specify the command to run on container start
CMD fastapi run app.py --host 0.0.0.0 --port 8080 --no-reload --workers $NUM_WORKERS
