# Use Python 3.11.9 slim as the base image
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install required Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port (if needed)
EXPOSE 8080

# Run the application (you can specify any of the scripts here, for example satisfaction_analysis.py)
CMD ["python", "notebooks/satisfaction_analysis.ipynb"]
