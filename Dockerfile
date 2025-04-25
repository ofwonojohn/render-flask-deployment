# Use Python base image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app


# Copy the local project files into the container
COPY . /app

# Copy requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
