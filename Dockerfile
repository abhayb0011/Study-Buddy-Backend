# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your app
COPY . /app

# Download the model at build time (optional: remove if model is downloaded at runtime)
# RUN python -c "import gdown; gdown.download('', 'studdybuddy_model.h5', quiet=False)"

# Expose port
EXPOSE 8080

# Start the server using Waitress
CMD ["python", "app.py"]
