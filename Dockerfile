# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app/FlaskAppRepo

# Copy the requirements file to the container
COPY requirements.txt .

# Install the project dependencies
RUN pip install -r requirements.txt

# Uninstall and install cv2
RUN pip3 install opencv-contrib-python==4.5.5.62

# Copy the rest of the application code to the container
COPY . .

# Expose the port on which your Flask application runs
EXPOSE 5000

# Set the environment variable for the Flask application
ENV FLASK_APP=webapp.py

# Set the entry point command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
