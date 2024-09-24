# chatbot_api/Dockerfile

FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the source code to /app in the container
COPY ./src /app/
COPY ./pdfs /app/pdfs

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

# Make sure entrypoint.sh is executable
# RUN chmod +x /app/src/entrypoint.sh

# Expose port 8000 for the API
EXPOSE 8000

# Set the entrypoint script to run
CMD ["sh", "/app/entrypoint.sh"]