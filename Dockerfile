FROM apache/airflow:2.7.3-python3.11

# Airflow explicitly switches to non-root user for security. 
# Switch to root to install additional system packages
USER root

### Install Java and bash for Spark ###
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # Create expected JAVA_HOME directory and symlink the java binary there (only if missing)
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    [ -f /usr/lib/jvm/java-17-openjdk-amd64/bin/java ] || ln -s "$(which java)" /usr/lib/jvm/java-17-openjdk-amd64/bin/java

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
###

# Switch back to non-root user for python dependencies
USER airflow

# Environment variables for development
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

VOLUME /app

COPY . .