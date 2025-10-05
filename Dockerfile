FROM python:3.12-slim

WORKDIR /app

### Install Java and bash for Spark ###
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jdk-headless bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # Create expected JAVA_HOME directory and symlink the java binary there (only if missing)
    mkdir -p /usr/lib/jvm/java-21-openjdk-amd64/bin && \
    [ -f /usr/lib/jvm/java-21-openjdk-amd64/bin/java ] || ln -s "$(which java)" /usr/lib/jvm/java-21-openjdk-amd64/bin/java
# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
###

# Environment variables for development
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes

COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

VOLUME /app

COPY . .

# CMD ["tail", "-f", "/dev/null"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]