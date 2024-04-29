# Use an official Python runtime as a parent image
FROM openjdk:8-jre-slim

# Set environment variables
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV PYSPARK_PYTHON="/opt/miniconda3/bin/python"

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y curl bzip2 wget unzip --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and set up Python environment
RUN curl -s -L --url "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -f -p "/opt/miniconda3" && \
    rm /tmp/miniconda.sh && \
    conda config --set auto_update_conda true && \
    conda config --set channel_priority false && \
    conda update conda -y --force-reinstall && \
    conda clean -tipy && \
    echo "PATH=/opt/miniconda3/bin:\${PATH}" > /etc/profile.d/miniconda.sh

# Install required Python packages
RUN pip install --no-cache pyspark==3.5.0 numpy pandas awscli

# Set SPARK_HOME environment variable
ENV SPARK_HOME=/opt/spark

# Download and extract Spark
WORKDIR /opt
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz" \
    && tar -xf apache-spark.tgz \
    && rm apache-spark.tgz \
    && ln -s spark-3.5.0-bin-hadoop3 spark

# Download AWS SDK JARs for S3 integration
RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.8.0/aws-java-sdk-1.8.0.jar -P /opt/spark/jars/
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.0.0/hadoop-aws-3.0.0.jar -P /opt/spark/jars/

# Copy necessary files into the container
COPY prediction.py /opt/
COPY ValidationDataset.csv /opt/
COPY trainedmodel /opt/trainedmodel/

# Set the entry point and default command
CMD ["spark-submit", "/opt/prediction.py", "/opt/ValidationDataset.csv"]
