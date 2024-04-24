#!/bin/bash

# Set up Python 3 environment for JupyterHub
sudo python3 -m pip install numpy pandas langchain pypdf2 langchain-community faiss-cpu boto3 sentence_transformers==2.2.2 openai==0.28.1

# Assuming PySpark is already part of EMR, just ensure PYTHONPATH is set 
echo "export PYTHONPATH=\$SPARK_HOME/python:\$PYTHONPATH" >> /etc/profile
echo "export PYSPARK_PYTHON=python3" >> /etc/profile

# Reload profile to apply changes
source /etc/profile