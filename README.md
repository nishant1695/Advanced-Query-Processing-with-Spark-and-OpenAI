# Advanced-Query-Processing-with-Spark-and-OpenAI

## Introduction
This project showcases the implementation of a big data system utilizing a Retrieval-Augmented Generation-based Large Language Model.

## About the Dataset

[arXiv](https://arxiv.org/) is a free distribution service and an open-access archive for nearly 2.4 million scholarly articles in the fields of physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering and systems science, and economics. The dataset can be accessed [here](https://www.kaggle.com/datasets/Cornell-University/arxiv/data).

The dataset is hosted on a google cloud stoarge bucket by arXiv with a size of over 1.1 TB. It consists of millions of research papers from different fields in PDF format.

For our project, we used about 28 GB of data (~9400 PDF files related to Computer Science).

The dataset can be downloaded to the local system using a simple shell script in [dataset_download.sh](https://github.com/nishant1695/Advanced-Query-Processing-with-Spark-and-OpenAI/blob/main/dataset_download.sh)


## Environment
Our project was majorly executed on AWS EMR clusters as handling and processing a huge volume of data on local system is impractical.

### Setup
Setting up the EMR cluster is fairly straight forward. We used a [bootstrap script](https://github.com/nishant1695/Advanced-Query-Processing-with-Spark-and-OpenAI/blob/main/install_libraries_rag.sh) while spinning up the EMR instances. The script installs the following libraries using pip:

```
sudo python3 -m pip install numpy pandas langchain pypdf2 langchain-community faiss-cpu boto3 sentence_transformers==2.2.2 openai==0.28.1
```

We also set the S3 bucket as public so that the data can be accessed from any AWS account. For turning on public acccess, we turned of public access restircition under permissions tab in the bucket and added the following bucket policy:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AddPerm",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:ListBucket",
            "Resource": "bucket-arn"
        },
        {
            "Sid": "AddPerm2",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "bucket-arn/*"
        }
    ]
}
```

## Data Flow

High level data flow in our project is as follows:

Retrieve data from arXiv --> Store the data in S3 bucket --> Extract text from the PDFs in batches stored in S3 bucket --> Store the extracted text in S3 bucket --> Read the extracted text and generate vector embeddings and create a vector store --> Save the vecotr store to S3 bucket --> Load the vector store and use it as retriever for the LLM



## Execution
