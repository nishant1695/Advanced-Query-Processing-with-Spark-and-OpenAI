# Advanced Query Processing with Spark and OpenAI

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

We also set the S3 bucket as public so that the data can be accessed from any AWS account. For turning on public acccess, we turned of public access restriction under permissions tab in the bucket and added the following bucket policy:

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

Retrieve data from arXiv --> Store the data in S3 bucket --> Extract text from the PDFs in batches stored in S3 bucket --> Store the extracted text in S3 bucket --> Read the extracted text and generate vector embeddings and create a vector store --> Save the vector store to S3 bucket --> Load the vector store and use it as retriever for the LLM

![Data Flow Diagram](https://github.com/nishant1695/Advanced-Query-Processing-with-Spark-and-OpenAI/blob/main/images/Data%20Flow.jpg)


## Execution

### 1. Text Extraction

Once the data has been downloaded from arXiv and upload to the S3 bucket, execute [Preprocessing and Text Extraction](https://github.com/nishant1695/Advanced-Query-Processing-with-Spark-and-OpenAI/blob/main/Preprocessing%20and%20Text%20Extraction.ipynb) jupyter notebook on the EMR cluster to read the PDF files from the S3 bucket, extract the text from PDFs and store then in .txt files in the S3 bucket.

S3 bucket name and directory to read the files from can be configure in the notebook as follow:

```
bucket_name = "your-bucket-name"
prefix = "parent_folder/"
```
Similarly output path can be configured in the last block of code as follow:

```
for i in range(500,5000,500):
    batch_files = pdf_file_paths[i-500:i]
    print(len(batch_files))
    files_rdd = sc.parallelize(batch_files)
    results = files_rdd.map(process_pdf).collect()
    results_str = '\n'.join([str(result) for result in results])
    s3_client.put_object(Bucket=bucket_name, Key="output_folder"+str(i+5000)+ ".txt", Body=results_str)
```


### 2. Vector Embedding Generation and Vector Store Creation

Once the text has been extracted and saved to the S3 bucket, we can continue to ingest the text, generate embeddings and create a vector store to save the embeddings.

We used the following model to generate the embeddings:

```
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name= "all-MiniLM-L6-v2", )
```

We used FAISS as the vector store to save the embeddings generated:

```
from langchain_community.vectorstores import FAISS

faiss = FAISS.from_texts(split_text, embeddings)
faiss.save_local(local_dir)
```

### 3. Running the App

Install the dependencies outlined in [requirements.txt](https://github.com/nishant1695/Advanced-Query-Processing-with-Spark-and-OpenAI/blob/main/requirement.txt)

```
pip install -U -r requirement.txt
```

The location of the index file of vector store generated and saved previously can be set in [app.py](https://github.com/nishant1695/Advanced-Query-Processing-with-Spark-and-OpenAI/blob/main/app.py) before running the application:

```
index_path = "location of index file"
```

To run the app, execute the following command:

```
streamlit run app.py
```

## Results

