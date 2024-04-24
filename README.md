# Advanced-Query-Processing-with-Spark-and-OpenAI

## Introduction
Apache Kafka is an open-source distributed streaming system used for stream processing, real-time data pipelines, and data integration at scale. 

Apache Airflow is an open-source platform for developing, scheduling, and monitoring batch-oriented workflows.

## About the Dataset

Instead of using a static database, we opted to use an open-source API called [Random User Generator](https://randomuser.me/) to simulate real-time data generation. We used Python code defined in an airflow DAG to periodically call the API to generate random user data for multiple users to simulate multiple users in a system submitting information through a form.

## Environment
There are 2 major components required for implementing a Kafka-Airflow pipeline:
1. Kafka and Zookeeper
3. Airflow

### Setup
#### 1. Kafka and Zookeeper

ZooKeeper plays a critical role in a Kafka cluster by providing distributed coordination and synchronization services. It maintains the cluster's metadata, manages leader elections, and enables consumers to track their consumption progress.

We start by downloading Apache Kafka and Zookeeper's binary files from the [Official Website](https://kafka.apache.org/downloads).
Extract the downloaded file to a folder.

Create two folders for saving zookeeper logs and Kafka server logs.

Go to the Kafka folder, navigate to config/zookeeper.properties and modify the following attributes in the file to setup zookeeper service:
1. dataDir= Give the path for the folder created for zookeeper logs.
2. client port: Specifies the port number at which the zookeeper service will run at. Default port is 2181.
3. maxClientCnxns= This property limits the actions from a host to a single zookeeper server. Default value is 0. We change it to 1.

   

Similarly, open server.properties in the same folder and modify the following attributes in the file to setup kafka server:
1. listeners= This property is commented by default and we need to uncomment this to specify on what ports the listeners will listen at.
2. log.dirs= Give the path for the folder created for kafka logs.
3. zookeeper.connect= Specifies the address for the zookeeper service. Default is localhost:2181

Once the configuration files are modified and saved, we can proceed to start the services.

#### Starting Services
For the following commands, our Kafka and Zookeeper files were unzipped at the following location: D:/Setups/Kafka/

We'll pass the path of the zookeeper.properties file as an argument while starting the service. Use the following command to start the zookeeper service:

```
D:/Setups/Kafka/kafka/bin/windows/zookeeper-server-start.bat D:/Setups/Kafka/kafka/config/zookeeper.properties
```

Once zookeeper is up, we'll use the following commands to start the Kafka server:

```
D:/Setups/Kafka/kafka/bin/windows/kafka-server-start.bat D:/Setups/Kafka/kafka/config/server.properties
```

Once the Kafka server is up, we'll check for the already existing topics in the server. We can do so with the following command:

```
D:/Setups/Kafka/kafka/bin/windows/kafka-topics.bat --bootstrap-server=localhost:9092 --list
```
Note that we need to pass the current IP of the running Kafka server to check the existing topics.

If the desired topic doesn't exist, we can create a new topic using the following command. Let's create a topic named "hello_world":

```
D:/Setups/Kafka/kafka/bin/windows/kafka-topics.bat --create --topic hello_world --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1
```
Once the topic is created, we can create a producer using the following command:

```
D:/Setups/Kafka/kafka/bin/windows/kafka-console-producer.bat --topic hello_world --bootstrap-server localhost:9092
```

Now we can create multiple consumers which will subscribe to a specific topic for information:

```
D:/Setups/Kafka/kafka/bin/windows/kafka-console-consumer.bat --topic hello_world --bootstrap-server localhost:9092
```

Running the above command in multiple terminals will create multiple consumers subscribed to the topic "hello_world"



#### 2. Airflow
Airflow is a powerful platform for programmatically authoring, scheduling, and monitoring workflows. While Airflow isn't officially supported on Windows OS, we can use Airflow on Windows using Docker. 

**Docker** is a software platform that packages software into standardized units called containers that have everything the software needs to run including libraries, system tools, code, and runtime.

Steps for setting up Docker and Airflow:
1. Download and install Docker from the [official website](https://docs.docker.com/desktop/install/windows-install/)
2. Once installed, open docker and create a custom docker image file for running airflow in a code edited (eg. VS Code). We create a workspace folder and used the following parameters to create the file

   ```docker
   FROM apache/airflow:latest

   USER root
   RUN apt-get update && \
       apt-get -y install git && \
       apt-get clean

   USER airflow
   RUN pip install kafka-python
   ```

4. Then create a docker-compose.yaml file to link the image file created in the previous step to the container. This will allow us to access Airflow's console through the web browser. The following parameters were used to create the file:

   ```docker
   version: '3'

   services:
     sleek-airflow:
       image: airflow:latest

       volumes:
         - ./Airflow:/opt/airflow

       ports:
         - "8080:8080"
      
       command: airflow standalone
   ```
6. Once the yaml file is created, we 'compose up' the file to start the docker container with Airflow running in it. Docker will show that the container is running as follows:

   ![image](https://github.com/nishant1695/Kafka-Airflow-CS777/assets/20843966/828d0304-052f-4d2b-832a-dfc31e3da7cf)

7. Airflow's console can be accessed at http://localhost:8080/. The default login ID is admin and the default password is saved in your local workspace folder.

   ![image](https://github.com/nishant1695/Kafka-Airflow-CS777/assets/20843966/6e7cbded-67b9-4263-8d71-72601c6cc3a8)

   ![image](https://github.com/nishant1695/Kafka-Airflow-CS777/assets/20843966/32ee0663-4812-4a3b-9412-7d7c9608eff1)


9. Once the Airflow is up and running, we'll proceed with writing our directed acyclic graph (DAG). Each DAG represents a collection of tasks we want to run and is organized to show relationships between tasks in the Airflow UI. Create a folder dag in the workspace directory and create a python file in it. We define the code for calling the Random User Generation API, extracting required information from the response for each user and pass that information to the producer service running on Kafka. The producer service will then publish this information to a topic(s). We can also define in this file how often Airflow will trigger the DAG to fetch the data from the source.

   Function for fetching and formatting the data:

   ```python
   def get_data():
       import requests

       res = requests.get("https://randomuser.me/api/")
       res = res.json()
       res = res['results'][0]

       return res

   def format_data(res):
       data = {}
       location = res['location']
       data['id'] = str(uuid.uuid4())
       data['first_name'] = str(res['name']['first'])
       data['last_name'] = str(res['name']['last'])
       data['gender'] = str(res['gender'])
       data['address'] = f"{str(location['street']['number'])} {location['street']['name']}, " \
                      f"{location['city']}, {location['state']}, {location['country']}"
       data['post_code'] = str(location['postcode'])
       data['email'] = str(res['email'])
       data['username'] = str(res['login']['username'])
       data['dob'] = str(res['dob']['date'])
       data['registered_date'] = str(res['registered']['date'])
       data['phone'] = str(res['phone'])
       data['picture'] = str(res['picture']['medium'])

    return data
   ```

   Streaming data to Producer Service:

   ```python
   def stream_data():
       import json
       from kafka import KafkaProducer
       import time
       import logging

       producer = KafkaProducer(bootstrap_servers=['10.0.0.143:9092'], max_block_ms=5000)
       curr_time = time.time()

    
    while True:
        if time.time() > curr_time + 15: #15 seconds
            break
        try:
            res = get_data()
            res = format_data(res)

            print("data formatted!")
            topic_name = "hello_world"
            data = json.dumps(str(res)).encode('utf-8')
            producer.send(topic_name, value=data)
        except Exception as e:
            logging.error(f'An error occured: {e}')
            continue
   ```

   Defining DAG:

   ```python
   with DAG('user_automation',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:

       streaming_task = PythonOperator(
        task_id='stream_data_from_api',
        python_callable=stream_data
    )
   ```

10. Once the DAG is defined, it'll be automatically triggered by the Airflow at defined intervals. It can be triggered manually as well through the Airflow console.

    Defining interval:

      ```python
      default_args = {
          'owner': 'airflow_test',
          'start_date': datetime(2024, 3, 15, 21, 00)
      }
      ```


