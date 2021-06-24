# Disaster Response Pipeline Project
### What is included in this project:
1. I have an ETL, which reads and cleans CSV files, which containe disaster response messages and writes them into an sqlite database. This ETL process can be found in --> web_app --> data --> process_data.py or in the Jupyter Notebook ETL Pipeline

2. I created a machine learning pipeline, which reads the sqlite database and trains a classifier to predict the type of message. This ML process can be found in --> web_app --> models --> train_classifier.py or in the Jupyter Notebook ML Pipeline

3. I created an app/website, which shows some insights of the data an predicts new messages by reading from an input field


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - Create virtual environment in the root folder disaster_response_app_pipeline_and_app
        - For Windows use: py -m venv "your_env_name"
        - Activate your virtual environment: .\your_env_name\Scripts\Activate
        - Install requirements: pip install -r requirements.txt
    
    - Go into web_app folder

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv DisasterResponse`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the web_app's directory to run your web app. (Got to folder web_app)
    `python disaster.py`

3. Go to the address, which is shown in the console output. In my case it's (http://192.168.0.165:3001/)

4. Once your app runs, it will look as follows:
    -![Alt text](http://full/path/to/img.jpg "Optional title")
