# Disaster Response Flask App

## Table of contents

- [What's included](#whats-included)
- [Libraries used](#libraries-used)
- [Instructions](#instructions)
- [Tools used](#tools-used)
- [Preview](#preview)
- [What's next](#next)



## What's included:
1. the repository containsan ETL process, which reads and cleans CSV files, which contain disaster response messages and which writes them into an sqlite database. This ETL process can be found in --> web_app --> data --> process_data.py or in the Jupyter Notebook ETL Pipeline

2. I created a machine learning pipeline, which reads the sqlite database and trains a classifier to predict the type of message. This ML process can be found in --> web_app --> models --> train_classifier.py or in the Jupyter Notebook ML Pipeline

3. I created a flask app/website, which shows some insights of the data an predicts new messages by reading from an input field

## Libraries used
- plotly
- pandas
- numpy
- sklearn
- nltk
- flask
- SQLAlchemy
- sklearn


## Tools used
- GridSearchCV
- FeatureUnion
- Pipeline
- Own class to tokenize


## Instructions:
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

## Preview
Once your app runs, it will look as follows:
    -![Cluster_by_genre](https://github.com/EriRika/disaster_response_app_pipeline_and_app/blob/master/pictures/App_preview.PNG "Cluster by Genre")
    -![Cluster_by_genre](https://github.com/EriRika/disaster_response_app_pipeline_and_app/blob/master/pictures/App_preview_2.PNG "Cluster by Genre")
    
    
## What's next
I tried to deploy the app on Heroku, but my pickle fiel and the database was too big and exceeded the free access heroku subscription policy. I would like to find a platform where I can publish the app

