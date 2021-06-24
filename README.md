# Disaster Response Pipeline Project

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