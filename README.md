# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



4. Try the following steps if you are working out of

Udacity workspace

Run your app with python run.py command
Open another terminal and type env|grep WORK this will give you the spaceid (it will start with view*** and some characters after that)
Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id that you got in the step 2
Press enter and the app should now run for you
Local Machine

Once your app is running (python run.py)
Go to http://localhost:3001 and the app will now run
