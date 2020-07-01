# Disaster Response Pipeline Project
### Introduction

This project is to classify disaster messages. ML model is trained and deployed as API to help emergency worker by automating received messages classififcation. The worker need to blug in the message and the catgories will be displayed. 

### Project Modules:
#### ETL Pipeline
process_data.py is a data prepocessing pipeline that:
- Loads messages and catgories datasets
- Merges datasets
- Removes duplicate
- Stores data as sqlite 
To run this module, go to command promt and use the following command
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

#### ML Pipeline
Machine learning pipeline that:
- loads sqlite data
- Splites dateset to training and testing
- Trains machine learning model and evaluates using dataset
- Save model as pickle file
To run this module, go to command promt and use the following command \n
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


#### Flask Web App
Calling app will run a web application that receives messages during disaster. The app classifes the type of the receives messages so approprite relief agnecy can help. to start the app run the following
    python run.py

### Instructions to Run Web App:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
