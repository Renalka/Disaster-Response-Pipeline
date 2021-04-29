# Disaster Response Pipeline

This project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset provided by Figure Eight contains pre-labelled tweet and messages from real-life disasters. The project is aimed at using ML and building a Natural Language Processing (NLP) pipeline to categorize these messages on a real time basis.


This project has the following sections:

Processing data, building an ETL pipeline, cleaning the data and saving it in a SQLite database

Building a machine learning pipeline to train a model which can classify the messages into various categories

Running a web app which can be used to visualise the results in real time

## Datasets
The following datasets are used:

disaster_categories.csv: contains id and categories columns

disaster_messages.csv: contains id, message, original, genre columns


## Installations:
This project requires Python 3.x and the following Python libraries installed:

1. sys
2. pandas
3. numpy
4. re
5. sklearn
6. sqlalchemy
7. nltk
8. pickle
9. flask
10. plotly
11. json
  

## Files and their purpose:

#### data/process_data.py:
An ETL (Extract Train Load) pipeline used for merging datasets, data cleaning, and storing final data in a SQLite database

#### models/train_classifier.py:
A machine learning pipeline used for loading data, training a model, and saving the trained model as a pickle file

#### run.py:
Used for launching the web app that can classify disaster messages

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To go to app's directory, type `cd app`
   Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
OR
4. Open another terminal and type `env|grep WORK` to get your space id.

5. Now open your browser window, type https://viewa7a4999b-3001.udacity-student-workspaces.com and replace the viewa7a4999b part with your spaceid.

6. Press enter to open the web app in your browser.

