# Disaster Response Pipeline Project
In this project, I implemented a data engineering process in building an ETL and an NLP pipeline as well as creating a web app with flask. As part of my task, I would be repairing the data with an ETL pipeline and using machine learning to build a supervised learning model to classify messages in to categories.

## ACKNOWLEDGE
### DATA CREDIT
UDACITY FIGURE8

Figure 8 provided me with pre-labeled tweets and text message from real life disasters.

## PROJECT MOTIVATION
Following a disaster, disaster agencies get millions of communication either directly or via social media at the same time. At this point, disaster organizations have the least capacity to filter and pull messages that are most important which is typically one in a thousand. When a disaster occurs dIfferent organisation usually take care of different part of the problem. These different scope of operation would be used to split the messages into categories

## LIBRARIES USED
The following libraries were used to excute this project

Numpy

Pandas

Matplot

Seaborn

Scikit Learn

Sqlalchemy

Re

NLTK

Plotly

Json

Flask

STACKS
Python

HTML

CSS

JAVASCRPT

### FILE DESCRIPTION
app | - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- InsertDatabaseName.db # database to save clean data to

models

|- train_classifier.py

|- classifier.pkl # saved model

README.md

## Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
Go to app directory: cd app

Run your web app: python run.py

Click the PREVIEW button to open the homepage
