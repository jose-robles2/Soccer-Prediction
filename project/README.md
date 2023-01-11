# Project Name
CptS 315 - English Premier League Club Top Scorer Predictor

## Project summary
A csv file named "newPlayers.csv" will be taken in and read in order to fetch all the information relating to all English premier league players in the 2018/19 season. This information will be utilized to set up a Dataframe object from the Pandas library, it will also serve as our training and testing data for the Python decision tree implementation from the sklearn library. 

Within the csv, classifiers of 2,1,0 are assigned to each player within the `target_attribute` column. This refers to a player's ranking within their club's top scoreres. 2 == top 5 club scorers, 1 == top 10 club scorers (but not top 5) and 0 == below the top 10 top club scorers. With the assignment of these classifiers, the decision tree will look at all the other attributes associated with a player in order to find out why it was assigned a 2,1,0. Things like (goals scored, assists, penalty goals, goal involvements, etc.)

The data is split into 70% training data and 30% testing data. The decision tree will look at the attributes associated with each EPL player within the testing data and attempt to accurately assign it a classifier of 2, 1, or 0 using the things it learned using the training data. 

## Packages
pandas - pip install pandas
sklearn - pip install -U scikit-learn