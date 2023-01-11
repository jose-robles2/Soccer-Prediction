'''
    This file is used for modifying the target attribute column within the newPlayers.csv file. Currently, the target attribute
    column refers to whether a player is in the top 5 scorers for their club, the top 10 scorers for their club (but not top 5), 
    and finally, if a player is below the top 10. 
    
    Classifiers: 
    2 = top 5
    1 = top 10 but not top 5 
    0 = everything below top 10 

    With these, we are able to identify where a playere lies in the top score ranks 
    within their own club. The decision tree will take this column as it's target attribute. The
    newPlayers csv file will be split into training and testing data where the decision tree will look 
    at the existing 2,1,0 classifiers and use that to predict if a player should be marked as 2,1,0 
    depending on their stats for the season (goals, assists, etc.)

    Modify the for loop below if you wish to change how the classifiers are assigned
'''

# Importing the required package
import pandas as pd

# Read the csv
player_data = pd.read_csv('newPlayers.csv',sep= ',')
for row in range(0,player_data.shape[0]): 

    clubScorerRank = player_data.rank_in_club_top_scorer.values[row]
    if clubScorerRank <= 5 and clubScorerRank >= 1:
        player_data.target_attribute.values[row] = 2
    elif clubScorerRank <= 10 and clubScorerRank >= 6:
        player_data.target_attribute.values[row] = 1
    else:
        player_data.target_attribute.values[row] = 0

player_data.to_csv("./newPlayers.csv")