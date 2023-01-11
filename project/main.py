'''
	CptS 315 Final Project
	Jose Robles
	Methodology: Decision tree using gini index and entropy
	Dataset: newPlayers.csv (modified version of england-premier-league-players-2018-to-2019-stats.csv -> irrelevant columns removed [Nationality, club name, etc.])
	Target attribute: classified with 2s, 1s, and 0s based on a different column known as rank_in_club_top_scorer
		If a player is in top 5 they'll receive a 2, if in top 10 but not in the top 5 they'll receive a 1, else they get a 0
		The decision tree will split up the csv into training and use that to correctly assign a 2, 1, or 0 to the rest of the csv
		which will be the testing data. 
		Attributes taken into consideration: goals, assists, goal involvements, penaltys, etc. 

	See README for package info 
'''

# Required packages
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Functions 
def prediction(X_test, object):
	# Make a prediction by taking ini the test and an object representing gini or entropy
	y_prediction = object.predict(X_test)
	print("Predicted values:")
	print(y_prediction)
	return y_prediction
	
def get_accuracy(y_test, y_prediction):
	# Confusion Matrix is used to understand the trained classifier behavior over the test dataset or validate the dataset.
	print("Confusion Matrix: ")
	print(confusion_matrix(y_test, y_prediction))
	print ("Accuracy: ", accuracy_score(y_test, y_prediction)*100)
	print("Report: ", classification_report(y_test, y_prediction))


# Driving Code
# Read/import the dataset from a csv into a pandas Dataframe object
player_data = pd.read_csv("./data/newPlayers.csv" , sep = ',')

print("Dataset attributes: ")
print("Length: ", len(player_data))
print("Shape: ", player_data.shape)
print("Dataset: ", player_data)

# Split the dataset into training and testing, ratio = 70% training, 30% testing
# Separating the target variable, x==all other columns, y==target column
X = player_data.values[:, 1:player_data.shape[1] - 1]
y = player_data.values[:, player_data.shape[1]-1]
y=y.astype('int')										# convert so sklearn can recognize y's type [necessary for no errors]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)


# As max_depth increases, accuracy increases

# Train the decision tree using giniIndex 
gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5)
# Performing training
gini.fit(X_train, y_train)


# Train the decision tree using entropy
entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth =5, min_samples_leaf = 5)
# Performing training
entropy.fit(X_train, y_train)


# Output
print("Decision tree results using gini index: ")
y_pred_gini = prediction(X_test, gini)
get_accuracy(y_test, y_pred_gini)

print("Decision tree results using entropy: ")
y_pred_entropy = prediction(X_test, entropy)
get_accuracy(y_test, y_pred_entropy)