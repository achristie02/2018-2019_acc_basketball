import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Connect to the SQLite database
conn = sqlite3.connect('acc1819.db')


# Query the data
box_scores_query = '''
    SELECT 
    (CASE 
        WHEN b.Home = 1 THEN b.Score - (SELECT Score FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.Score - (SELECT Score FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS MarginOfVictory,
    (CASE 
        WHEN b.Home = 1 THEN b.AST - (SELECT AST FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.AST - (SELECT AST FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS ASTdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.TOV - (SELECT TOV FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.TOV - (SELECT TOV FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS TOVdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.STL - (SELECT STL FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.STL - (SELECT STL FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS STLdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.BLK - (SELECT BLK FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.BLK - (SELECT BLK FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS BLKdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.ORB - (SELECT ORB FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.ORB - (SELECT ORB FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS ORBdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.DRB - (SELECT DRB FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.DRB - (SELECT DRB FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS DRBdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.FGA - (SELECT FGA FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.FGA - (SELECT FGA FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS FGAdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.FGM - (SELECT FGM FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.FGM - (SELECT FGM FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS FGMdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.[3FGA] - (SELECT [3FGA] FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.[3FGA] - (SELECT [3FGA] FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS [3FGAdiff],
    (CASE 
        WHEN b.Home = 1 THEN b.[3FGM] - (SELECT [3FGM] FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.[3FGM] - (SELECT [3FGM] FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS [3FGMdiff],
    (CASE 
        WHEN b.Home = 1 THEN b.FTA - (SELECT FTA FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.FTA - (SELECT FTA FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS FTAdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.FTM - (SELECT FTM FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.FTM - (SELECT FTM FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS FTMdiff,
    (CASE 
        WHEN b.Home = 1 THEN b.Fouls - (SELECT Fouls FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.Fouls - (SELECT Fouls FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS Foulsdiff
    FROM 
        box_scores b
    JOIN 
        games g ON b.GameId = g.GameId;

'''

# Load the box score data into a pandas DataFrame
box_scores_df = pd.read_sql_query(box_scores_query, conn)

# Create the independent variable matrix X and the dependent variable vector y
X = box_scores_df.drop(['MarginOfVictory'], axis=1).reset_index(drop=True)

# Create the dependent variable vector y
y = box_scores_df['MarginOfVictory']

# Set values to type float
X = X.astype('float')

# Different types of models to be tested
rf_model = RandomForestClassifier()
lr_model = LogisticRegression()
svm_model = SVC()

# create a list of the models
models = [rf_model, lr_model, svm_model]

# create the k-fold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# initialize variables to keep track of the best model and its score
best_model = None
best_score = 0

for model in models:
    # initialize a variable to keep track of the score for this model
    model_score = 0
    # Loop through X
    for train_idx, val_idx in kf.split(X):
        # split the data into training and validation sets
        X_train, y_train = X.values[train_idx], y.values[train_idx]
        X_val, y_val = X.values[val_idx], y.values[val_idx]
        
        # fit the model on the training data
        model.fit(X_train, y_train)
        
        # evaluate the model on the validation data
        score = model.score(X_val, y_val)
        
        # update the score for this model
        model_score += score
        
    # calculate the average score for this model
    model_score /= kf.n_splits
    
    # check if this model has the best score so far
    if model_score > best_score:
        best_model = model
        best_score = model_score

# print the best model and its score
print("Best model:", best_model)
print("Best score:", best_score)


# Query the data
box_scores_query2 = '''
    SELECT b.team, 
    AVG(CASE 
        WHEN b.Home = 1 THEN b.Score - (SELECT Score FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.Score - (SELECT Score FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS MarginOfVictory,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.AST - (SELECT AST FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.AST - (SELECT AST FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS ASTdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.TOV - (SELECT TOV FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.TOV - (SELECT TOV FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS TOVdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.STL - (SELECT STL FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.STL - (SELECT STL FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS STLdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.BLK - (SELECT BLK FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.BLK - (SELECT BLK FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS BLKdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.ORB - (SELECT ORB FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.ORB - (SELECT ORB FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS ORBdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.DRB - (SELECT DRB FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.DRB - (SELECT DRB FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS DRBdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.FGA - (SELECT FGA FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.FGA - (SELECT FGA FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS FGAdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.FGM - (SELECT FGM FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.FGM - (SELECT FGM FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS FGMdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.[3FGA] - (SELECT [3FGA] FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.[3FGA] - (SELECT [3FGA] FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS [3FGAdiff],
    AVG(CASE 
        WHEN b.Home = 1 THEN b.[3FGM] - (SELECT [3FGM] FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.[3FGM] - (SELECT [3FGM] FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS [3FGMdiff],
    AVG(CASE 
        WHEN b.Home = 1 THEN b.FTA - (SELECT FTA FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.FTA - (SELECT FTA FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS FTAdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.FTM - (SELECT FTM FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.FTM - (SELECT FTM FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS FTMdiff,
    AVG(CASE 
        WHEN b.Home = 1 THEN b.Fouls - (SELECT Fouls FROM box_scores WHERE GameId = b.GameId AND Home = 0) 
        ELSE b.Fouls - (SELECT Fouls FROM box_scores WHERE GameId = b.GameId AND Home = 1)
     END) AS Foulsdiff
    FROM 
        box_scores b
    JOIN 
        games g ON b.GameId = g.GameId
    Group 
        by team

'''

# Load the box score data into a pandas DataFrame
box_scores_df2 = pd.read_sql_query(box_scores_query2, conn)


# Test data is each team's average stats over the season
X_test = box_scores_df2.drop(['Team', 'MarginOfVictory'], axis=1).reset_index(drop=True)

# Predict the average margin of victory against neutral team
y_pred = best_model.predict(X_test)

box_scores_df2['Rating'] = y_pred
box_scores_df2 = box_scores_df2.sort_values(by=['Rating'], ascending=False)
box_scores_df2['Rank'] = range(1, len(box_scores_df2)+1)
box_scores_df2[['Team', 'Rating']].to_csv('ACCRankings1819.csv', index=False)
