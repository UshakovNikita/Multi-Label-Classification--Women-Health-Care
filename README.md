Planned Parenthood wants to assist women in America to choose proper health-care service for them.
So, they were hosted a competition at DrivenData to challenge Data Scientist around the world to predict which reproductive health-care services accessed by women in America.

More about problem description: http://www.drivendata.org/competitions/6/page/26/

My approach:

To solve the problem, several models with common preprocessing described in Preprocessing.py were used. I started with the simplest logistic regression, and then switched to “tree” models (DecisionTree, RandomForest and their modifications). The best result was shown by Catboost, a gradient boosting on trees algorithm from Yandex, specifically designed to work with categorical data, which is what this dataset mainly consists of. I used GridSearch to find the optimal parameters for both preprocessing and the algorithm itself.

