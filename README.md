Planned Parenthood wants to assist women in America to choose proper health-care service for them.
So, they were hosted a competition at DrivenData to challenge Data Scientist around the world to predict which reproductive health-care services accessed by women in America.

More about problem description: http://www.drivendata.org/competitions/6/page/26/

MY APPROACH:
To solve the problem, several models with common preprocessing were used. I started with the simplest logistic regression, and then switched to “tree” models (DecisionTree, RandomForest and their modifications). The best result was shown by Catboost, a gradient boosting on trees algorithm from Yandex, specifically designed to work with categorical data, which is what this dataset mainly consists of.

PREPROCESSING PIPELINE:
For data preprocessing, I will use a pipeline, which, depending on the type of column, will produce 3 main transformations:

1. CLEANING: removing some of the columns (uninformative or highly correlated with others), counting the number of missing values in a row (extra feature)

2. IMPUTING & SCALING: imputing missing values using one of the algorithms:
        -SimpleImputer - imputing with a constant value that obviously goes beyond the boundaries of the existing data (a solid choice for decision trees e.g). In our case, a large negative value is suitable, because the dataset is filled with positive values only
        -KNNImputer - we will experiment with the number of neighbors
        -IterativeImputer  - an algorithm that considers each column in the dataset as a function of the others, trains a model using a selected algorithm, and imputes missing values using the prediction
             
      After imputation, the data is optionally normalized (where this is important) and the most informative columns are selected using PCA
   
3. ENCODING: for ML algorithms that work only with numbers, produce one-hot encoding or another type of encoding of categorical columns, depending on the statistics of their unique values

These steps will be implemented as classes equipped with "fit" and "transform" methods.

DYNAMIC COLUMN_TRANSFORMER:
To assemble a model in which the preprocessing will be the first step before applying the ML algorithm, I will create a separate class that will collect all the transformers using Pipeline and ColumnTransformer. ColumnTransformer will allow us to separately deal with each of the three data types, and the fit and transform methods will allow us to use the resulting pipeline during cross-validation. In such case only the training data will be used to configure the algorithm, and only the resulting transformations will be applied to the validation data.

To make the ColumnTransformer work one needs to specify a list of columns to which the transformation will be applied. Since I throw out a certain a priori unknown number of columns at the first step, I need to create the list of columns for the ColumnTransformer dynamically. To do so, I create a class "make_column_selector" in which I specify the pattern for selecting columns. I indicate the column type as a pattern, and for the categorical type I also separate columns with a small and large number of unique values (the exact division boundary is set experimentally during GridSearch).
