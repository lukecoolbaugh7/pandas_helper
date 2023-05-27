import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

class pandas_helper: 


    def load_csv_data(file_path):
        """
        Load CSV data into a DataFrame.
        """
        return pd.read_csv(file_path)


    def clean_data(df, method='drop'):
        """
        Handle missing data in the DataFrame.
        If method is 'drop', drop rows with missing data.
        If method is 'fill', fill missing data with 0.
        """
        if method == 'drop':
            return df.dropna()
        elif method == 'fill':
            return df.fillna(0)
        else:
            raise ValueError("Invalid method. Choose either 'drop' or 'fill'.")
        
    def filter_data(df, condition):
        """
        Filter DataFrame based on condition.
        condition should be a valid boolean mask.
        """
        return df[condition]


    def aggregate_data(df, column, aggfunc):
        """
        Perform aggregation on DataFrame.
        column is the column to aggregate on.
        aggfunc is the aggregation function (e.g., 'sum', 'mean')
        """
        return df[column].agg(aggfunc)


    def merge_dataframes(df1, df2, how='inner', on=None):
        """
        Merge two DataFrames.
        how is the type of merge to be performed.
        on is the column(s) on which to join.
        """
        return pd.merge(df1, df2, how=how, on=on)


    def write_to_csv(df, file_path):
        """
        Write DataFrame into a CSV file.
        """
        df.to_csv(file_path, index=False)

    def filter_by_value(df, column, value):
        """
        Filter DataFrame rows where 'column' equals 'value'.
        """
        return df[df[column] == value]
    
    def plot_frequency(df, column):
        """
        Plot the frequency of each unique value in a specific column of the DataFrame.
        """
        plt.figure(figsize=(10,6))
        sns.countplot(data=df, x=column)
        plt.title(f"Frequency of {column}")
        plt.show()
        
    def plot_pie(df, column):
        """
        Plot a pie chart of the unique values in a specific column of the DataFrame.
        """
        plt.figure(figsize=(10,6))
        df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f"Pie Chart of {column}")
        plt.ylabel('')
        plt.show()


    def plot_histogram(df, column, bins=10):
        """
        Plot a histogram of a specific column of the DataFrame.
        """
        plt.figure(figsize=(10,6))
        plt.hist(df[column], bins=bins, edgecolor='k')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


    def plot_scatter(df, x_column, y_column):
        """
        Plot a scatter plot using two specific columns of the DataFrame.
        """
        plt.figure(figsize=(10,6))
        plt.scatter(df[x_column], df[y_column])
        plt.title(f"Scatter plot of {x_column} vs {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
        
    def train_linear_regression(df, target_column):
        """
        Train a Linear Regression model.
        This model is best suited for situations where you want to predict a continuous target variable and 
        you expect a linear relationship between the features and target variable.

        The function splits the data into a training set and a test set, trains a Linear Regression model 
        on the training set, and returns the trained model.
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model


    def train_decision_tree(df, target_column):
        """
        Train a Decision Tree Classifier model.
        This model is best suited for classification tasks. It can handle both binary and multiclass classification.

        The function splits the data into a training set and a test set, trains a Decision Tree Classifier model 
        on the training set, and returns the trained model.
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return model


    def train_naive_bayes(df, target_column):
        """
        Train a Naive Bayes model.
        This model is often used for text classification or when the features are categorical or 
        independent from each other.

        The function splits the data into a training set and a test set, trains a Naive Bayes model 
        on the training set, and returns the trained model.
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model

    def train_logistic_regression(df, target_column):
        """
        Train a Logistic Regression model.
        This model is often used for binary classification problems.

        The function splits the data into a training set and a test set, trains a Logistic Regression model 
        on the training set, and returns the trained model.
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(df, target_column):
        """
        Train a Random Forest model.
        This model is used for both regression and classification tasks, and it also provides a good 
        indicator of the importance it assigns to your features.

        The function splits the data into a training set and a test set, trains a Random Forest model 
        on the training set, and returns the trained model.
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    
    def k_means_clustering(df, n_clusters):
        """
        Perform K-Means Clustering.
        This model is used for dividing data into clusters or groups. It is an unsupervised algorithm 
        and it tries to group similar clusters together in your data.

        The function trains a K-Means model on the entire dataset and returns the trained model.
        """
        model = KMeans(n_clusters=n_clusters)
        model.fit(df)
        return model