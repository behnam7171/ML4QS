##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

# Simple class to impute missing values of a single columns.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class ImputationMissingValues:

    # Impute the mean values in case if missing data.
    def impute_mean(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].mean())
        return dataset

    # Impute the median values in case if missing data.
    def impute_median(self, dataset, col):
        dataset[col] = dataset[col].fillna(dataset[col].median())
        return dataset

    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset

    # Linear regression to predict missing values
    def impute_linear_regression(self,dataset,col):
        features_dataset = dataset[dataset[col].notna() == True]

        print(dataset[dataset[col].notna() == True])

        X = features_dataset.drop(col, axis=1)
        y = features_dataset[col]

        prediction_data = dataset[dataset[col].isnull()]
        prediction_data_X = prediction_data.drop(col, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        linear_regression = LinearRegression()
        linear_regression.fit(X_train, y_train)

        # predicted values
        y_pred = linear_regression.predict(prediction_data_X)

        # formatting the predicted values
        predictions = []
        for prediction in y_pred:
            predictions.append(float(format(prediction, '.3f')))

        # converting series to fill NaN values
        predictions_series = pd.Series(predictions)

        dataset.loc[dataset[col].isnull(), col] = predictions
        return dataset