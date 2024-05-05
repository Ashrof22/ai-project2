import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the URL for the CSV file on GitHub
file_url = "merged_ww_case_simplify.csv"

# Read the CSV file
try:
    df = pd.read_csv(file_url)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df[['Date']], df['case_nor_03d'], test_size=0.2, random_state=42)

    # Convert 'Date' column to ordinal numbers
    X_train['Date_ordinal'] = X_train['Date'].apply(lambda x: x.toordinal())
    X_test['Date_ordinal'] = X_test['Date'].apply(lambda x: x.toordinal())

    # Drop 'Date' column from train and test sets
    X_train.drop(columns=['Date'], inplace=True)
    X_test.drop(columns=['Date'], inplace=True)

    # Linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Print the coefficients
    print("Coefficients:", model.coef_)

    # Print train and test Mean Squared Error
    print("Train MSE:", mean_squared_error(y_train, y_train_pred))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred))

    # Combine actual and predicted values with all columns for test set
    output_df_test = pd.concat([X_test, pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})], axis=1)

    # Print the DataFrame for test set
    print("Output DataFrame for Test Set:")
    print(output_df_test)

except Exception as e:
    print("An error occurred:", e)
