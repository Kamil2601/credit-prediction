import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path):
    """
    Loads data from a csv file, removes useless columns and rows, and splits date columns into month and year.
    """
    data = pd.read_csv(file_path, low_memory=False)

    # Drop last two rows with with summary ("Total amount...")
    data.drop(data.index[-2:], inplace=True)

    # Drop columns with all NaN values
    data.dropna(axis=1, how='all', inplace=True)
    
    # Drop columns where each (or almost each) value is unique
    # Each row has its own unique id and url
    # "desc", "emp_title" and "title" can be any string 
    columns_to_drop=["id", "url", "desc", "emp_title", "title"]
    data.drop(columns=columns_to_drop, axis=1, inplace=True)

    data = data[data['loan_status'] != 'Current']
    data['loan_status'] = data['loan_status'].apply(lambda x: 0 if x == 'Fully Paid' else 1)

    # Split columns with dates into month and year
    # Date format is "Mon-YYYY"
    for col in data.columns:
        if data[col].dtype == "object" and data[col].str.match(r"[A-z][a-z]{2}-\d{4}").any():
            date = pd.to_datetime(data[col], format='%b-%Y', errors='coerce')
            data[f'{col}_month'] = date.dt.month
            data[f'{col}_year'] = date.dt.year
            data.drop(col, axis=1, inplace=True)

    return data


def data_preprocessor(data):
    """
    Creates a preprocessor that scales numerical data and one-hot encodes categorical data.
    """
    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Define preprocessing for numerical data: impute missing values and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical data: impute missing values and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor