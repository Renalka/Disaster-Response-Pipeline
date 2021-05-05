#importing required libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to load messages data with categories
    
    Input:
    messages_filepath: Path to the CSV file containing messages
    categories_filepath: Path to the CSV file containing categories 
    Output:
    df: Merged dataframe with messages and categories
    """
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    df = messages.merge(categories, on='id', how='inner')
    
    return df


def clean_data(df):
    """
    Function to clean the data
    
    Input:
    df: Merged messages and categories dataset
    Output:
    df: Cleaned data
    """
    #Getting the names in the categories column
    categories = df["categories"].str.split(';', expand=True)
    cat = categories.iloc[0,:]
    category_names = cat.apply(lambda x: x[:-2])
    categories.columns = category_names
    #Converting category values to 0 or 1
    for x in categories:
        categories[x] = categories[x].str[-1] #Extracting last character of string
        categories[x] = categories[x].astype(int) #Converting column from string to int data type
    categories.replace(2, 1, inplace=True)
    
    df.drop('categories', axis=1, inplace = True)
    #Concatenating the new columns
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df
    

def save_data(df, database_filename):
    """
    Function to save dataframe to SQLite database
    
    Input:
    df: Combined and cleaneed dataset containing messages and categories
    database_filename: path to SQLite database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
