import sys
import pandas as pd
import numpy as np
import sqlalchemy as db
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load CSVs and merge them into one dataframe
    Merging is based on commin id
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id', how='left')
    return df


def clean_data(df):
    """
    Clean categorie columns, by extracting binary values from string and by renaming columns
    Drop duplicates after cleaning
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0,:]

    #extract a list of new column names for categories.
    category_colnames = [x.split("-")[0] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-",expand= True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns = 'categories', inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace = True)

    #column related contained class 2, but it was only 2, if all other columns were 0, which was also true for related = 0, hence I overwrite it
    
    #column related contained class 2, but it was only 2, if all other columns were 0, which was also true for related = 0, hence I overwrite it
    df.related.replace(2,0, inplace = True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///data/' + database_filename + '.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')
    pass  


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