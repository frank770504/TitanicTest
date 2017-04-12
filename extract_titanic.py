import pandas as pd
import numpy as np

def get_training_data(file_name):
    # Deal with the Training Data (Most from the sample code)
    train_df = pd.read_csv(file_name, header=0)
    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.

    # female = 0, Male = 1
    train_df['Gender'] = train_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:
        train_df['Embarked'].fillna(train_df.Embarked.dropna().value_counts().idxmax(), inplace=True)

    Ports = list(enumerate(np.unique(train_df['Embarked'])))  # determine all values of Embarked,
    Ports_dict = {name: i for i, name in Ports}  # set up a dictionary in the form  Ports : index
    train_df.Embarked = train_df.Embarked.map(lambda x: Ports_dict[x]).astype(int)  # Convert all Embark strings to int

    # All the ages with no data -> make the median of all Ages
    median_age = train_df['Age'].dropna().median()
    if len(train_df.Age[train_df.Age.isnull()]) > 0:
        train_df.loc[(train_df.Age.isnull()), 'Age'] = median_age

    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    # extract data
    train_data = train_df.values
    train_x = train_data[0::,1::]
    train_y = train_data[0::,0]
    # x data, y data, panda thing
    return train_x, train_y, train_df