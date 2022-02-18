import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path('.').absolute()))

import logging
import datetime
import numpy as np
from tqdm import tqdm
import numbers
import pyarrow.parquet as pq
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import config

# ignore col assignment warning, since no chained assignment happens
pd.options.mode.chained_assignment = None  # default='warn'


def load_df(data_description_path: str):
    """ Load dataframe from the given path.
    Expecting the data to be in a parquet format.
    :param data_description_path:   Name of the given dataset (Path is resolved automatically from config)

    :returns pandas dataframe
    """
    input_base_path = os.path.abspath(config.BASE_PROCESSED_DATA_FOLDER)

    input_file = f"{data_description_path}.parquet"
    path = input_base_path + "/" + input_file

    if not os.path.exists(path):
        logging.info(f"Path {path} does not exist, quitting")
        quit()

    logging.info(f"Loading from {path}")
    loaded_table = pq.read_table(path)

    return loaded_table.to_pandas()


def prepare_dataset(df: pd.DataFrame, ignore_times: bool=False, ignore_lists: bool=False) -> None:
    """ Preparing the given data frame for the process by writing into it's attributes.

    :param df:              Given dataframe
    :param ignore_times:    Whether to transform time-variables to numerical variables or not
    :param ignore_lists:    Whether to transform list-variables to categorical variables or not 
    
    """
    for col in df.columns:
        logging.info(f"\nHandeling col {col}")

        # Pandas just umbrellas string and lists to object type, therefore look at the actual value
        value = df[col].iloc[0]
 
        if isinstance(value, numbers.Number):
            min_max_scaler = MinMaxScaler()
            df[col] = min_max_scaler.fit_transform(df[[col]])
            df.attrs[col] = {"datatype": numbers.Number, "min_value": min_max_scaler.data_min_[0], "max_value": min_max_scaler.data_max_[0]}
            print(f"[{col}] Number -> Scaled to [0, 1] with a min-max of {min_max_scaler.data_min_[0]}-{min_max_scaler.data_max_[0]}")
        elif isinstance(value, str):
            # day of week is a sstring but should be treated as a time variable, therefore hack it manually!
            if col == "day_of_week":
                weekday_lookup = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

                # Day of the Week is a cyclic variable, therefore it should be transformed into a numerical variable
                if ignore_times:
                    min_max_scaler = MinMaxScaler()
                    # Transform weekdays to their proper indices as defined in weekday_lookup
                    d = np.array(df[col].apply(lambda x: weekday_lookup[x]))
                    #data = np.array([int(value.strftime(config.TIME_FORMAT)) for value in tqdm(df[col])])
                    df[col] = min_max_scaler.fit_transform(d.reshape((-1, 1))).reshape(-1)
                    df.attrs[col] = {"datatype": numbers.Number, "min_value": min_max_scaler.data_min_[0], "max_value": min_max_scaler.data_max_[0]}
                    print(f"[{col}] Number -> Scaled to [0, 1] with a min-max of {min_max_scaler.data_min_[0]}-{min_max_scaler.data_max_[0]}")

                else:
                    df[col] = df[col].apply(lambda x: weekday_lookup[x])
                    #df[col] = [int(value.strftime(config.TIME_FORMAT)) for value in tqdm(df[col])]
                    print(f"[{col}] Time/Format -> Applied format DAY OF THE WEEK")

                    df.attrs[col] = {"datatype": datetime.date, "format": "DAY_OF_WEEK", "max_time_value": len(weekday_lookup)}
            else:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                df.attrs[col] = {"datatype": str, "classes" : label_encoder.classes_}
                print(f"[{col}] Str/Categorical -> Scaled to the following classes {label_encoder.classes_}")

        elif isinstance(value, (np.ndarray, list)):
            if ignore_lists:
                label_encoder = LabelEncoder()
                raw_data = [frozenset(v) for v in df[col].to_numpy()]
                df[col] = label_encoder.fit_transform(raw_data)
                df.attrs[col] = {"datatype": str, "classes" : label_encoder.classes_}
                print(f"[{col}] Str/List Treated as Categorical -> Scaled to the following classes {label_encoder.classes_}")
            else:
            # Theoretically i could go more in depth (e.g. list of strings, list of numbers etc., assume it is list of strings)
                unique_labels = set()
                label_encoder = LabelEncoder()
                data = df[col].to_numpy()


                unique_labels = np.unique(np.concatenate(data))
                label_encoder.fit(list(unique_labels))
                df[col] = [frozenset(label_encoder.transform(value)) for value in tqdm(df[col])]
                
                print(f"[{col}] List/Categorical -> Applied label encoder on all existing unique values with classes {label_encoder.classes_}")
                df.attrs[col] = {"datatype": list, "classes": label_encoder.classes_}
        elif isinstance(value, datetime.date):
            # Assuming here that each date column will be transformed to the configured TIME_FORMAT (which should be representable through a single integer).
            if ignore_times:
                min_max_scaler = MinMaxScaler()
                data = np.array([int(value.strftime(config.TIME_FORMAT)) for value in tqdm(df[col])])
                df[col] = min_max_scaler.fit_transform(data.reshape((-1, 1))).reshape(-1)
                df.attrs[col] = {"datatype": numbers.Number, "min_value": min_max_scaler.data_min_[0], "max_value": min_max_scaler.data_max_[0]}
                print(f"[{col}] Number -> Scaled to [0, 1] with a min-max of {min_max_scaler.data_min_[0]}-{min_max_scaler.data_max_[0]}")
            else:
                # For simplicity reasons only the format "Hour of the Day" is included. More can be easily added in the future.
                if config.TIME_FORMAT == "%H":
                    max_time_value = 24
                else:
                    raise NotImplementedError(f"Unrecognized time format, currently only supporting '%H', found {config.TIME_FORMAT}")
                
                df[col] = [int(value.strftime(config.TIME_FORMAT)) for value in tqdm(df[col])]
                print(f"[{col}] Time/Format -> Applied format {config.TIME_FORMAT}")

                df.attrs[col] = {"datatype": datetime.date, "format": config.TIME_FORMAT, "max_time_value": max_time_value}
        else:
            raise RuntimeError(f"Unrecognized Datatype: {value} for column {col}")

class CustomDataset:
    def __init__(self, data_description_path: str, ignore_times: bool=False, ignore_lists: bool=False) -> None:
        """ CustomDataset is a wrapper class to represent and prepare the underlying dataset by adding encoding features and adding meta information.
        
        :param data_description_path:   Name of the given dataset (Path is resolved automatically from config)
        :param ignore_times:            Whether to transform time-variables to numerical variables or not
        :param ignore_lists:            Whether to transform list-variables to categorical variables or not 
        
        """
        df = load_df(data_description_path)
        df = df[df.columns.difference(config.IMMEDIATLY_DROPPABLE_COLUMNS)]
        self.df_training = df[df.columns.difference(config.IGNORABLE_FEATURES_FOR_EVAL)]
        print(self.df_training)

        # Analyses column type, adds meta data, scales data.
        prepare_dataset(self.df_training, ignore_times, ignore_lists)
        self.df_eval = df[config.IGNORABLE_FEATURES_FOR_EVAL]

    def __str__(self) -> str:
        nl = '\n\t'
        return f"{'*'*30}\n" + \
        f"Shape of Training Set (Rows x Columns):\t{self.df_training.shape}\n\n" +\
        f"-> enhanced the columns by their datatype information:{nl}{nl.join([f'{k}: {v}' for k, v in self.df_training.attrs.items()])}\n\n"

    def get_training_df(self) -> pd.DataFrame:
        return self.df_training

    def get_eval_df(self) -> pd.DataFrame:
        return self.df_training
        
    def update_training_df(self, remaining_columns) -> None:
        """ Removes non-existing columns from the training dataframe. Updates the attributes of the dataframe.
        
        :param remaining_columns:   List of remaining column names to keep in the dataframe
        """
        if len(self.df_training.columns) > len(remaining_columns):
            self.df_training = self.df_training[remaining_columns]
            self.df_training.attrs = {r: self.df_training.attrs[r] for r in remaining_columns}
            columns = list(self.df_training.columns)
            logging.info(f"\nThe following columns represent the data (after update):\n{pd.DataFrame(columns, columns=['Column Names']).to_markdown()}")

if __name__ == "__main__":
    dataset = CustomDataset(config.create_processed_data_path(True), False, False)
    print(dataset)
    print(dataset.get_training_df())
