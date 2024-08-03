import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import List, Dict, Tuple


class engineerFeaturesForTraining:
    def __init__(self, config) -> None:
        self.config = config

    def drop_race_ids(self, feature_dataframe: pd.DataFrame) -> pd.DataFrame:
        """drop race data with less amount of lap data"""

        ls_race_id = []

        for race_id in feature_dataframe["raceId"].unique():

            number_of_laps = feature_dataframe[
                feature_dataframe["raceId"] == race_id
            ].shape[0]

            if number_of_laps >= self.config.get("MINIMUM_LAPS_IN_GP"):

                ls_race_id.append(race_id)

        feature_dataframe = feature_dataframe[
            feature_dataframe["raceId"].isin(ls_race_id)
        ]

        feature_dataframe = feature_dataframe.reset_index(drop=True)

        return feature_dataframe

    def add_date_features(self, feature_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add month and date of the race event"""

        feature_dataframe["date"] = pd.to_datetime(feature_dataframe["date"])

        feature_dataframe["month"] = feature_dataframe["date"].dt.month
        feature_dataframe["day"] = feature_dataframe["date"].dt.day

        feature_dataframe = feature_dataframe.drop(columns=["date"])

        return feature_dataframe

    def create_one_hot_encoding(
        self, feature_dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, OneHotEncoder]:
        """get the one hot encoded columns for the categorical column"""

        ohe_features = self.config.get("FEATURES_TO_ONEHOT_ENCODE")
        oh = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist").set_output(
            transform="pandas"
        )
        oh.fit(feature_dataframe[ohe_features])

        return oh.transform(feature_dataframe[ohe_features]), oh

    def shift_column(
        self, feature_dataframe: pd.DataFrame, column_list: List[str]
    ) -> pd.DataFrame:
        """Add previous lap's lap time to the dataframe"""

        ls_prior_columns = [
            "milliseconds_1_prior",
            "lap_number_1_prior",
            "position_1_prior_lap",
        ]

        feature_dataframe = feature_dataframe.sort_values(by=["lap"])

        feature_dataframe = feature_dataframe.reset_index(drop=True)

        feature_dataframe[ls_prior_columns] = feature_dataframe[column_list].shift(1)

        return feature_dataframe

    def add_lagged_features(self, feature_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features by raceId"""

        feature_dataframe_grp = feature_dataframe.groupby("raceId").apply(
            self.shift_column, column_list=["milliseconds", "lap", "position"]
        )

        feature_dataframe = feature_dataframe_grp.reset_index(drop=True)

        return feature_dataframe

    def engineer_data(self, lap_times_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer the data by applying relevent engineering techinques"""

        lap_times_selective_races = self.drop_race_ids(feature_dataframe=lap_times_data)

        lap_times_data_selected_features = lap_times_selective_races[
            self.config.get("FEATURE_USED_IN_TRAINING")
        ]

        lap_times_data_selected_features = self.add_lagged_features(
            feature_dataframe=lap_times_data_selected_features
        )

        lap_times_data_selected_features = lap_times_data_selected_features.dropna()

        lap_time_with_date_features = self.add_date_features(
            feature_dataframe=lap_times_data_selected_features
        )

        one_hot_encoded_df, one_hot_encoder = self.create_one_hot_encoding(
            feature_dataframe=lap_time_with_date_features
        )

        engineered_lap_time_data = pd.concat(
            [lap_time_with_date_features, one_hot_encoded_df], axis=1
        )

        engineered_lap_time_data = engineered_lap_time_data.drop(
            columns=self.config.get("FEATURES_TO_ONEHOT_ENCODE")
        )

        columns_from_oh_encoder = one_hot_encoded_df.columns

        return engineered_lap_time_data, one_hot_encoder, columns_from_oh_encoder


class splitData:

    def train_test_split(
        self, data: pd.DataFrame, race_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """select a race event as a test set to show the real life scenario"""

        test_data = data[data["raceId"] == race_id]
        test_data = test_data.reset_index(drop=True)

        train_data = data[data["raceId"] != race_id]
        train_data = train_data.reset_index(drop=True)

        return train_data, test_data

    def train_validation_split(
        self, train_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """create a split so that train set consist of 80% of that laps"""

        train_set_ls = []
        validation_set_ls = []

        for race_id in train_data["raceId"].unique():

            train_data_by_race_id = train_data[train_data["raceId"] == race_id]

            train_data_by_race_id = train_data_by_race_id.sort_values(by=["lap"])

            train_data_by_race_id = train_data_by_race_id.reset_index(drop=True)

            split_point = int(train_data_by_race_id.shape[0] * 0.8)

            train_set = train_data_by_race_id.iloc[:split_point]

            validation_set = train_data_by_race_id.iloc[split_point:]

            train_set = train_set.reset_index(drop=True)
            validation_set = validation_set.reset_index(drop=True)

            train_set_ls.append(train_set)
            validation_set_ls.append(validation_set)

        train_set_df = pd.concat(train_set_ls, axis=0)
        validation_set_df = pd.concat(validation_set_ls, axis=0)

        return train_set_df, validation_set_df
