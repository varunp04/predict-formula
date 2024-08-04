import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


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
        oh = OneHotEncoder(
            sparse_output=False, handle_unknown="infrequent_if_exist"
        ).set_output(transform="pandas")
        oh.fit(feature_dataframe[ohe_features])

        return oh.transform(feature_dataframe[ohe_features]), oh

    def shift_column(
        self, feature_dataframe: pd.DataFrame, column_list: List[str]
    ) -> pd.DataFrame:
        """Add previous lap's lap time to the dataframe"""

        feature_dataframe = feature_dataframe.sort_values(by=["lap"])

        feature_dataframe = feature_dataframe.reset_index(drop=True)

        feature_dataframe[self.config.get("LAGGED_FEATURE_NAMES")] = feature_dataframe[
            column_list
        ].shift(1)

        return feature_dataframe

    def add_lagged_features(self, feature_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features by raceId"""

        feature_dataframe_grp = feature_dataframe.groupby("raceId").apply(
            self.shift_column, column_list=self.config.get("LAGGED_FEATURES")
        )

        feature_dataframe = feature_dataframe_grp.reset_index(drop=True)

        return feature_dataframe

    def create_cumulative_pit_stop_indicator(
        self, feature_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add column that indicates how many pit stops are taken before the lap"""

        feature_df["PitStopIndicator"] = feature_df.groupby("raceId")[
            "isPitStop"
        ].cumsum()

        return feature_df

    def race_flags(
        self, feature_data: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Flagging the lap times that were 30,000 ms greater than fastest lap time"""

        race_id = feature_data["raceId"].unique()[0]

        df_temp = results_df[results_df["raceId"] == race_id]

        fastest_lap_time = (
            pd.to_timedelta("00:" + df_temp["fastestLapTime"]).dt.total_seconds() * 1000
        )

        flag_time = fastest_lap_time.values[0] + 40000.0

        feature_data["isRaceFlag"] = feature_data["milliseconds"] > flag_time

        return feature_data

    def add_race_flag_condition(
        self, data: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add column that define if the flag was raised"""

        data_grp = data.groupby("raceId").apply(self.race_flags, results_df=results_df)

        feature_data = data_grp.reset_index(drop=True)

        return feature_data

    def engineer_data(
        self, lap_times_data: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Engineer the data by applying relevent engineering techinques"""

        lap_times_selective_races = self.drop_race_ids(feature_dataframe=lap_times_data)

        lap_times_data_selected_features = lap_times_selective_races[
            self.config.get("FEATURE_USED_IN_TRAINING")
        ]

        lap_times_data_selected_features["isPitStop"] = (
            lap_times_data_selected_features["isPitStop"].astype(int)
        )

        lap_times_data_selected_features = self.create_cumulative_pit_stop_indicator(
            feature_df=lap_times_data_selected_features
        )

        lap_times_data_selected_features = self.add_race_flag_condition(
            data=lap_times_data_selected_features, results_df=results_df
        )

        lap_times_data_selected_features["isRaceFlag"] = (
            lap_times_data_selected_features["isRaceFlag"].astype(int)
        )

        lap_times_data_selected_features["lapDifference"] = (
            lap_times_data_selected_features.groupby(["raceId"])["milliseconds"].diff()
        )
        lap_times_data_selected_features["lapDifference"] = (
            lap_times_data_selected_features["lapDifference"].fillna(0)
        )

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
        self, data: pd.DataFrame, race_id_list: List
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """select a race event as a test set to show the real life scenario"""

        test_data = data[data["raceId"].isin(race_id_list)]
        test_data = test_data.reset_index(drop=True)

        train_data = data[~data["raceId"].isin(race_id_list)]
        train_data = train_data.reset_index(drop=True)

        return train_data, test_data

    def train_validation_split(
        self, x_sequential_ls: List, y_sequential_ls: List
    ) -> Tuple[List, List, List, List]:
        """create a split so that train set consist of 80% of that laps"""

        x_train, x_val, y_train, y_val = train_test_split(
            x_sequential_ls, y_sequential_ls, train_size=0.8
        )

        return x_train, x_val, y_train, y_val
