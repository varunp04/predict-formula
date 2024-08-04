from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import pickle
from utils import save_as_pickle


class transformData:

    def __init__(
        self,
        n_steps_input: int,
        n_steps_output: int,
        config: Dict,
    ) -> None:

        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.config = config
        self.TARGET_COLUMN = self.config.get("TARGET_COLUMN")

    def scale_month_column(
        self, column_series: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.sin(2 * np.pi * column_series / 12), np.cos(
            2 * np.pi * column_series / 12
        )

    def perform_robust_scaler(
        self, data: pd.DataFrame, columns_list: List
    ) -> Tuple[np.ndarray, RobustScaler]:
        """Perform RobustScaler on a specific column"""

        column_scaler = RobustScaler()
        column_scaled = column_scaler.fit_transform(data[columns_list])

        return column_scaled, column_scaler

    def perform_min_max_scaler(
        self, data: pd.DataFrame, columns_list: List
    ) -> Tuple[np.ndarray, MinMaxScaler]:
        """Perform Minmaxscaler on a specific column"""

        column_scaler = MinMaxScaler(feature_range=(0, 1))
        column_scaled = column_scaler.fit_transform(data[columns_list])

        return column_scaled, column_scaler

    def scale_data_validation(
        self, data: pd.DataFrame, scaler_dict: Dict
    ) -> Tuple[pd.DataFrame, List]:
        """Scale the numerical columns in the data by using the scalers in training"""

        ls_all_scaled_columns = []

        ## year scaler

        ls_all_scaled_columns.append("year")

        year_column_scaled = scaler_dict["year_scaler"].transform(data[["year"]])

        scaled_df = pd.DataFrame(year_column_scaled, columns=["year_scaled"])

        ## date scaler

        if "day" in data.columns:

            ls_all_scaled_columns.append("day")

            scaled_df["day_scaled"] = scaler_dict["day_scaler"].transform(data[["day"]])

        ## minmax scaler columns

        for column_name in self.config.get("MINMAX_SCALING_COLUMNS"):

            if column_name in data.columns:

                ls_all_scaled_columns.append(column_name)

                scaled_df[f"{column_name}_scaled"] = scaler_dict[
                    f"{column_name}_scaler"
                ].transform(data[[column_name]])

        ## Robust scaling

        for column_name in self.config.get("ROBUST_SCALING"):
            if column_name in data.columns:

                ls_all_scaled_columns.append(column_name)

                scaled_df[f"{column_name}_scaled"] = scaler_dict[
                    f"{column_name}_scaler"
                ].transform(data[[column_name]])

        ## month scaler

        if "month" in data.columns:

            ls_all_scaled_columns.append("month")

            scaled_df["month_sin"], scaled_df["month_cos"] = self.scale_month_column(
                data["month"]
            )

        ## target scaler

        ls_all_scaled_columns.append(self.TARGET_COLUMN)

        scaled_df[f"{self.TARGET_COLUMN}_scaled"] = scaler_dict[
            f"{self.TARGET_COLUMN}_scaler"
        ].transform(data[[self.TARGET_COLUMN]])

        return scaled_df, ls_all_scaled_columns

    def scale_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List, Dict]:
        """Scale the numerical columns in the data"""

        scaler_dict = {}

        ls_all_scaled_columns = []

        ## year scaler

        year_scaler = StandardScaler()
        ls_all_scaled_columns.append("year")

        year_column_scaled = year_scaler.fit_transform(data[["year"]])

        scaler_dict["year_scaler"] = year_scaler

        scaled_df = pd.DataFrame(year_column_scaled, columns=["year_scaled"])

        ## date scaler

        if "day" in data.columns:
            day_scaler = StandardScaler()
            ls_all_scaled_columns.append("day")

            day_column_scaled = day_scaler.fit_transform(data[["day"]])

            scaler_dict["day_scaler"] = day_scaler

            scaled_df["day_scaled"] = day_column_scaled

        ## minmax scaler columns

        for column_name in self.config.get("MINMAX_SCALING_COLUMNS"):

            if column_name in data.columns:

                ls_all_scaled_columns.append(column_name)

                (
                    scaled_df[f"{column_name}_scaled"],
                    scaler_dict[f"{column_name}_scaler"],
                ) = self.perform_min_max_scaler(data=data, columns_list=[column_name])

        ## robust scaler

        for column_name in self.config.get("ROBUST_SCALING"):

            if column_name in data.columns:

                ls_all_scaled_columns.append(column_name)

                (
                    scaled_df[f"{column_name}_scaled"],
                    scaler_dict[f"{column_name}_scaler"],
                ) = self.perform_robust_scaler(data=data, columns_list=[column_name])

        ## month scaler

        if "month" in data.columns:

            ls_all_scaled_columns.append("month")

            scaled_df["month_sin"], scaled_df["month_cos"] = self.scale_month_column(
                data["month"]
            )

        ## target scaler

        ls_all_scaled_columns.append(self.TARGET_COLUMN)

        (
            scaled_df[f"{self.TARGET_COLUMN}_scaled"],
            scaler_dict[f"{self.TARGET_COLUMN}_scaler"],
        ) = self.perform_robust_scaler(data=data, columns_list=[self.TARGET_COLUMN])

        return scaled_df, ls_all_scaled_columns, scaler_dict

    def split_sequence(
        self, input_df: np.ndarray, output_df: np.ndarray
    ) -> tuple[List, List]:
        """Convert the data into a numpy array of shape: (n_steps, timesteps, n_feature)"""
        x, y = [], []

        for i in range(len(input_df)):
            end_ix = i + self.n_steps_input
            out_end_ix = end_ix + self.n_steps_output - 1
            if out_end_ix > len(input_df):
                break
            # gather input and output of the pattern
            seq_x, seq_y = input_df[i:end_ix], output_df[end_ix - 1 : out_end_ix, -1]
            x.append(seq_x), y.append(seq_y)

        return x, y

    def create_scaled_input_output_data(
        self,
        data: pd.DataFrame,
        train=False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Create scaled input data and scaled output data"""

        data = data.sort_values(by=["raceId", "lap"])
        data = data.reset_index(drop=True)

        columns_list = data.columns

        if train == True:

            scaled_df, all_scaled_columns, scaler_dict = self.scale_data(data=data)

            unscaled_columns = list(set(columns_list) - set(all_scaled_columns))

            scaled_df[unscaled_columns] = data[unscaled_columns]

            scaled_input_df = scaled_df.drop(columns=[f"{self.TARGET_COLUMN}_scaled"])
            scaled_output_df = scaled_df[["raceId", f"{self.TARGET_COLUMN}_scaled"]]

            save_as_pickle(
                path=self.config.get("MODEL_PATH"),
                artifact_name="scaler_dict.pkl",
                artifact=scaler_dict,
            )

            return scaled_input_df, scaled_output_df, scaler_dict

        else:

            with open(f"{self.config.get('MODEL_PATH')}scaler_dict.pkl", "rb") as f:
                scaler_dict = pickle.load(f)

            scaled_df, all_scaled_columns = self.scale_data_validation(
                data=data, scaler_dict=scaler_dict
            )

            unscaled_columns = list(set(columns_list) - set(all_scaled_columns))

            scaled_df[unscaled_columns] = data[unscaled_columns]

            scaled_input_df = scaled_df.drop(columns=[f"{self.TARGET_COLUMN}_scaled"])
            scaled_output_df = scaled_df[["raceId", f"{self.TARGET_COLUMN}_scaled"]]

            return scaled_input_df, scaled_output_df

    def create_sequence(
        self, scaled_input_df: pd.DataFrame, scaled_output_df: pd.DataFrame
    ) -> Tuple[List, List]:
        """Create sequence of the input and output data for model training"""

        x_sequential, y_sequential = [], []

        for id in scaled_input_df["raceId"].unique():
            x_trans_for_ss = scaled_input_df[scaled_input_df["raceId"] == id].drop(
                columns=["raceId"]
            )

            y_trans_for_ss = scaled_output_df[scaled_output_df["raceId"] == id].drop(
                columns=["raceId"]
            )

            x_ss, y_ss = self.split_sequence(
                input_df=x_trans_for_ss.to_numpy(), output_df=y_trans_for_ss.to_numpy()
            )

            x_sequential, y_sequential = x_sequential + x_ss, y_sequential + y_ss

        return x_sequential, y_sequential


class tranformDataInference(transformData):

    def __init__(
        self,
        n_steps_input: int,
        n_steps_output: int,
        config: Dict,
    ) -> None:
        super().__init__(n_steps_input, n_steps_output, config)

    def create_scaled_data_inference(self, data: pd.DataFrame):

        data = data.sort_values(by=["raceId", "lap"])
        data = data.reset_index(drop=True)

        columns_list = data.columns

        with open(f"{self.config.get('MODEL_PATH')}best_scaler_dict.pkl", "rb") as f:
            scaler_dict = pickle.load(f)

        scaled_df, all_scaled_columns = self.scale_data_validation(
            data=data, scaler_dict=scaler_dict
        )

        unscaled_columns = list(set(columns_list) - set(all_scaled_columns))

        unscaled_columns.extend([self.TARGET_COLUMN, "lap"])

        scaled_df[unscaled_columns] = data[unscaled_columns]

        scaled_input_df = scaled_df.drop(
            columns=["lap", self.TARGET_COLUMN, f"{self.TARGET_COLUMN}_scaled"]
        )
        output_df = scaled_df[["raceId", self.TARGET_COLUMN]]
        lap_df = scaled_df[["raceId", "lap"]]

        return scaled_input_df, output_df, lap_df
