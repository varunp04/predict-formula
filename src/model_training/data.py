import pandas as pd
from typing import Dict, Tuple
from torch.utils.data import Dataset



class gatherData:

    def __init__(self, config: Dict) -> None:
        self.config = config

    def load_data(self) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        read csv file from the data folder and return the dataframes for each
        csv file
        Return:
            tuple of dataframes
        """
        lap_times_df = pd.read_csv(self.config.get("DATA_FOLDER") + "lap_times.csv")
        pit_stops_df = pd.read_csv(self.config.get("DATA_FOLDER") + "pit_stops.csv")
        qualifying_df = pd.read_csv(self.config.get("DATA_FOLDER") + "qualifying.csv")
        races_df = pd.read_csv(self.config.get("DATA_FOLDER") + "races.csv")
        results_df = pd.read_csv(self.config.get("DATA_FOLDER") + "results.csv")
        sprint_results_df = pd.read_csv(
            self.config.get("DATA_FOLDER") + "sprint_results.csv"
        )
        status_df = pd.read_csv(self.config.get("DATA_FOLDER") + "status.csv")


        return (
            lap_times_df,
            pit_stops_df,
            qualifying_df,
            races_df,
            results_df,
            sprint_results_df,
            status_df,
        )


class TensorDataset(Dataset):

    def __init__(self, feature_set, target):
        self.X = feature_set
        self.y = target

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]