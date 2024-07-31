import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class processData:

    def __init__(self, config: Dict) -> None:
        self.config = config

    def create_initial_dataset(
        self,
        lap_times_df: pd.DataFrame,
        races_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ):
        """This function cretaes an intial dataset that has all the columns mentioned in the
        question document

        Args:
            lap_times_df: table with lap times
            races_df: table with desciption on the race
            results_df: table with final race results
        Return:
            Master dataframe with all the columns
        """

        lap_times_merge_races = pd.merge(
            lap_times_df,
            races_df[self.config.get("RACES_DF_COLUMNS")],
            on="raceId",
            how="left",
        )

        lap_times_master_df = pd.merge(
            lap_times_merge_races,
            results_df[self.config.get("RESULTS_DF_COLUMNS")],
            on=["driverId", "raceId"],
            how="left",
        )

        return lap_times_master_df

    def process_data(
        self,
        lap_times_df,
        pit_stops_df,
        qualifying_df,
        races_df,
        results_df,
        sprint_results_df,
        status_df,
    ) -> pd.DataFrame:
        """
        process data for data analysis
        """

        lap_times_master_df = self.create_initial_dataset(
            lap_times_df=lap_times_df, races_df=races_df, results_df=results_df
        )

        return lap_times_master_df
