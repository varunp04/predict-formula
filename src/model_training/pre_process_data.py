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
        """

        lap_times_merge_races = pd.merge(
            lap_times_df,
            races_df[self.config.get("RACES_DF_COLUMNS_FOR_MERGE")],
            on="raceId",
            how="left",
        )

        lap_times_master_df = pd.merge(
            lap_times_merge_races,
            results_df[self.config.get("RESULTS_DF_COLUMNS_FOR_MERGE")],
            on=["driverId", "raceId"],
            how="left",
        )

        lap_times_master_df = lap_times_master_df.drop(columns=["time"])
        return lap_times_master_df

    def add_pitstop_data(
        self, master_laptime_data: pd.DataFrame, pit_stop_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add pitstop duraction in minutes and milliseconds
        and a boolean column that says if the pitstop was taken in the lap

        """

        pit_stop_data = pit_stop_data.rename(
            columns={
                "milliseconds": "pitStopMilliseconds",
            }
        )

        master_laptime_data_with_pit_stops = pd.merge(
            master_laptime_data,
            pit_stop_data[self.config.get("PIT_STOP_DF_COLUMNS_FOR_MERGE")],
            on=["raceId", "driverId", "lap"],
            how="left",
        )

        master_laptime_data_with_pit_stops = master_laptime_data_with_pit_stops.fillna(
            0
        )

        master_laptime_data_with_pit_stops["isPitStop"] = (
            master_laptime_data_with_pit_stops["pitStopMilliseconds"] > 0
        )

        return master_laptime_data_with_pit_stops
    