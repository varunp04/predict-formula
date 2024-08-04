import numpy as np
from collections import defaultdict
import pandas as pd
from typing import List


class evaluateModellingResults:

    def get_model_evaluation_metrics(
        self, ls_prediction: List, ls_actual: List, lap_sequential: List
    ):
        """Return RMSE, MAE and MAPE from the list"""

        rmse_mae_mape_dict = defaultdict(list)

        prediction_actual_dict = defaultdict(list)

        for i in range(len(ls_prediction)):
            prediction_5_laps = ls_prediction[i]
            actuals_5_laps = ls_actual[i]

            prediction_actual_dict["laps_list"].append(lap_sequential[i])
            prediction_actual_dict["current_lap"].append(min(lap_sequential[i]) - 1)
            prediction_actual_dict["predictions"].append(prediction_5_laps)
            prediction_actual_dict["actuals"].append(actuals_5_laps)

            # Calculate RMSE and MAE once
            rmse = np.sqrt(np.mean((actuals_5_laps - prediction_5_laps) ** 2))
            mae = np.mean(np.abs(actuals_5_laps - prediction_5_laps))
            mape = (
                np.mean(np.abs((actuals_5_laps - prediction_5_laps) / actuals_5_laps))
                * 100
            )

            # Append values to the dictionary
            rmse_mae_mape_dict["laps_list"].append(lap_sequential[i])
            rmse_mae_mape_dict["current_lap"].append(min(lap_sequential[i]) - 1)
            rmse_mae_mape_dict["RMSE"].append(rmse)
            rmse_mae_mape_dict["MAE"].append(mae)
            rmse_mae_mape_dict["sMAPE"].append(mape)

        return rmse_mae_mape_dict, prediction_actual_dict
