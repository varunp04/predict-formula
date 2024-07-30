from sklearn.metrics import mean_absolute_error, mean_squared_error


class EvaluateModellingResults:

    def get_metrics(self, actuals, predictions):
        """Return model metrics (mean squared error and mean absolute error)"""

        mse = mean_squared_error(y_true=actuals, y_pred=predictions)

        mae = mean_absolute_error(y_true=actuals, y_pred=predictions)

        return mse, mae
