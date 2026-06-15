"""
Evaluation metrics for demand forecasting, including M5 competition metrics (RMSSE, WRMSSE).
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pandas as pd

def absolute_error(y_true, y_pred):
    """Absolute error for each prediction."""
    return np.abs(np.asarray(y_true) - np.asarray(y_pred))

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    return mean_absolute_percentage_error(y_true, y_pred)


M5_LEVELS = {
    'total': None,
    'state': ['state_id'],
    'store': ['store_id'],
    'category': ['cat_id'],
    'department': ['dept_id'],
    'state_category': ['state_id', 'cat_id'],
    'state_department': ['state_id', 'dept_id'],
    'store_category': ['store_id', 'cat_id'],
    'store_department': ['store_id', 'dept_id'],
    'item': ['item_id'],
    'item_state': ['item_id', 'state_id'],
    'item_store': ['item_id', 'store_id'],
}


def rmsse(y_true, y_pred, y_train, eps=1e-8):
    """
    Root Mean Squared Scaled Error according to the M5 methodology.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    # Remove leading zeros before calculating the naive scale
    non_zero = np.flatnonzero(y_train > 0)

    if len(non_zero) == 0:
        return np.nan

    y_train_active = y_train[non_zero[0]:]

    if len(y_train_active) < 2:
        return np.nan

    scale = np.mean(np.diff(y_train_active) ** 2)

    if scale <= eps:
        return np.nan

    forecast_mse = np.mean((y_true - y_pred) ** 2)

    return np.sqrt(forecast_mse / scale)

def wrmsse_m5(
    df_train,
    df_test,
    target_col='sales',
    pred_col='y_pred',
    price_col='sell_price',
    date_col='date',
):
    """
    Calculate WRMSSE using the M5 competition methodology.

    Parameters
    ----------
    df_train : pd.DataFrame
        Historical training data.

    df_test : pd.DataFrame
        Forecast horizon containing actual and predicted values.

    target_col : str
        Actual sales column.

    pred_col : str
        Prediction column.

    price_col : str
        Unit selling price column.

    date_col : str
        Date column.

    Returns
    -------
    wrmsse : float
        Mean WRMSSE across the 12 M5 hierarchy levels.

    level_results : pd.DataFrame
        WRMSSE for each hierarchy level.
    """

    train = df_train.copy()
    test = df_test.copy()

    train[date_col] = pd.to_datetime(train[date_col])
    test[date_col] = pd.to_datetime(test[date_col])

    # ---------------------------------------------------------
    # Revenue used for the M5 weights:
    # last 28 days of the training period
    # ---------------------------------------------------------

    last_train_dates = (
        train[date_col]
        .drop_duplicates()
        .sort_values()
        .tail(28)
    )

    weight_period = train[
        train[date_col].isin(last_train_dates)
    ].copy()

    weight_period['revenue'] = (
        weight_period[target_col]
        * weight_period[price_col]
    )

    level_results = []

    # ---------------------------------------------------------
    # Evaluate each of the 12 hierarchy levels
    # ---------------------------------------------------------

    for level_name, group_cols in M5_LEVELS.items():

        # -------------------------
        # Total level
        # -------------------------

        if group_cols is None:

            train_series = (
                train
                .groupby(date_col)[target_col]
                .sum()
                .sort_index()
            )

            test_series = (
                test
                .groupby(date_col)
                .agg(
                    y_true=(target_col, 'sum'),
                    y_pred=(pred_col, 'sum')
                )
                .sort_index()
            )

            level_rmsse = rmsse(
                y_true=test_series['y_true'].values,
                y_pred=test_series['y_pred'].values,
                y_train=train_series.values,
            )

            level_wrmsse = level_rmsse

            n_series = 1

        # -------------------------
        # Remaining levels
        # -------------------------

        else:

            train_agg = (
                train
                .groupby(group_cols + [date_col], observed=True)[target_col]
                .sum()
                .reset_index()
            )

            test_agg = (
                test
                .groupby(group_cols + [date_col], observed=True)
                .agg(
                    y_true=(target_col, 'sum'),
                    y_pred=(pred_col, 'sum')
                )
                .reset_index()
            )

            revenue_agg = (
                weight_period
                .groupby(group_cols, observed=True)['revenue']
                .sum()
                .reset_index()
            )

            total_revenue = revenue_agg['revenue'].sum()

            revenue_agg['weight'] = (
                revenue_agg['revenue']
                / total_revenue
            )

            rmsse_values = []

            # Generate one RMSSE per aggregated time series
            for keys, test_group in test_agg.groupby(
                group_cols,
                observed=True
            ):

                if not isinstance(keys, tuple):
                    keys = (keys,)

                train_mask = np.ones(len(train_agg), dtype=bool)

                for col, value in zip(group_cols, keys):
                    train_mask &= train_agg[col].eq(value).values

                train_group = (
                    train_agg.loc[train_mask]
                    .sort_values(date_col)
                )

                test_group = test_group.sort_values(date_col)

                series_rmsse = rmsse(
                    y_true=test_group['y_true'].values,
                    y_pred=test_group['y_pred'].values,
                    y_train=train_group[target_col].values,
                )

                result = {
                    col: value
                    for col, value in zip(group_cols, keys)
                }

                result['rmsse'] = series_rmsse

                rmsse_values.append(result)

            rmsse_df = pd.DataFrame(rmsse_values)

            # Join RMSSE with revenue weights
            weighted = rmsse_df.merge(
                revenue_agg,
                on=group_cols,
                how='left'
            )

            weighted = weighted.dropna(
                subset=['rmsse', 'weight']
            )

            # Re-normalize if any invalid RMSSE series were discarded
            weighted['weight'] = (
                weighted['weight']
                / weighted['weight'].sum()
            )

            level_wrmsse = (
                weighted['rmsse']
                * weighted['weight']
            ).sum()

            n_series = len(weighted)

        level_results.append({
            'level': level_name,
            'n_series': n_series,
            'wrmsse': level_wrmsse,
        })

    level_results = pd.DataFrame(level_results)

    # Official M5 gives equal importance to the 12 hierarchy levels
    final_wrmsse = level_results['wrmsse'].mean()

    return float(final_wrmsse), level_results