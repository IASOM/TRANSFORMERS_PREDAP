from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
import numpy as np

def normalize_dataframe(train_df, test_df, csv_file=None, save_data=False, target_code=None, scaler=None):
    if 'timestamp' not in train_df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")
    train_df['timestamp'] = train_df['timestamp'].astype('datetime64[ns]')
    test_df['timestamp'] = test_df['timestamp'].astype('datetime64[ns]')

    if scaler is None:
        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()
    else:
        scaler_target = clone(scaler)

    codes = [code for code in train_df.columns if (code != 'timestamp' and code != target_code)]
    scaler.fit(train_df[codes].values)
    train_df[codes] = scaler.transform(train_df[codes].values)
    test_df[codes] = scaler.transform(test_df[codes].values)
    if target_code is not None:
        scaler_target = scaler_target.fit(train_df[[target_code]].values)
        test_df[[target_code]] = scaler_target.transform(test_df[[target_code]].values)
        train_df[[target_code]] = scaler_target.transform(train_df[[target_code]].values)

    if save_data and csv_file is not None:
        input_path = Path(csv_file)
        parent_dir = input_path.parent if input_path.parent != Path('.') else AssertionError("Input path must have a parent directory.")
        output_file = parent_dir / f"train_df_normalized_{input_path.name}"
        train_df.to_csv(output_file, index=False)

    return train_df, test_df


def inverse_transform_predictions(predictions, original_scale_df, code, lookback, forecast, cutoff_date='2010-01-01', max_date='2025-09-30', scaler = None, eliminate_covid_data=False, covid_dates=None, split_ratio = 0.8):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.base import clone
    code = code.replace("#", ":")
    if eliminate_covid_data:
        assert covid_dates is not None
        # Filter out covid dates
        for start_date, end_date in covid_dates:
            original_scale_df['timestamp'] = original_scale_df['timestamp'].astype('datetime64[ns]')
            original_scale_df = original_scale_df[~((original_scale_df['timestamp'] >= start_date) & (original_scale_df['timestamp'] <= end_date))]

    original_scale_df['timestamp'] = original_scale_df['timestamp'].astype('datetime64[ns]')
    original_scale_df = original_scale_df[(original_scale_df['timestamp'] >= cutoff_date) & (original_scale_df['timestamp'] <= max_date)].reset_index(drop=True)

    n = int(len(original_scale_df) * split_ratio)
    train_df = original_scale_df.iloc[:n]

    train_df_seq = []
    for i in range(len(train_df[[code]]) - lookback - forecast + 1):
        train_df_seq.append(train_df[[code]].values[i + lookback : i + lookback + forecast])
    train_df_seq = np.array(train_df_seq).squeeze()

    if scaler is None:
        scaler_target = MinMaxScaler()
    else:
        scaler_target = clone(scaler)

    if len(train_df_seq.shape) < 2:
        train_df_seq = train_df_seq.reshape(-1,1)

    scaler_target.fit(train_df_seq)
    pred_original_scale = scaler_target.inverse_transform(predictions)
    return pred_original_scale
