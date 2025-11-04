# This file is an script to find those more representative diagnostic codes in different precentiles of the data 
import pandas as pd 
import numpy as np

def find_code_percentile_diagnostics(data, percentile=95):
    """
    Find diagnostic codes that fall within the specified percentile of total counts.

    Parameters:
    - data (pd.DataFrame): DataFrame containing diagnostic codes and their counts.
    - percentile (float): Percentile threshold to filter diagnostic codes.

    Returns:
    - List[str]: List of diagnostic codes within the specified percentile.
    """
    import pandas as pd

    # Calculate the threshold count for the given percentile
    threshold = data['count'].quantile(percentile / 100.0)

    # Filter diagnostic codes that meet or exceed the threshold
    representative_codes = data[data['count'] >= threshold]['diagnostic_code'].tolist()

    return representative_codes

def find_more_prevalent_diagnostics(data, top_n=1):
    """
    Find the top N most prevalent diagnostic codes based on their counts.

    Parameters:
    - data (pd.DataFrame): DataFrame containing diagnostic codes and their counts.
    - top_n (int): Number of top prevalent diagnostic codes to return.

    Returns:
    - List[str]: List of the top N most prevalent diagnostic codes.
    """
    import pandas as pd

    # Sort the data by count in descending order and select the top N codes
    prevalent_codes = data.sort_values(by='count', ascending=False).head(top_n)['diagnostic_code'].tolist()

    return prevalent_codes

def aggregate_dataframe(df):
    """
    Aggregate the DataFrame by diagnostic code, summing the counts.

    Parameters:
    - df (pd.DataFrame): DataFrame containing diagnostic codes and their counts.

    Returns:
    - pd.DataFrame: Aggregated DataFrame with summed counts per diagnostic code.
    """
    import pandas as pd

    agg_df = pd.DataFrame()
    df = df.drop(columns=['timestamp'])

    for code in df.columns:
        agg_df[code] = np.sum(df[code].values)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv')  # Example CSV file
    aggregated_df = aggregate_dataframe(df)
    print("Aggregated DataFrame:")
    print(aggregated_df)