def combine_csv_files_dask(
        file_paths: str,
        save_path: str,
        save: bool = True,
        remove_duplicates: bool = True,
        dtype: dict = None,
        **kwargs,
):
    """
    Load and row combine all csv files inside Reports folder and save them into a large csv file. Preferably,
    this should be done in parallel and using dask since the csv files are large.
    """
    from dask import dataframe as dd
    from glob import glob
    # Get the list of csv files
    csv_files = glob(file_paths)

    # Read the csv files in parallel
    df = dd.read_csv(csv_files, dtype=dtype, **kwargs)
    # Combine the dataframes
    df = df.compute()

    # What is the length of the dataframe?

    if save:
        # Save the dataframe
        df.to_csv(save_path, index=False, **kwargs)

    if remove_duplicates:
        # Remove duplicates
        df.drop_duplicates(inplace=True)
    return df
