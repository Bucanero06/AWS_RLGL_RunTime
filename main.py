from gc import collect
from glob import glob
from os import remove

import ray
from psutil import sensors_temperatures, virtual_memory
import numpy as np
import vectorbtpro as vbt
from numpy import nan, array, arange
from pandas import DataFrame, Timedelta, Series, concat
from time import perf_counter, sleep
from vectorbtpro.portfolio.enums import SizeType

from _utils import combine_csv_files_dask
from strategy_functions import run_RLGL_Optimization

# Settings
vbt.settings.set_theme("dark")
csv_data_name = "XAUUSD"
#
data = vbt.CSVData.fetch("XAUUSD.csv")
data.save("XAUUSD.pickle")
data = vbt.Data.load("XAUUSD.pickle")
#
RUN_N_SAMPLES = 100
N_Epochs = 2

INIT_CASH = 200_000
BET_SIZE = 100_000
SIZE_TYPE = SizeType.Value
#
TICK_FREQUENCY = "1 min"
TICK_SIZE = 0.01
FEES = 0.00005
SLIPPAGE = 0
#
USE_RAY = True
RAY_KWARGS = dict(
    num_cpus=28,  # cpu_count() - 2
)

CLEAR_CACHE_EVERY_N = 50
COLLECT_GARBAGE_EVERY_N = 50

# INPUTS:
TrendTimeframe = ["30 min", "1h", "4h"]
TrendRsi_Window = arange(5, 50, 20)
TrendGreen_Window = arange(2, 14, 10)
TrendBand_Window = arange(21, 63, 20)
#
EntryTimeframe = ["1 min", "5 min", "15 min"]
EntryRsi_Window = arange(5, 50, 20)
EntryGreen_Window = arange(2, 14, 10)
EntryRed_Window = arange(7, 21, 10)
EntryBand_Window = arange(21, 63, 20)
#
TrendMode = array([0, 1, 2, 3])  # 4 total "Modes"
EntryMode = array([0, 1, 2])  # 3 total "Modes"
ATR_Mult = arange(0.5, 2, 1)  # 0.5 to 2 step = 0.1

for i_epoch in range(N_Epochs):
    # todo should prepare parameter combinations prior to the loop and then iterate over them to save time
    ############################################################################################################
    # Below is the code to run:

    # %%
    total_number_of_combinations = len(TrendTimeframe) * len(TrendRsi_Window) * len(TrendGreen_Window) * len(
        TrendBand_Window) * len(EntryTimeframe) * len(EntryRsi_Window) * len(EntryGreen_Window) * len(
        EntryRed_Window) * len(EntryBand_Window) * len(TrendMode) * len(EntryMode) * len(ATR_Mult)

    # print all of them
    print(f'{TrendTimeframe = }')
    print(f'{TrendRsi_Window = }')
    print(f'{TrendGreen_Window = }')
    print(f'{TrendBand_Window = }')
    print(f'{EntryTimeframe = }')
    print(f'{EntryRsi_Window = }')
    print(f'{EntryGreen_Window = }')
    print(f'{EntryRed_Window = }')
    print(f'{EntryBand_Window = }')
    print(f'{TrendMode = }')
    print(f'{EntryMode = }')
    print(f'{ATR_Mult = }')
    print(f'{total_number_of_combinations = }')

    print(f'Total number of combinations: {total_number_of_combinations}')

    """You can either just run this function to backtest n_samples of the parameter space"""
    perf = run_RLGL_Optimization(data,
                                 trend_timeframe=TrendTimeframe,
                                 trend_rsi_window=TrendRsi_Window,
                                 trend_green_window=TrendGreen_Window,
                                 trend_band_window=TrendBand_Window,
                                 entry_timeframe=EntryTimeframe,
                                 entry_rsi_window=EntryRsi_Window,
                                 entry_green_window=EntryGreen_Window,
                                 entry_red_window=EntryRed_Window,
                                 entry_band_window=EntryBand_Window,
                                 trend_mode=TrendMode,
                                 entry_mode=EntryMode,
                                 atr_mult=ATR_Mult,
                                 #
                                 #  Additional Settings
                                 _size=BET_SIZE,
                                 _size_type=SIZE_TYPE,
                                 _init_cash=INIT_CASH,
                                 _fees=FEES,
                                 _slippage=SLIPPAGE,
                                 _tick_frequency=TICK_FREQUENCY,
                                 #
                                 #  Optimization Settings
                                 _use_ray=USE_RAY,
                                 _ray_kwargs=RAY_KWARGS,
                                 _run_n_samples=RUN_N_SAMPLES,
                                 _clear_cache_every_n=CLEAR_CACHE_EVERY_N,
                                 _collect_garbage_every_n=COLLECT_GARBAGE_EVERY_N,
                                 )

    perf.to_csv(f"Reports/{csv_data_name}_RLGL_Perf_{i_epoch}.csv")
    print(perf)

    del perf
    # Delete vbt cache
    vbt.CAQueryDelegator().clear_cache()

# Delete Ray cache
if USE_RAY:
    ray.shutdown()

# Combine the csv files
df = combine_csv_files_dask(
    file_paths="Reports/*.csv",
    save_path="AllReports.csv",
    save=True,
    dtype={"TrendTimeframe": "category", "EntryTimeframe": "category",
           "TrendMode": "category", "EntryMode": "category",
           "TrendRsi_Window": "int64", "TrendGreen_Window": "int64",
           "TrendBand_Window": "int64",
           "EntryRsi_Window": "int64", "EntryGreen_Window": "int64",
           "EntryRed_Window": "int64", "EntryBand_Window": "int64",
           "ATR_Mult": "float64", }
)

# Print the dataframe
print(df)

# Clear Report folder
for file in glob("Reports/*.csv"):
    remove(file)
