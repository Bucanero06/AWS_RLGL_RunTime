import gc
from time import perf_counter

import vectorbtpro as vbt
import numpy as np
from numpy import nan
from pandas import DataFrame
from vectorbtpro.portfolio.enums import SizeType


def Red_Light_Green_Light_Strat(data,
                                trend_timeframe, trend_rsi_window, trend_green_window, trend_band_window,
                                entry_timeframe, entry_rsi_window, entry_green_window, entry_red_window,
                                entry_band_window, trend_mode, entry_mode, atr_mult,
                                # arguements for the strategy
                                _tick_frequency, _init_cash, _fees, _slippage,
                                _size_type, _size, return_full_stats=True
                                ):
    ##  **Note, using vbt's method does not result in the desired output;
    #           trend_timeframe_data = data.resample(trend_timeframe)
    #           entry_timeframe_data = data.resample(entry_timeframe)
    #   **however since we know what to expect, we can disable the autofill and do it "manually"
    # todo - dropna should be all that is needed but either way we do not make use of all the price values (only close)
    #  ; also make the below code more efficient (mem & re-computation)
    #   previously solved!. Also I am not using caching here for resampling.

    # Resampling the data
    trend_timeframe_close = data.close.vbt.resample_apply(trend_timeframe,
                                                          vbt.nb.last_reduce_nb).dropna() if trend_timeframe != '1 min' else data.close
    trend_timeframe_low = data.low.vbt.resample_apply(trend_timeframe,
                                                      vbt.nb.min_reduce_nb).dropna() if trend_timeframe != '1 min' else data.low
    trend_timeframe_high = data.high.vbt.resample_apply(trend_timeframe,
                                                        vbt.nb.max_reduce_nb).dropna() if trend_timeframe != '1 min' else data.high
    #
    entry_timeframe_close = data.close.vbt.resample_apply(entry_timeframe,
                                                          vbt.nb.last_reduce_nb).dropna() if entry_timeframe != '1 min' else data.close

    resampler_dict = dict()
    for tf, resampled_close in zip([trend_timeframe, entry_timeframe], [trend_timeframe_close, entry_timeframe_close]):
        resampler_dict[tf] = vbt.Resampler(
            resampled_close.index,
            data.close.index,
            source_freq=tf,
            target_freq=_tick_frequency) if tf != _tick_frequency else None

    """Using the Trend Timeframe"""
    # Trend Computation (resample using TrendTimeframe; Default is 'h1'):
    TrendRsi = vbt.RSI.run(trend_timeframe_close, trend_rsi_window).rsi
    TrendGreen = vbt.MA.run(TrendRsi, trend_green_window)
    TrendBB = vbt.talib('BBANDS').run(TrendRsi, trend_band_window, 1.6185, 1.6185, 0)

    # Trend Signals (multiple):
    # Signals are Masks, optimize by selecting either Long_1, Long_2...
    TrendLong_0 = TrendBB.middleband_above(50)
    TrendLong_1 = TrendGreen.ma_above(TrendBB.middleband)
    TrendLong_2 = TrendLong_0 | TrendLong_1
    TrendLong_3 = TrendLong_0 & TrendLong_1
    #
    TrendShort_0 = TrendBB.middleband_below(50)
    TrendShort_1 = TrendGreen.ma_below(TrendBB.middleband)
    TrendShort_2 = TrendShort_0 | TrendShort_1
    TrendShort_3 = TrendShort_0 & TrendShort_1
    #
    #
    TrendLong = [TrendLong_0, TrendLong_1, TrendLong_2, TrendLong_3]
    TrendShort = [TrendShort_0, TrendShort_1, TrendShort_2, TrendShort_3]

    """Using the Entry Timeframe"""
    # Entry Computation (resample using EntryTimeframe; Default is 'm5'):
    EntryRsi = vbt.RSI.run(entry_timeframe_close, entry_rsi_window).rsi
    EntryGreen = vbt.MA.run(EntryRsi, entry_green_window)
    EntryRed = vbt.MA.run(EntryRsi, entry_red_window)
    # We calculate a Bollinger Band of the Rsi, only want the "Middle" line
    EntryBB = vbt.talib('BBANDS').run(EntryRsi, entry_band_window, 1.6185, 1.6185, 0)

    # Entry Signals (multiple):
    EntryLong_0 = EntryGreen.ma.vbt.crossed_above(EntryRed.ma)
    EntryLong_1 = EntryLong_0 & EntryBB.middleband_above(50)
    EntryLong_2 = EntryLong_0 & EntryGreen.ma_above(EntryBB.middleband)
    #
    EntryShort_0 = EntryGreen.ma.vbt.crossed_below(EntryRed.ma)
    EntryShort_1 = EntryShort_0 & EntryBB.middleband_below(50)
    EntryShort_2 = EntryShort_0 & EntryGreen.ma_below(EntryBB.middleband)
    #
    #
    EntryLong = [EntryLong_0, EntryLong_1, EntryLong_2]
    EntryShort = [EntryShort_0, EntryShort_1, EntryShort_2]

    del EntryRsi, EntryGreen, EntryRed, EntryBB, EntryLong_0, EntryLong_1, EntryLong_2, EntryShort_0, EntryShort_1, EntryShort_2
    del TrendRsi, TrendGreen, TrendBB, TrendLong_0, TrendLong_1, TrendLong_2, TrendLong_3, TrendShort_0, TrendShort_1, TrendShort_2, TrendShort_3
    gc.collect()

    """TP and SL using ATR and Trend Timeframe"""
    # Calculate the ATR and Resample to the original timeframe
    ATR = vbt.ATR.run(trend_timeframe_high, trend_timeframe_low, trend_timeframe_close, 10).atr

    """Combining the Signals and Timeframes"""
    # Open New Order Signals (using this approach, we should resample prior to comparing them here, we will only use the
    #   close price of the desired signal for the resampling and will need to properly use shift to align bars)
    # After we have signals in the last step we will shift one bar forward to avoid lookahead bias
    #
    # e.g. without any shifting we would have a signal at 10:00:00, but the entry would be at 10:05:00 (if the timeframe is 5min)
    #   Before:
    #     2019-12-31 21:30:00+00:00    False
    #     2019-12-31 21:35:00+00:00     True *** Signal
    #     2019-12-31 21:40:00+00:00    False
    #
    #   After resampling:
    #     2019-12-31 21:30:00+00:00     True *** Signal came too early (5 minutes too early)
    #     2019-12-31 21:31:00+00:00     True
    #     2019-12-31 21:32:00+00:00     True
    #     2019-12-31 21:33:00+00:00     True
    #     2019-12-31 21:34:00+00:00    False
    #     2019-12-31 21:35:00+00:00    False
    #     2019-12-31 21:36:00+00:00    False
    #     2019-12-31 21:37:00+00:00    False
    #     2019-12-31 21:38:00+00:00    False
    #     2019-12-31 21:39:00+00:00     True *** Signal came too early
    #     2019-12-31 21:40:00+00:00     True
    #
    # e.g. with .vbt.signals.fshift(-1) before but no .vbt.signals.fshift() after resampling
    #   Before:
    #     2019-12-31 21:30:00+00:00    False
    #     2019-12-31 21:35:00+00:00     True *** Signal
    #     2019-12-31 21:40:00+00:00    False
    #
    #   After shifting backwards and resampling:
    #     2019-12-31 21:30:00+00:00    False
    #     2019-12-31 21:31:00+00:00    False
    #     2019-12-31 21:32:00+00:00    False
    #     2019-12-31 21:33:00+00:00    False
    #     2019-12-31 21:34:00+00:00     True *** Signal came early by 1 minute, this is due to the resampling shifting
    #     2019-12-31 21:35:00+00:00     True
    #     2019-12-31 21:36:00+00:00     True
    #     2019-12-31 21:37:00+00:00     True
    #     2019-12-31 21:38:00+00:00     True
    #     2019-12-31 21:39:00+00:00    False
    #     2019-12-31 21:40:00+00:00    False
    #
    # e.g. with .vbt.signals.fshift(-1) and .vbt.signals.fshift()
    #   Before:
    #     2019-12-31 21:30:00+00:00    False
    #     2019-12-31 21:35:00+00:00     True *** Signal
    #     2019-12-31 21:40:00+00:00    False
    #
    #   After:
    #     2019-12-31 21:30:00+00:00    False
    #     2019-12-31 21:31:00+00:00    False
    #     2019-12-31 21:32:00+00:00    False
    #     2019-12-31 21:33:00+00:00    False
    #     2019-12-31 21:34:00+00:00    False
    #     2019-12-31 21:35:00+00:00     True *** Signal came at the right time when it was available
    #     2019-12-31 21:36:00+00:00     True
    #     2019-12-31 21:37:00+00:00     True
    #     2019-12-31 21:38:00+00:00     True
    #     2019-12-31 21:39:00+00:00     True
    #     2019-12-31 21:40:00+00:00    False
    #
    # We will use the .vbt.signals.fshift() method one more time to shift the signals one bar forward to avoid
    # lookahead bias since we will be using the close price of the previous bar to enter the trade meaning the signal
    # will be available one bar later

    if resampler_dict[trend_timeframe]:
        # print(TrendLong[trend_mode]["2019-12-31 21:30:00+00:00":"2019-12-31 21:40:00+00:00"])
        TrendLong[trend_mode] = TrendLong[trend_mode].vbt.signals.fshift(-1).vbt.resample_closing(
            resampler_dict[trend_timeframe]).replace(nan, 0).astype(bool)
        # print(TrendLong[trend_mode]["2019-12-31 21:30:00+00:00":"2019-12-31 21:40:00+00:00"])
        #
        TrendShort[trend_mode] = TrendShort[trend_mode].vbt.signals.fshift(-1).vbt.resample_closing(
            resampler_dict[trend_timeframe]).replace(nan, 0).astype(bool).vbt.signals.fshift()
        #
        # Replacing the original NaN with 1, there is no right method but at least we are consistent and the likelyhood
        # of a trade this early is unlikely as well as the probability of random noise triggering the tp or sl is approx
        # 16%
        ATR = ATR.shift(-1).vbt.resample_closing(resampler_dict[trend_timeframe]).replace(nan, 1).shift()

    if resampler_dict[entry_timeframe]:
        EntryLong[entry_mode] = EntryLong[entry_mode].vbt.signals.fshift(-1).vbt.resample_closing(
            resampler_dict[entry_timeframe]).replace(
            nan, 0).astype(bool).vbt.signals.fshift()
        #
        EntryShort[entry_mode] = EntryShort[entry_mode].vbt.signals.fshift(-1).vbt.resample_closing(
            resampler_dict[entry_timeframe]).replace(
            nan, 0).astype(bool).vbt.signals.fshift()

    GoLong = DataFrame(TrendLong[trend_mode] & EntryLong[entry_mode])
    GoShort = DataFrame(TrendShort[trend_mode] & EntryShort[entry_mode])

    # we are going to be using the closing price to determine the tp and sl percentages
    tp_sl = (ATR * atr_mult) / data.close
    Stop_Loss_perc = tp_sl
    Take_Profit_perc = tp_sl

    # Shift the signals one bar forward to avoid lookahead bias
    GoLong = GoLong.vbt.signals.fshift()
    GoShort = GoShort.vbt.signals.fshift()
    Stop_Loss_perc = Stop_Loss_perc.shift()
    Take_Profit_perc = Take_Profit_perc.shift()

    # Create a new order for each signal
    pf = vbt.Portfolio.from_signals(data, entries=GoLong, exits=GoShort, size=_size, fees=_fees,
                                    upon_opposite_entry=3,
                                    slippage=_slippage, freq=_tick_frequency,
                                    sl_stop=Stop_Loss_perc, tp_stop=Take_Profit_perc,
                                    direction='both', group_by=False, cash_sharing=False,
                                    allow_partial=False, log=False,
                                    ##!!THESE ARE NOT BEING USED IN THE PIPELINE!!##
                                    init_cash=_init_cash,
                                    size_type=_size_type,
                                    ##!!THESE ARE NOT BEING USED IN THE PIPELINE!!##
                                    )
    gc.collect()

    if not return_full_stats:
        return pf.sharpe_ratio
    return pf.stats(agg_func=None).replace([np.inf, -np.inf], nan)


def run_RLGL_Optimization(data,
                          trend_timeframe,
                          trend_rsi_window,
                          trend_green_window,
                          trend_band_window,
                          entry_timeframe,
                          entry_rsi_window,
                          entry_green_window,
                          entry_red_window,
                          entry_band_window,
                          trend_mode,
                          entry_mode,
                          atr_mult,
                          #
                          _size,
                          _size_type=SizeType.Value,
                          _fees=0,
                          _slippage=0,
                          _tick_frequency='1T',
                          _init_cash='auto',
                          #
                          _use_ray=False,
                          _ray_kwargs=None,
                          _run_n_samples=None,
                          _clear_cache_every_n=None,
                          _collect_garbage_every_n=None,
                          #
                          return_full_stats=True,  # return the full stats dataframe in row format else concat sharpes
                          ):
    print("Running RLGL_Optimization")
    #
    start = perf_counter()
    #
    print("trend_timeframe: ", trend_timeframe)
    print("trend_rsi_window: ", trend_rsi_window)
    print("trend_green_window: ", trend_green_window)
    print("trend_band_window: ", trend_band_window)
    print("entry_timeframe: ", entry_timeframe)
    print("entry_rsi_window: ", entry_rsi_window)
    print("entry_green_window: ", entry_green_window)
    print("entry_red_window: ", entry_red_window)
    print("entry_band_window: ", entry_band_window)
    print("trend_mode: ", trend_mode)
    print("entry_mode: ", entry_mode)
    print("atr_mult: ", atr_mult)
    try:
        total_combinations = len(trend_timeframe) * len(trend_rsi_window) * len(trend_green_window) * len(
            trend_band_window) * len(entry_timeframe) * len(entry_rsi_window) * len(entry_green_window) * len(
            entry_red_window) * len(entry_band_window) * len(trend_mode) * len(entry_mode) * len(atr_mult)
        print(f'{total_combinations = }')
    except:
        pass

    # @todo: condense this into a single function
    if _use_ray:
        if _ray_kwargs is None:
            from multiprocessing import cpu_count
            _ray_kwargs = {
                # 'address': 'auto',
                'num_cpus': cpu_count() - 2,
                # 'memory': 100 * 10 ** 9,
                # 'object_store_memory': 100 * 10 ** 9,
            },
        parametrized_Red_Light_Green_Light_Entries = vbt.parameterized(
            Red_Light_Green_Light_Strat,
            merge_func="row_stack" if return_full_stats else "concat",
            show_progress=True,
            random_subset=_run_n_samples,
            engine='ray',
            init_kwargs=_ray_kwargs
        )
    else:
        parametrized_Red_Light_Green_Light_Entries = vbt.parameterized(
            Red_Light_Green_Light_Strat,
            merge_func="row_stack" if return_full_stats else "concat",
            show_progress=True,
            random_subset=_run_n_samples,
        )

    perf = parametrized_Red_Light_Green_Light_Entries(
        data,
        trend_timeframe=vbt.Param(trend_timeframe),
        trend_rsi_window=vbt.Param(trend_rsi_window),
        trend_green_window=vbt.Param(trend_green_window),
        trend_band_window=vbt.Param(trend_band_window),
        entry_timeframe=vbt.Param(entry_timeframe),
        entry_rsi_window=vbt.Param(entry_rsi_window),
        entry_green_window=vbt.Param(entry_green_window),
        entry_red_window=vbt.Param(entry_red_window),
        entry_band_window=vbt.Param(entry_band_window),
        trend_mode=vbt.Param(trend_mode),
        entry_mode=vbt.Param(entry_mode),
        atr_mult=vbt.Param(atr_mult),
        #
        _fees=_fees,
        _slippage=_slippage,
        _tick_frequency=_tick_frequency,
        _init_cash=_init_cash,
        _size_type=_size_type,
        _size=_size,
        #
        return_full_stats=return_full_stats,
        #
        _engine_kwargs=dict(
            clear_cache=_clear_cache_every_n,
            collect_garbage=_collect_garbage_every_n,
        )
    )

    print(f'Elapsed time: {perf_counter() - start} seconds')

    # Clean Memory
    del data
    del trend_timeframe
    del trend_rsi_window
    del trend_green_window
    del trend_band_window
    del entry_timeframe
    del entry_rsi_window
    del entry_green_window
    del entry_red_window
    del entry_band_window
    del trend_mode
    del entry_mode
    del atr_mult
    del _size
    del _size_type
    del _fees
    del _slippage
    del _tick_frequency
    del _init_cash
    del _use_ray
    del _ray_kwargs
    del _run_n_samples
    del _clear_cache_every_n
    del _collect_garbage_every_n
    del return_full_stats
    del start
    del parametrized_Red_Light_Green_Light_Entries
    gc.collect()

    return perf.vbt.drop_redundant_levels()


