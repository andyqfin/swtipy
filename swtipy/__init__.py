import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit
from functools import reduce

@njit(fastmath=True, cache=True)
def ema_func(x):
    if np.isnan(x).any(): return np.nan
    alpha, val = 2 / (len(x) + 1), x[0]
    return reduce(lambda v, xi: alpha * xi + (1 - alpha) * v, x[1:], x[0])

class SWTI():

    def __init__(self, data=None, time_periods=None):
        self.data = data
        self.time_periods = time_periods

    def set_time_periods(self, periods):
        self.time_periods = periods

    def set_data(self, data):
        self.data = data

    @staticmethod
    def nan_pad(x, sub_period):
        return np.pad(x, ((0, 0), (sub_period, 0), (0, 0)), constant_values=np.nan)

    def ind_set(self):

        indicators = {}
        indicators['cmf'], indicators['cmf_diff'], indicators['cmf_diff2'] = self.cmf_func()
        indicators['macd'], indicators['macd_diff'], indicators['macd_diff2'] = self.macd_func()

        indicators['rsi'], indicators['rsi_diff'], indicators['rsi_diff2'] = self.rsi_func()
        indicators['srsi'], indicators['srsi_diff'], indicators['srsi_diff2'] = self.stoch_rsi_func()

        indicators['mfi'], indicators['mfi_diff'], indicators['mfi_diff2'] = self.load_mfi_func()
        indicators['kdj'], indicators['kdj_diff'], indicators['kdj_diff2'] = self.kdj_set_params()

        indicators['wpr'], indicators['wpr_diff'], indicators['wpr_diff2'] = self.wpr_func()
        indicators['smi'], indicators['smi_diff'], indicators['smi_diff2'] = self.smi_func()
        indicators['dss'], indicators['dss_diff'], indicators['dss_diff2']  = self.dss_func()
        indicators['atr'], indicators['kc_mid'], indicators['natr'], indicators['natr_diff'], indicators['natr_diff2'] = self.atr_func()

        indicators['vola'], indicators['vola_diff'], indicators['vola_diff2'] = self.load_vola()


        return indicators

    def macd_func(self, fast=12, slow=26, signal=9, cross=True):

        print('macd function')
        x = self.data
        x = x[:, :, 1:2]
        assert fast < slow, "macd: short_period should be less than long_period"

        short_ema = np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(x, window_shape=fast, axis=1))
        long_ema = np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(x, window_shape=slow, axis=1))

        min_len = min(short_ema.shape[1], long_ema.shape[1])
        dif = short_ema[:, -min_len:, :] - long_ema[:, -min_len:, :]
        dem = np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(dif, window_shape=signal, axis=1))

        dif = self.nan_pad(dif, slow - 1)
        dem = self.nan_pad(dem, slow - 1 + signal - 1)
        osc = dif - dem

        if cross:
            osc = osc / x

        osc_diff, osc_diff2 = self.indicator_change(osc, -1, 1)
        # print(np.nanmin(osc_change), np.nanmax(osc_change))

        return osc, osc_diff, osc_diff2

    def smi_func(self, period=14, fast=3, slow=3):

        print('smi function')
        # SMI usage -100 to 100
        print('Stochastic Momentum Index (SMI)')
        # adj, close, high, low, open, volume = swap_data(data)

        x = self.data
        high, low, close = x[:, :, 2:3], x[:, :, 3:4], x[:, :, 1:2]

        high_slide = self.nan_pad(
            np.apply_along_axis(np.max, axis=3, arr=sliding_window_view(high, window_shape=period, axis=1)), period - 1)
        low_slide = self.nan_pad(
            np.apply_along_axis(np.min, axis=3, arr=sliding_window_view(low, window_shape=period, axis=1)), period - 1)

        midpoint = (high_slide + low_slide) / 2

        dn = close - midpoint
        dsm = self.nan_pad(np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(dn, window_shape=fast, axis=1)),
                           fast - 1)

        hld = high_slide - low_slide
        hld = self.nan_pad(np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(hld, window_shape=fast, axis=1)),
                           fast - 1)

        dsm_slide2 = self.nan_pad(
            np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(dsm, window_shape=slow, axis=1)), slow - 1)
        hld_slide2 = self.nan_pad(
            np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(hld, window_shape=slow, axis=1)), slow - 1)

        smi = dsm_slide2 / hld_slide2 * 200
        smi_diff, smi_diff2 = self.indicator_change(smi, -100, 100)

        return smi, smi_diff, smi_diff2

    def dss_func(self, period=9, fast=3, slow=3):

        print('Double Smoothed Stochastic (DSS) function')

        # value DSS 0 to 100
        # adj, close, high, low, open, volume = swap_data(data)

        x = self.data
        close, high, low = x[:, :, 1:2], x[:, :, 2:3], x[:, :, 3:4]

        high_slide = self.nan_pad(
            np.apply_along_axis(np.max, axis=3, arr=sliding_window_view(high, window_shape=period, axis=1)), period - 1)
        low_slide = self.nan_pad(
            np.apply_along_axis(np.min, axis=3, arr=sliding_window_view(low, window_shape=period, axis=1)), period - 1)

        up = close - low_slide
        down = high_slide - low_slide

        up_ema = self.nan_pad(np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(up, window_shape=fast, axis=1)),
                              fast - 1)
        down_ema = self.nan_pad(
            np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(down, window_shape=fast, axis=1)), fast - 1)

        up_ema2 = self.nan_pad(
            np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(up_ema, window_shape=slow, axis=1)), slow - 1)
        down_ema2 = self.nan_pad(
            np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(down_ema, window_shape=slow, axis=1)),
            slow - 1)

        dss = up_ema2 / down_ema2 * 100
        dss_diff, dss_diff2 = self.indicator_change(dss, 0, 100)

        return dss, dss_diff, dss_diff2

    def cmf_func(self, period=21):

        print('Chaikin Money Flow function')

        x = self.data

        # Chaikin Money Flow ranged from -1 to 1, numerator, denominator
        numer = (x[:, :, 1:2] - x[:, :, 3:4]) - (x[:, :, 2:3] - x[:, :, 1:2])
        denom = x[:, :, 2:3] - x[:, :, 3:4]

        mfm = np.divide(numer, denom, out=np.full_like(denom, np.nan), where=(denom != 0) & (~np.isnan(denom)))

        volume = x[:, :, 5:6]
        mfv = mfm * volume
        mfv = np.apply_along_axis(np.sum, axis=3, arr=sliding_window_view(mfv, window_shape=period, axis=1))
        volume_value = np.apply_along_axis(np.sum, axis=3, arr=sliding_window_view(volume, window_shape=period, axis=1))

        cmf = self.nan_pad(mfv / volume_value, period - 1)
        cmf_diff, cmf_diff2 = self.indicator_change(cmf, -1, 1)

        return cmf, cmf_diff, cmf_diff2

    def wpr_func(self, period=14):

        print('william percentage range function')

        # adj, close, high, low, open, volume = swap_data(data)
        x = self.data
        close = x[:, :, 1:2]
        high_n_days = self.nan_pad(
            np.apply_along_axis(np.max, axis=3, arr=sliding_window_view(x[:, :, 2:3], window_shape=period, axis=1)),
            period - 1)
        low_n_days = self.nan_pad(
            np.apply_along_axis(np.min, axis=3, arr=sliding_window_view(x[:, :, 3:4], window_shape=period, axis=1)),
            period - 1)

        wpr = (high_n_days - close) / (high_n_days - low_n_days) * -100
        wpr_diff, wpr_diff2 = self.indicator_change(wpr, -100, 0)

        return wpr, wpr_diff, wpr_diff2

    def atr_func(self, period=14):

        print('average true range function')

        # adj, close, high, low, open, volume = swap_data(data)
        # average true range

        x = self.data

        def kc_middle_func():

            close = x[:, :, 1:2]

            kc_middle = np.apply_along_axis(ema_func, axis=3,
                                            arr=sliding_window_view(close, window_shape=period, axis=1))

            kc_middle = self.nan_pad(kc_middle, period - 1)

            return kc_middle

        prev_close = self.nan_pad(x[:, :-1, 1:2], 1)
        atr_value = np.maximum(x[:, :, 2:3], prev_close) - np.minimum(x[:, :, 3:4], prev_close)

        atr_ema = np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(atr_value, window_shape=period, axis=1))
        atr_ema = self.nan_pad(atr_ema, period - 1)

        natr = atr_ema / x[:, :, 1:2]
        # print(np.nanmin(natr), np.nanmax(natr))
        natr_diff, natr_diff2 = self.indicator_change(natr, 0, 1)

        return atr_ema, kc_middle_func(), natr, natr_diff, natr_diff2

    def load_vola(self, period=60):

        print('volatility function')

        x = self.data

        def volatility(x):
            vola = np.apply_along_axis(np.std, axis=3, arr=sliding_window_view(x, window_shape=period, axis=1))
            return self.nan_pad(vola, period)

        vola = volatility(np.log(x[:, 1:, :] / x[:, :-1, :]))
        # print(np.nanmin(vola), np.nanmax(vola) )
        vola_diff, vola_diff2 = self.indicator_change(vola, 0, 2)
        # print(np.nanmin(vola_change), np.nanmax(vola_change) )

        return vola, vola_diff, vola_diff2

    def load_mfi_func(self, period=14):

        x = self.data

        # money flow index
        tp = (x[:, :, 2:3] + x[:, :, 3:4] + x[:, :, 1:2]) / 3
        tp_diff = self.nan_pad((tp[:, 1:, :] - tp[:, :-1, :]), 1)

        mf_shifted = self.nan_pad((tp * x[:, :, 5:6])[:, 1:, :], 1)

        pos_mf = self.nan_pad(np.apply_along_axis(np.sum, axis=3, arr=sliding_window_view(
            np.where(tp_diff > 0, mf_shifted, np.where(np.isnan(mf_shifted), np.nan, 0)), window_shape=period, axis=1)),
                              period - 1)
        neg_mf = self.nan_pad(np.apply_along_axis(np.sum, axis=3, arr=sliding_window_view(
            np.where(tp_diff < 0, mf_shifted, np.where(np.isnan(mf_shifted), np.nan, 0)), window_shape=period, axis=1)),
                              period - 1)

        # Money Flow Ratio，避免除零
        mfr = np.divide(pos_mf, neg_mf, out=np.full_like(pos_mf, np.inf, dtype=float), where=neg_mf != 0)
        mfi = 100 - 100 / (1 + mfr)

        mfi = np.where(neg_mf == 0, 100, mfi)  # 全部正流 → MFI=100
        mfi_diff, mfi_diff2 = self.indicator_change(mfi, 0, 100)

        return mfi, mfi_diff, mfi_diff2

    def kdj_set_params(self, k=9, d=3, d_slow=3):

        print('KDJ function')

        basic_data = self.data

        def kdj_indicator(x, period, period2, period3):

            def get_value(x):
                return np.nan if np.isnan(x).any() else x[-1]

            def get_max(x):
                return np.nan if np.isnan(x).any() else np.max(x)

            def get_min(x):
                return np.nan if np.isnan(x).any() else np.min(x)

            close = np.apply_along_axis(get_value, axis=3,
                                        arr=sliding_window_view(x[:, :, 1:2], window_shape=period, axis=1))

            high = np.apply_along_axis(get_max, axis=3,
                                       arr=sliding_window_view(x[:, :, 2:3], window_shape=period, axis=1))

            low = np.apply_along_axis(get_min, axis=3,
                                      arr=sliding_window_view(x[:, :, 3:4], window_shape=period, axis=1))

            k_value = self.nan_pad((close - low) / (high - low) * 100, period - 1)

            d_value = self.nan_pad(
                np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(k_value, window_shape=period2, axis=1)),
                period2 - 1)

            d_value_slow = self.nan_pad(
                np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(d_value, window_shape=period3, axis=1)),
                period3 - 1)

            j_value = 3 * k_value - 2 * d_value
            j_value2 = 3 * d_value - 2 * d_value_slow

            value = np.concatenate((k_value, d_value, d_value_slow, j_value, j_value2), axis=2)

            return value[:, :, 3:5]

        print('kdj', (k, d, d_slow), ' j only')
        kdj_all = kdj_indicator(basic_data, k, d, d_slow)

        kdj_diff, kdj_diff2 = self.indicator_change(kdj_all, -200, 300)

        return kdj_all, kdj_diff, kdj_diff2

    def rsi_func(self, period=10):

        print('RSI function')

        x = self.data

        xrsi = x[:, 1:, 1:2] - x[:, :-1, 1:2]

        gains = sliding_window_view(np.where(np.isnan(xrsi) | (xrsi > 0), xrsi, 0), window_shape=period, axis=1)
        loss = sliding_window_view(np.where(np.isnan(xrsi) | (xrsi < 0), -xrsi, 0), window_shape=period, axis=1)

        up_ma = np.apply_along_axis(ema_func, axis=3, arr=gains)
        down_ma = np.apply_along_axis(ema_func, axis=3, arr=loss)

        # 完全横盘，设为中性值 50  # 没有下跌，强势上涨 # 没有上涨，弱势下跌
        with np.errstate(divide='ignore', invalid='ignore'):
            rsi = np.where((up_ma == 0) & (down_ma == 0), 50,
                           np.where(down_ma == 0, 100, np.where(up_ma == 0, 0, up_ma / (up_ma + down_ma) * 100)))

        rsi = self.nan_pad(rsi, period - 1 + 1)
        print(np.shape(rsi))
        rsi_diff, rsi_diff2 = self.indicator_change(rsi, 0, 100)

        return rsi, rsi_diff, rsi_diff2

    def stoch_rsi_func(self, period=10):

        print('stoch RSI function')

        rsi, _, _ = self.rsi_func(period)
        rsi_slide = sliding_window_view(rsi, window_shape=period, axis=1)
        rsi_min = self.nan_pad(np.apply_along_axis(np.min, axis=3, arr=rsi_slide), period - 1)
        rsi_max = self.nan_pad(np.apply_along_axis(np.max, axis=3, arr=rsi_slide), period - 1)

        denominator = rsi_max - rsi_min

        with np.errstate(invalid='ignore', divide='ignore'):
            srsi = np.where(denominator == 0, 50, (rsi - rsi_min) / denominator)

        srsi_diff, srsi_diff2 = self.indicator_change(srsi, 0, 1)

        return srsi, srsi_diff, srsi_diff2

    def indicator_change(self, arr, vmin, vmax):

        def value_log_chance(N, diff):

            def MinMaxNorm_0to1(x):
                return (x - vmin) / (vmax - vmin)

            arr_norm = MinMaxNorm_0to1(arr)

            if diff == 1:
                diff_value = self.nan_pad(np.diff(arr_norm, axis=1, n=1), 1)
                ema_diff = self.nan_pad(
                    np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(diff_value, window_shape=N, axis=1)),
                    N - 1)
                return ema_diff

            if diff == 2:
                diff2_value = self.nan_pad(np.diff(arr_norm, axis=1, n=2), 2)
                ema_diff2 = self.nan_pad(
                    np.apply_along_axis(ema_func, axis=3, arr=sliding_window_view(diff2_value, window_shape=N, axis=1)),
                    N - 1)
                return ema_diff2

            return None

        ema_diff_set = np.concatenate([value_log_chance(N, diff=1) for N in self.time_periods], axis=2)
        ema_diff2_set = np.concatenate([value_log_chance(N, diff=2) for N in self.time_periods], axis=2)

        return ema_diff_set, ema_diff2_set

