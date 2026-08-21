import datetime
import os
import numpy as np
import pandas as pd
import yfinance as yf
import pickle

from .swti import SWTI

class stocksdata():

    def __init__(self, startdate, enddate, period=[2, 5, 10, 20], raw = 'stock', raw2 = 'sp100'):

        self.raw = raw
        self.period = period

        self.startdate = startdate
        self.enddate = enddate

        self.year = str(datetime.strptime(self.startdate, "%Y-%m-%d").year)
        self.year2 = str(datetime.strptime(self.enddate, "%Y-%m-%d").year)

        self.basic_data = None

        self.raw = f'{raw}_{self.year}_{self.year2}'
        self.raw2 = f'{raw2}_{self.year}_{self.year2}'

    def multi_screen(self):

        swti_obj = SWTI(self.basic_data, self.period)
        indicators = swti_obj.ind_set()

        return indicators

    def load_data_multi_screen(self):

        org_data = self.load_data()
        # ben_data = load_data(ben_period, start_date, end_date)

        self.basic_data = org_data

        path = 'saved_values.pkl'
        if os.path.exists(path):
            values = pickle.load(open(path, 'rb'))
        else:
            values = self.triple_screen()
            pickle.dump(values, open(path, 'wb'))

        return values, self.basic_data

    def load_data(self):

        valid_data = {}
        txt, pkl = f"data/{self.raw}.txt", f"data/{self.raw}.pkl"
        stocks = [line.strip() for line in open(txt)]

        for i in stocks:
            filename = f"data/{i}.pkl"
            if os.path.exists(filename):
                print(f"Loading cached file: {filename}", end='')
                df = pd.read_pickle(filename)
            else:
                df = yf.download(i, start=self.startdate, end=self.enddate, interval="1d", auto_adjust=False)
                df.to_pickle(filename)
                print(f"Saved {filename}")

            if not df.isna().all().all():
                print(' added')
                valid_data[i] = df
            else:
                print(' not added')

        aligned_data = pd.concat(valid_data.values(), axis=1, keys=valid_data.keys(), join="outer")
        aligned_data = aligned_data.replace(0, pd.NA).ffill()
        data = aligned_data.to_numpy().reshape(len(aligned_data),
                                               len(aligned_data.columns.levels[0]),
                                               len(aligned_data.columns.levels[1])).transpose(1, 0, 2)

        print("Shape:", data.shape)  # (时间长度, 股票数量, 特征数量)

        return data.astype(np.float64)
