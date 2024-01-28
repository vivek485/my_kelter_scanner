import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import time
import ta
from ta.momentum import stoch
import asyncio
import aiohttp

import nest_asyncio
from list import symbols
from ta.volatility import keltner_channel_hband , keltner_channel_lband, keltner_channel_mband
from ta.trend import macd, macd_diff


def gr(df):
    fig = go.Figure(data=[go.Candlestick(x=df.name,open=df.Open,close=df.Open,high=df.High,low=df.Low,name='Candelstick'),
                          go.Scatter(x=df.name,y=df.ma200,line=dict(color='orange',width = 1),name='Ma200'),
                          go.Scatter(x=df.name,y=df.highband,line=dict(color='blue',width = 1),name='Upperband'),
                          go.Scatter(x=df.name,y=df.middleband,line=dict(color='blue',width = 1),name='Middleband'),
                          go.Scatter(x=df.name,y=df.lowerband,line=dict(color='blue',width = 1),name='Lowerband')])
    return st.plotly_chart(fig)

st. set_page_config(layout="wide")
st.title("Screener")
dt = st.number_input(label='days_back',min_value=1,max_value=50)

myst = []
dic_buy = {'buy_symbol':[],'buydate':[]}
dic_sell = {'sell_symbol':[],'selldate':[]}
interval = 60  # enter 15,60,240,1440,10080,43800
dayback = 300
ed = datetime.now()
stdate = ed - timedelta(dayback)


def conunix(ed):
    ed1 = str(round(time.mktime(ed.timetuple())))
    ed1 = (ed1[:-1])
    ed1 = (ed1 + '0000')
    return ed1


fromdate = conunix(stdate)
todate = conunix(ed)
stt = time.time()



async def getdata(session, stock):


    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.5',
        #'Accept-Encoding': 'gzip, deflate, br'
    }
    url = f'https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/{stock}?endTimeInMillis={todate}&intervalInMinutes={interval}&startTimeInMillis={fromdate}'
    async with session.get(url, headers=headers) as response:
        try:
            resp = await response.json()
            candle = resp['candles']
            dt = pd.DataFrame(candle)
            fd = dt.rename(columns={0: 'time', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'})
            tim = []
            for each in fd['time']:
                a = each
                a = datetime.fromtimestamp(a).strftime('%Y-%m-%d %H:%M:%S')
                tim.append(a)
            dt = pd.DataFrame(tim)
            fd = pd.concat([dt, fd['time'], fd['Open'], fd['High'], fd['Low'], fd['Close'], fd['Volume']],
                           axis=1).rename(columns={0: 'datetime'})
            fd['symbol'] = stock
            pd.options.mode.chained_assignment = None
            final_df = fd

            final_df['Open'] = final_df['Open'].astype(float)
            final_df['Close'] = final_df['Close'].astype(float)
            final_df['High'] = final_df['High'].astype(float)
            final_df['Low'] = final_df['Low'].astype(float)
            final_df['Volume'] = final_df['Volume'].astype(float)
            final_df['datetime'] = pd.to_datetime(final_df.datetime)  # final_df['datetime']#.astype('datetime64[ns]')

            final_df.set_index(final_df.datetime, inplace=True)
            final_df.drop(['time', 'datetime'], axis=1, inplace=True)

            final_df['prevlow'] = final_df['Low'].shift(1)
            final_df['prevhigh'] = final_df['High'].shift(1)
            final_df['prevlow1'] = final_df['Low'].shift(2)
            final_df['prevhigh1'] = final_df['High'].shift(2)
            final_df['prevlow2'] = final_df['Low'].shift(3)
            final_df['prevhigh2'] = final_df['High'].shift(3)
            final_df['prevclose'] = final_df['Close'].shift(1)
            # ma200.stochiastic
            final_df['ma200'] = round(ta.trend.ema_indicator(final_df.Close,200))

            final_df['st'] = round(
                ta.momentum.stoch(high=final_df['High'], low=final_df['Low'], close=final_df['Close'], window=14,
                                  smooth_window=3),
                2)
            final_df['prst'] = final_df['st'].shift(1)
            #ema of h/l
            hl = (final_df['High'] + final_df['Low']) / 2
            final_df['hl'] = ta.trend.ema_indicator(hl,4)
            final_df['prhl'] = final_df['hl'].shift(1)

            # kelterchannel
            kl = ta.volatility.KeltnerChannel(close=final_df.Close, high=final_df.High, low=final_df.Low,
                                              original_version=False)
            final_df['highband'] = round(kl.keltner_channel_hband())
            final_df['middleband'] = round(kl.keltner_channel_mband())
            final_df['lowerband'] = round(kl.keltner_channel_lband())
            # macd
            final_df['macd_ind'] = ta.trend.macd_diff(close=final_df['Close'], window_slow=26, window_fast=12,
                                                      window_sign=9,
                                                      fillna=False)
            final_df['prevmacd'] = final_df['macd_ind'].shift(1)


            final_df['symbol'] = stock
            
            # conditions
            final_df['sig_sell'] = np.where((final_df.Close < final_df.hl), 1, 0)
            final_df['sig_buy'] = np.where((final_df.Close > final_df.hl), 2, 0)
            final_df['sig_sellst'] = np.where(((final_df.st < 90) & (final_df.prst > final_df.st) & (final_df.st > 50)),
                                              1, 0)
            final_df['sig_buyst'] = np.where(((final_df.st > 10) & (final_df.prst < final_df.st) & (final_df.st < 50)),
                                             2, 0)
            final_df['sig_sellmacd'] = np.where((final_df.macd_ind < final_df.prevmacd), 1, 0)
            final_df['sig_buymacd'] = np.where((final_df.macd_ind > final_df.prevmacd), 2, 0)
            final_df['sellsig'] = (final_df.sig_sell + final_df.sig_sellst + final_df.sig_sellmacd)
            final_df['buysig'] = (final_df.sig_buy + final_df.sig_buyst + final_df.sig_buymacd)
            final_df['prbuysig'] = final_df['buysig'].shift(1)
            final_df['prsellsig'] = final_df['sellsig'].shift(1)
            final_df['hlpreviousbuy'] = np.where(
                ((final_df.Close > final_df.hl) & (final_df.prevclose < final_df.prhl)), 1, 0)
            final_df['hlprevioussell'] = np.where(
                ((final_df.Close < final_df.hl) & (final_df.prevclose > final_df.prhl)), 1, 0)

            final_df['signal1'] = np.where(((final_df.buysig == 6) & (final_df.prbuysig != 6)), 1, np.nan)
            final_df['signal2'] = np.where(((final_df.sellsig == 3) & (final_df.prsellsig != 3)), 2, np.nan)

            final_df['signal1p'] = np.where(((final_df.buysig == 6) & (final_df.prbuysig != 6)), final_df['Low'], np.nan)
            final_df['signal2p'] = np.where(((final_df.sellsig == 3) & (final_df.prsellsig != 3)), final_df['High'], np.nan)
#-----------------------------------------------------------------------------------------------------------------------

            # fig = go.Figure(data=[
            #     go.Candlestick(x=final_df.index, open=final_df.Open, close=final_df.Open, high=final_df.High,
            #                    low=final_df.Low, name=stock),
            #     go.Scatter(x=final_df.index, y=final_df.ma200, line=dict(color='red', width=1), name='Ma200'),
            #     go.Scatter(x=final_df.index, y=final_df.highband, line=dict(color='blue', width=1),
            #                name='Upperband'),
            #     go.Scatter(x=final_df.index, y=final_df.middleband, line=dict(color='blue', width=1),
            #                name='Middleband'),
            #     go.Scatter(x=final_df.index, y=final_df.lowerband, line=dict(color='blue', width=1),
            #                name='Lowerband')])
            # fig.add_trace(go.Scatter(x=final_df.index,y=final_df.signal1p,mode='markers',marker_symbol='triangle-up',marker_size=13))
            # fig.add_trace(go.Scatter(x=final_df.index, y=final_df.signal2p, mode='markers', marker_symbol='triangle-down',
            #                          marker_size=13))
            #
            # fig.update_layout(autosize=False, width=1800, height=800, xaxis_rangeslider_visible=False)
            # fig.layout.xaxis.type = 'category'
            #
            # st.title(stock)
            # st.plotly_chart(fig)

            by = final_df[final_df.signal1 == 1].iloc[-1]

            dic_buy['buy_symbol'].append(by.symbol)
            dic_buy['buydate'].append(by.name)
            #st.write(by)





            #print(by.name)
            sl = final_df[final_df.signal2 == 2].iloc[-1]


            dic_sell['sell_symbol'].append(sl.symbol)
            dic_sell['selldate'].append(sl.name)
            #st.write(sl)





            final_df = final_df.dropna()
            final_df = final_df.iloc[-5:]

            #
            #


            last_candle = final_df.sum()#final_df.iloc[-1]

            if last_candle['signal1'] != 0:
                st.write(last_candle['symbol'] + ' buymy_kelter_stratergy  ')
                fig = go.Figure(data=[
                    go.Candlestick(x=final_df.index, open=final_df.Open, close=final_df.Open, high=final_df.High,
                                   low=final_df.Low,
                                   name=stock),
                    go.Scatter(x=final_df.index, y=final_df.ma200, line=dict(color='red', width=1), name='Ma200'),
                    go.Scatter(x=final_df.index, y=final_df.highband, line=dict(color='blue', width=1),
                               name='Upperband'),
                    go.Scatter(x=final_df.index, y=final_df.middleband, line=dict(color='blue', width=1),
                               name='Middleband'),
                    go.Scatter(x=final_df.index, y=final_df.lowerband, line=dict(color='blue', width=1),
                               name='Lowerband')])
                fig.add_trace(
                    go.Scatter(x=final_df.index, y=final_df.signal1p, mode='markers', marker_symbol='triangle-up',
                               marker_size=15))
                fig.update_layout(autosize=False, width=1800, height=800, xaxis_rangeslider_visible=False)
                fig.layout.xaxis.type = 'category'
                st.title(stock)
                st.plotly_chart(fig)

                myst.append(last_candle['symbol'])
            if last_candle['signal2'] != 0:
                st.write(last_candle['symbol'] + ' sellmy_kelter_stratergy  ')
                fig = go.Figure(data=[
                    go.Candlestick(x=final_df.index, open=final_df.Open, close=final_df.Open, high=final_df.High,
                                   low=final_df.Low, name=stock),
                    go.Scatter(x=final_df.index, y=final_df.ma200, line=dict(color='red', width=1), name='Ma200'),
                    go.Scatter(x=final_df.index, y=final_df.highband, line=dict(color='blue', width=1),
                               name='Upperband'),
                    go.Scatter(x=final_df.index, y=final_df.middleband, line=dict(color='blue', width=1),
                               name='Middleband'),
                    go.Scatter(x=final_df.index, y=final_df.lowerband, line=dict(color='blue', width=1),
                               name='Lowerband')])
                fig.add_trace(
                    go.Scatter(x=final_df.index, y=final_df.signal2p, mode='markers', marker_symbol='triangle-down',
                               marker_size=15))
                fig.update_layout(autosize=False, width=1800, height=800, xaxis_rangeslider_visible=False)
                fig.layout.xaxis.type = 'category'
                st.title(stock)
                st.plotly_chart(fig)

                myst.append(last_candle['symbol'])

            return
        except:
            pass


async def main():
    async with aiohttp.ClientSession() as session:

        tasks = []
        for stocks in symbols:
            try:
                stock = stocks

                task = asyncio.ensure_future(getdata(session, stock))

                tasks.append(task)
            except:
                pass
        df = await asyncio.gather(*tasks)
        # print(df)


nest_asyncio.apply()
button = st.button(label='Run_kelter_channel',key='kelter')
if button:

    asyncio.run(main())
    st.write(myst)
    bcdate = ed - timedelta(dt)


    d_buydf = pd.DataFrame(dic_buy)

    d_selldf = pd.DataFrame(dic_sell)
    col1,col2 = st.columns(2,gap='small')

    col1.write(d_buydf[d_buydf.buydate > bcdate])
    col2.write(d_selldf[d_selldf.selldate > bcdate])




# print('mykelter')
# print(myst)
# bcdate = ed - timedelta(1)
# print(bcdate)
#
# d_buydf = pd.DataFrame(dic_buy)
#
#
# d_selldf = pd.DataFrame(dic_sell)
#
# print(d_buydf[d_buydf.buydate > bcdate])
# print(d_selldf[d_selldf.selldate > bcdate])
#
# ett = time.time()
# totaltime = ett - stt
# wait = 100 - totaltime
# print(('time taken  '), totaltime)
# print('sleeping')
# time.sleep(wait)

#
# fig = go.Figure(data=[
#     go.Candlestick(x=final_df.index, open=final_df.Open, close=final_df.Open, high=final_df.High, low=final_df.Low,
#                    name=stock),
#     go.Scatter(x=final_df.index, y=final_df.ma200, line=dict(color='red', width=1), name='Ma200'),
#     go.Scatter(x=final_df.index, y=final_df.highband, line=dict(color='blue', width=1), name='Upperband'),
#     go.Scatter(x=final_df.index, y=final_df.middleband, line=dict(color='blue', width=1), name='Middleband'),
#     go.Scatter(x=final_df.index, y=final_df.lowerband, line=dict(color='blue', width=1), name='Lowerband')])
# fig.update_layout(autosize=False, width=1800, height=800, xaxis_rangeslider_visible=False)
# fig.layout.xaxis.type = 'category'
# st.title(stock)
# st.plotly_chart(fig)