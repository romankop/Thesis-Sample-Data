import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.reload_library()
plt.style.use(['science','no-latex'])
import seaborn as sns
from functools import reduce
from itertools import product

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error as mse

macro_columns = ['DFF', 'UNRATE', 'VIXCLS', 'TEDRATE',
 'DAAA', 'DBAA', 'def_spread', 'div_yield', 'DTB3',
 'DTB3_ma', 'rel_tbill', 'DGS3MO', 'DGS10', 'term_spread',
 'CPIAUCSL', 'GDP', 'S&P500_ret', 'S&P500_ret_diff',
 'excess_market_ret', 'abnormal_fund_ret']

# function for data preprocessing

def clean_data(data):

    MAX_FILING = pow(10, 13)
    MAX_SSH_PRN_AMP = pow(10, 8)
    MAX_MVAL = pow(10, 10)

    mvsum = data.groupby('access_id').market_value.sum().reset_index()

    data = data[(data.market_value > 0) & (data.ssh_prn_amt > 0) & \
                (data.ssh_prn_amt < MAX_SSH_PRN_AMP) & (data.market_value < MAX_MVAL) & \
                data.access_id.isin(mvsum[mvsum.market_value < MAX_FILING].access_id.unique())]

    data = data[data.security_class.eq("STOCK") & data.investment_discretion.eq("SOLE") & data.put_call.isna()]
    data = data[data['voting_authority_sole'].gt(0) & \
                (data['voting_authority_shared'] + data['voting_authority_none'] == 0)]

    data = data[data.ssh_prn_amt_type.eq('SH')]
    
    data = data.rename({'ssh_prn_amt': 'shares_num'}, axis=1)

    data['shares_num'] = data.shares_num.round()
    data['cik'] = data['cik'].astype('int')

    data['period'] = pd.to_datetime(data.conformed_period_of_report).dt.to_period('Q')

    data = data.groupby(['cik', 'period', 'id', 'access_id', 'conformed_submission_type']) \
        .agg({'shares_num': 'sum',
              'market_value': 'sum',
              'stamp': 'max',
              'close': 'max'}).reset_index()

    data = data.sort_values(by='stamp').groupby(['cik', 'period', 'id']).last().reset_index()
    
    data['close'] = data.market_value.div(data.shares_num, axis=0)
    
    data = data.merge(data.pivot(index='period', columns=['cik', 'id'], values='shares_num') \
                      .isna().unstack().reset_index(name='nid'), on=['cik', 'id', 'period'],
                      how='outer', suffixes=(False, False))

    data['shares_num'] = data.shares_num.fillna(0)
    data['market_value'] = data.market_value.fillna(0)

    return data

# function adding deviation from benchmark of the position in ticker

def add_bench_dev(data):

    ticker_value = data.groupby(['period', 'id'])['market_value'].sum().reset_index()

    ticker_value.rename(columns={'market_value': 'market_value_ticker'}, inplace=True)
    ticker_value['tickers_total_value'] = ticker_value.groupby(by='period')['market_value_ticker'].transform('sum')

    ticker_value['ticker_market_share'] = ticker_value.market_value_ticker.div(ticker_value.tickers_total_value, axis=0)

    ticker_value['period'] = ticker_value.period.astype(data.period.dtype)
    ticker_value['id'] = ticker_value['id'].astype(data['id'].dtype)

    data = data.merge(ticker_value, on=['period', 'id'], how='left', suffixes=(None, None))
    data['bench_dev'] = (data.share - data.ticker_market_share).div(2).abs()
    data['bench_dev_value'] = data.bench_dev.mul(data.market_value, axis=0)

    return data

# function evaluating active share metric

def add_active_share(data):

    data['active_share'] = data.groupby(['period', 'cik']).bench_dev.transform('sum')
    data['active_share'] += (1 - data.groupby(['period', 'cik']).ticker_market_share.transform('sum')).div(2)
    data['active_share_value'] = data.active_share.mul(data.aum, axis=0)

    return data

# functions implemeting ticker-based characterisitcs
    
def lf(data, unit='shares'):
    
    df = data.copy()
    
    if unit not in ('shares', 'count'):
        raise AttributeError(unit)

    thr = 0.5
    value = 'shares_num'

    df['liquid_fund'] = df.groupby(['cik', 'period']).active_share.transform('max').gt(thr)
    
    df[value] = df[value].apply(lambda x: np.nan if x == 0 else x)
    
    df['liquid_value'] = df[value].mul(df.liquid_fund, axis=0)

    func = 'sum' if unit == 'shares' else 'count'

    largest_values = df.groupby('id').liquid_value.transform(func)
    all_values = df.groupby('id')[value].transform(func)

    liquidity = largest_values.div(all_values, axis=0)

    return liquidity

def bosi(df, unit='shares'):

    if unit not in ('shares', 'count'):
        raise AttributeError(unit)

    value = 'diff_shares_num_lag_1'
    data = df.copy()

    if unit == 'shares':
        data[value] = data[value]
    else:
        data[value] = data[value].apply(np.sign)

    data['abs_' + value] = data[value].abs()

    return data.groupby(['id', 'period'])[value].transform('sum') \
            .div(data.groupby(['id', 'period'])['abs_' + value].transform('sum'), axis=0)

def busi(df, unit='shares'):

    if unit not in ('shares', 'count'):
        raise AttributeError(unit)

    value = 'diff_shares_num_lag_1'
    data = df.copy()

    if unit == 'shares':
        data[value] = data[value]
    else:
        data[value] = data[value].apply(np.sign)
        
    adjust = np.logical_xor(df.shares_num.eq(0), df.shares_num_lag_1.eq(0))
    
    data[value] = data[value].mul(adjust, axis=0)

    data['abs_' + value] = data[value].abs()

    return data.groupby(['id', 'period'])[value].transform('sum') \
            .div(data.groupby(['id', 'period'])['abs_' + value].transform('sum'), axis=0)

def pb(df, unit='shares'):

    if unit not in ('shares', 'count'):
        raise AttributeError(unit)

    value = 'diff_shares_num_lag_1'
    data = df.copy()

    if unit == 'shares':
        data[value] = data[value].mul(data[value].gt(0), axis=0)
    else:
        data[value] = data[value].gt(0).apply(int)

    data[value + '_new'] = data[value].mul(data.shares_num_lag_1.eq(0))

    return data.groupby(['id', 'period'])[value + '_new'].transform('sum') \
            .div(data.groupby(['id', 'period'])[value].transform('sum'), axis=0)

def ps(df, unit='shares'):

    if unit not in ('shares', 'count'):
        raise AttributeError(unit)

    value = 'diff_shares_num_lag_1'
    data = df.copy()

    if unit == 'shares':
        data[value] = data[value].mul(data[value].lt(0), axis=0)
    else:
        data[value] = data[value].lt(0).apply(int)

    data[value + '_new'] = data[value].mul(data.shares_num.eq(0))

    return data.groupby(['id', 'period'])[value + '_new'].transform('sum') \
            .div(data.groupby(['id', 'period'])[value].transform('sum'), axis=0)

№ function adding lags of column

def add_lag(df, cols, ref_cols, lag=1):

    data = df.copy()
    data.sort_values(by=ref_cols + ['period'], inplace=True)

    data['quarter_sum'] = data.period.dt.year * 4 + data.period.dt.quarter - 1

    value = ''

    for ref_col in ref_cols:
        if ref_col not in data:
            raise KeyError(ref_col)

        value += data[ref_col].apply(str) + '_'

    data['object'] = value

    for col in cols:
        if col not in data:
            raise KeyError(col)

        data[col + '_lag_' + str(lag)] = data[col].shift(periods=lag). \
            where((data.object == data.object.shift(periods=lag)) &
                  (data.quarter_sum.diff(periods=lag) == lag))

    return data.drop(['quarter_sum', 'object'], axis=1)

# function winsorizing column
        
def winsor(data, name, lim=0.5):

    df = data.copy()

    df[name] = df[name].clip(lower=np.nanpercentile(df[name], lim, interpolation='lower'),
                             upper=np.nanpercentile(df[name], 100 - lim, interpolation='lower'))

    return df

# reading and cleaning data

df = pd.read_csv(r'f13_data.csv')
df = clean_data(df).sort_values(by=['cik', 'id', 'period'])

# adding base variables

df['aum'] = df.groupby(by=['period', 'cik'])['market_value'].transform('sum')
df['share'] = df['market_value'] / df['aum']

pos_count = df.groupby(['period', 'cik']).apply(lambda x: x[x.shares_num != 0].id.nunique()).reset_index()
pos_count.rename(columns={0: 'pos_count'}, inplace=True)

df = df.merge(pos_count, on=['period', 'cik'], how='left', suffixes=(False, False))

# fixing prices

df['ym_stamp'] = pd.to_datetime(df.stamp).dt.to_period('M')
prices = df.groupby(['period', 'id', 'ym_stamp'])['close'].median().reset_index()
prices.rename(columns={'close': 'mode_close'}, inplace=True)

prices = prices.groupby(['period', 'id']).first().reset_index().drop('ym_stamp', axis=1)

df = df.merge(prices, on=['period', 'id'], suffixes=(False, False), how='left')

df['close'] = df.close.fillna(df.mode_close)
df = df.drop(['ym_stamp', 'mode_close'], axis=1)

# adding lags

for i in range(1, 5):
    df = add_lag(df, ['share', 'close'], ['cik', 'id'], i)
    
df_funds = df[['cik', 'period', 'aum', 'pos_count']].drop_duplicates()

for i in range(1, 5):
    df_funds = dataset.add_lag(df_funds, ['aum', 'pos_count'], ['cik'], i)

df_funds.drop(['aum', 'pos_count'], axis=1, inplace=True)
df = df.merge(df_funds, on=['period', 'cik'], suffixes=(False, False), how='left')

for value in ('aum', 'pos_count'):
    for lag in range(1, 5):

        suffix1 = '_lag' + '_' + str(lag - 1) if lag > 1 else ''
        suffix2 = '_lag' + '_' + str(lag)

        df['diff_' + value + suffix2] = df[value + suffix1].sub(df[value + suffix2])

        df['diff_' + value + '_rank' + suffix2] = df.groupby(['period'])['diff_' + value + suffix2] \
                                                            .transform('rank', pct=True, method='dense')
                                                            
for value in ('close', ):
    for lag in range(1, 5):

        suffix1 = '_lag' + '_' + str(lag - 1) if lag > 1 else ''
        suffix2 = '_lag' + '_' + str(lag)

        df['pct_' + value + suffix2] = df[value + suffix1].div(df[value + suffix2]).sub(1)

        df['pct_' + value + '_rank' + suffix2] = df.groupby(['period'])['pct_' + value + suffix2] \
                                                    .transform('rank', pct=True, method='dense')

for value in ('share', ):
    for lag in range(1, 5):

        suffix1 = '_lag' + '_' + str(lag - 1) if lag > 1 else ''
        suffix2 = '_lag' + '_' + str(lag)
        
        df['diff_' + value + suffix2] = df[value + suffix1].sub(df[value + suffix2])
        df['pct_' + value + suffix2] = df[value + suffix1].div(df[value + suffix2]).sub(1)

        df['diff_' + value + '_rank_along_fund' + suffix2] = df.groupby(['period', 'cik'])['diff_' + value + suffix2] \ 
                            .transform('rank', pct=True, method='dense')
        df['pct_' + value + '_rank_along_fund' + suffix2] = df.groupby(['period', 'cik'])['pct_' + value + suffix2] \
                            .transform('rank', pct=True, method='dense')
                            
        df['diff_' + value + '_rank' + suffix2] = df.groupby(['period'])['diff_' + value + suffix2] \
                                                    .transform('rank', pct=True, method='dense')
        df['pct_' + value + '_rank' + suffix2] = df.groupby(['period'])['pct_' + value + suffix2] \
                                                    .transform('rank', pct=True, method='dense')
                                                    
for value in ('aum', 'pos_count'):
    for lag in range(1, 5):

        suffix1 = '_lag' + '_' + str(lag - 1) if lag > 1 else ''
        suffix2 = '_lag' + '_' + str(lag)

        df['pct_' + value + suffix2] = df[value + suffix1].div(df[value + suffix2]).sub(1)

        df['pct_' + value + '_rank' + suffix2] = df.groupby(['period'])['pct_' + value + suffix2] \
                                                .transform('rank', pct=True, method='dense')

# adding return metrics
                                                
df['contrib_in_old_port_return'] = df['pct_close_lag_1'].mul(df['share_lag_1'], axis=0)
df['contrib_in_new_port_return'] = df['pct_close_lag_1'].mul(df['share'], axis=0)

df['old_port_return'] = df.groupby(['cik', 'period']).contrib_in_old_port_return.transform('sum')
df['new_port_return'] = df.groupby(['cik', 'period']).contrib_in_new_port_return.transform('sum')

df['inflow_return'] = df.pct_aum_lag_1 - df.new_port_return
df['sup_return'] = df.new_port_return - df.old_port_return

# adding macrovariables

import pandas_datareader as wb

rates = wb.DataReader(['DFF', 'UNRATE'], 'fred', start=2012).resample('Q-DEC').last().div(100).diff()
rates.reset_index(inplace=True)
rates.DATE = rates.DATE.dt.to_period('Q')

vix = wb.DataReader(['VIXCLS'], 'fred', start=2012).resample('Q-DEC').last().div(100).diff()
vix.reset_index(inplace=True)
vix.DATE = vix.DATE.dt.to_period('Q')

ted = wb.DataReader(['TEDRATE'], 'fred', start=2012).resample('Q-DEC').last().div(100).diff()
ted.reset_index(inplace=True)
ted.DATE = ted.DATE.dt.to_period('Q')

aaa = wb.DataReader(['DAAA'], 'fred', start=2012).resample('Q-DEC').last().div(100).diff()
aaa.reset_index(inplace=True)
aaa.DATE = aaa.DATE.dt.to_period('Q')

baa = wb.DataReader(['DBAA'], 'fred', start=2012).resample('Q-DEC').last().div(100).diff()
baa.reset_index(inplace=True)
baa.DATE = baa.DATE.dt.to_period('Q')

def_spread = pd.Series(data=(baa.DBAA - aaa.DAAA).values, index=baa.DATE, name='def_spread')

div_yield = wb.DataReader(['MULTPL/SP500_DIV_YIELD_MONTH'], 'quandl', start=2012, api_key='EZD_sUk9RPTNCHUVmqdi') \
            .resample('Q-DEC').last().div(100).diff()
div_yield = div_yield.unstack().reset_index().drop(['Attributes', 'Symbols'], axis=1)\
            .rename({0: 'div_yield', 'Date':'DATE'}, axis=1)
div_yield.DATE = div_yield.DATE.dt.to_period('Q')

t_bill = wb.DataReader(['DTB3'], 'fred', start=2012).resample('Q-DEC').last().div(100).diff()
t_bill.reset_index(inplace=True)
t_bill.DATE = t_bill.DATE.dt.to_period('Q')


tbill_ma = wb.DataReader(['DTB3'], 'fred', start=2012).resample('M').median().rolling(12).mean().resample('Q-DEC') \
            .last().div(100).diff()
tbill_ma.reset_index(inplace=True)
tbill_ma.DATE = tbill_ma.DATE.dt.to_period('Q')

rel_tbill = pd.Series(data=(t_bill.DTB3 - tbill_ma.DTB3).values, index=t_bill.DATE, name='rel_tbill')

yield_3mo = wb.DataReader(['DGS3MO'], 'fred', start=2012).resample('Q-DEC').last().div(100).diff()
yield_3mo.reset_index(inplace=True)
yield_3mo.DATE = yield_3mo.DATE.dt.to_period('Q')

yield_10y = wb.DataReader(['DGS10'], 'fred', start=2012).resample('Q-DEC').last().div(100).diff()
yield_10y.reset_index(inplace=True)
yield_10y.DATE = yield_10y.DATE.dt.to_period('Q')

term_spread = pd.Series(data=(yield_10y.DGS10 - yield_3mo.DGS3MO).values, index=yield_10y.DATE, name='term_spread')

inflation = wb.DataReader('CPIAUCSL', 'fred', start=2012).resample('Q-DEC').last().pct_change().diff()
inflation.reset_index(inplace=True)
inflation.DATE = inflation.DATE.dt.to_period('Q')

gdp_growth = wb.DataReader('GDP', 'fred', start=2012).pct_change()
gdp_growth.reset_index(inplace=True)
gdp_growth.DATE = gdp_growth.DATE.dt.to_period('Q')

sp500ret = wb.DataReader('^GSPC', 'yahoo', start=2012).resample('Q-DEC').last().pct_change()
sp500ret = sp500ret[['Adj Close']]
sp500ret = sp500ret.reset_index().rename({"Date": "DATE", 'Adj Close': 'S&P500_ret'}, axis=1)
sp500ret['S&P500_ret_diff'] = sp500ret['S&P500_ret'].diff()
sp500ret.DATE = sp500ret.DATE.dt.to_period('Q')

df = df.merge(rates, left_on='period', right_on='DATE', how='left', suffixes=(False, False))
df = df.merge(vix, on="DATE", how='left', suffixes=(False, False))
df = df.merge(ted, on="DATE", how='left', suffixes=(False, False))
df = df.merge(aaa, on="DATE", how='left', suffixes=(False, False))
df = df.merge(baa, on="DATE", how='left', suffixes=(False, False))
df = df.merge(def_spread, on="DATE", how='left', suffixes=(False, False))
df = df.merge(div_yield, on="DATE", how='left', suffixes=(False, False))
df = df.merge(t_bill, on="DATE", how='left', suffixes=(False, False))
df = df.merge(tbill_ma, on="DATE", how='left', suffixes=(False, '_ma'))
df = df.merge(rel_tbill, on="DATE", how='left', suffixes=(False, False))
df = df.merge(yield_3mo, on="DATE", how='left', suffixes=(False, False))
df = df.merge(yield_10y, on="DATE", how='left', suffixes=(False, False))
df = df.merge(term_spread, on="DATE", how='left', suffixes=(False, False))
df = df.merge(inflation, on="DATE", how='left', suffixes=(False, False))
df = df.merge(gdp_growth, on="DATE", how='left', suffixes=(False, False))
df = df.merge(sp500ret, on="DATE", how='left', suffixes=(False, False))
df.drop('DATE', axis=1, inplace=True)

df['excess_market_ret'] = df['S&P500_ret'] - df.DGS3MO
df['abnormal_fund_ret'] = df.new_port_return - df['S&P500_ret']

# adding turnover and hhi with lags

df['turnover_qtr'] = df.groupby(['cik', 'period']).diff_share_lag_1.transform(lambda x: x.abs().sum())
df['turnover_hy'] = df.groupby(['cik', 'period']).diff_share_lag_2.transform(lambda x: x.abs().sum())
df['turnover_y'] = df.groupby(['cik', 'period']).diff_share_lag_4.transform(lambda x: x.abs().sum())

df['fund_hhi'] = df.groupby(['cik', 'period']).share.transform(lambda x: x.pow(2).sum())

for i in range(1, 5):
    df = add_lag(df, ['turnover_qtr', 'turnover_hy', 'turnover_y', 'fund_hhi'], ['cik', 'id'], i)
    
for value in ('turnover_qtr', 'turnover_hy', 'turnover_y', 'fund_hhi'):
    for lag in range(1, 5):

        suffix1 = '_lag' + '_' + str(lag - 1) if lag > 1 else ''
        suffix2 = '_lag' + '_' + str(lag)

        df['diff_' + value + suffix2] = df[value + suffix1].sub(df[value + suffix2])

        df['diff_' + value + '_rank' + suffix2] = df.groupby(['period'])['diff_' + value + suffix2] \
                                            .transform('rank', pct=True, method='dense')

# counting shares and tickers in fund
                                            
df['shares_in_fund'] = df.groupby(['cik', 'period'])['shares_num'].transform('sum')
df['share_in_shares'] = df.shares_num.div(df.shares_in_fund, axis=0)

df['top_5_tickers_share_count'] = df.groupby(['cik', 'period'])['share_in_shares'] \
                            .transform(lambda grp: grp.nlargest(5).sum())
df['top_10_tickers_share_count'] = df.groupby(['cik', 'period'])['share_in_shares'] \
                            .transform(lambda grp: grp.nlargest(10).sum())
df['top_50_tickers_share_count'] = df.groupby(['cik', 'period'])['share_in_shares'] \
                            .transform(lambda grp: grp.nlargest(50).sum())
df['top_100_tickers_share_count'] = df.groupby(['cik', 'period'])['share_in_shares'] \
                            .transform(lambda grp: grp.nlargest(100).sum())
df['top_500_tickers_share_count'] = df.groupby(['cik', 'period'])['share_in_shares'] \
                            .transform(lambda grp: grp.nlargest(500).sum())
df['top_1000_tickers_share_count'] = df.groupby(['cik', 'period'])['share_in_shares'] \
                            .transform(lambda grp: grp.nlargest(1000).sum())

df['top_5_tickers_share_value'] = df.groupby(['cik', 'period'])['share'] \
                            .transform(lambda grp: grp.nlargest(5).sum())
df['top_10_tickers_share_value'] = df.groupby(['cik', 'period'])['share'] \
                            .transform(lambda grp: grp.nlargest(10).sum())
df['top_50_tickers_share_value'] = df.groupby(['cik', 'period'])['share'] \
                            .transform(lambda grp: grp.nlargest(50).sum())
df['top_100_tickers_share_value'] = df.groupby(['cik', 'period'])['share'] \
                            .transform(lambda grp: grp.nlargest(100).sum())
df['top_500_tickers_share_value'] = df.groupby(['cik', 'period'])['share'] \
                            .transform(lambda grp: grp.nlargest(500).sum())
df['top_1000_tickers_share_value'] = df.groupby(['cik', 'period'])['share'] \
                            .transform(lambda grp: grp.nlargest(1000).sum())
                         
# adding active_share

df = add_bench_dev(df)
df = add_active_share(df)

# adding other fund-based variables and active share and number of shares lags

df['fund_share_in_ticker'] = df.market_value.div(df.market_value_ticker, axis=0)
df['fund_share_among_funds'] = df.aum.div(df.tickers_total_value, axis=0)

df['liquidity_flow_shares'] = lf(df)
df['liquidity_flow_count'] = lf(df, 'count')

df_funds = df[['cik', 'period', 'active_share']].drop_duplicates()

for i in range(1, 5):
    df_funds = df_funds, ['active_share'], ['cik'], i)

df_funds.drop(['active_share'], axis=1, inplace=True)
df = df.merge(df_funds, on=['period', 'cik'], suffixes=(False, False), how='left')

for value in ('active_share', ):
    for lag in range(1, 5):

        suffix1 = '_lag' + '_' + str(lag - 1) if lag > 1 else ''
        suffix2 = '_lag' + '_' + str(lag)

        df['diff_' + value + suffix2] = (df[value + suffix1] - df[value + suffix2])

        df['diff_' + value + '_rank' + suffix2] = df.groupby(['period'])['diff_' + value + suffix2] \
                                                .transform('rank', pct=True, method='dense')
                                                
for i in range(1, 5):
    df = add_lag(df, ['shares_num'], ['cik', 'id'], i)
    
for value in ('shares_num', ):
    for lag in range(1, 5):

        suffix1 = '_lag' + '_' + str(lag - 1) if lag > 1 else ''
        suffix2 = '_lag' + '_' + str(lag)
        
        df['diff_' + value + suffix2] = (df[value + suffix1] - df[value + suffix2])
        df['pct_' + value + suffix2] = (df[value + suffix1] / df[value + suffix2]) - 1

        df['diff_' + value + '_rank_along_fund' + suffix2] = df.groupby(['period', 'cik'])['diff_' + value + suffix2] \
                                            .transform('rank', pct=True, method='dense')
        df['pct_' + value + '_rank_along_fund' + suffix2] = df.groupby(['period', 'cik'])['pct_' + value + suffix2] \
                              .transform('rank', pct=True, method='dense')

# adding ticker-based variables

df['bosi_shares'] = bosi(df)
df['bosi_count'] = bosi(df, 'count')

df['busi_shares'] = busi(df)
df['busi_count'] = busi(df, 'count')

df['pb_shares'] = pb(df)
df['pb_count'] = pb(df, 'count')

df['ps_shares'] = ps(df)
df['ps_count'] = ps(df, 'count')

df['top_5_funds_share_value'] = df.groupby(['id', 'period'])['fund_share_in_ticker'] \
                            .transform(lambda grp: grp.nlargest(5).sum())
df['top_10_funds_share_value'] = df.groupby(['id', 'period'])['fund_share_in_ticker'] \
                            .transform(lambda grp: grp.nlargest(10).sum())
df['top_50_funds_share_value'] = df.groupby(['id', 'period'])['fund_share_in_ticker'] \
                            .transform(lambda grp: grp.nlargest(50).sum())
df['top_100_funds_share_value'] = df.groupby(['id', 'period'])['fund_share_in_ticker'] \
                            .transform(lambda grp: grp.nlargest(100).sum())
                            
df['ticker_hhi'] = df.groupby(['id', 'period']).fund_share_in_ticker.transform(lambda x: x.pow(2).sum())

df['funds_in_ticker'] = df.groupby(['id', 'period']).shares_num.transform(lambda x: x.gt(0).sum())

# adding market value lags

for i in range(1, 5):
    df = add_lag(df, ['market_value'], ['cik', 'id'], i)

# adding dependent variable and its lags
    
df['change'] = df.diff_shares_num_lag_1.mul(df.close_lag_1, axis=0).div(df.aum_lag_1, axis=0).replace([-np.inf, np.inf], np.nan)
df['change_predict'] = add_lag(df, ['change'], ['cik', 'id'], -1)['change_lag_-1']

for i in range(1, 5):
    df = add_lag(df, ['change'], ['cik', 'id'], i)
    
for value in ('change',):
    for lag in range(1, 5):

        suffix1 = '_lag' + '_' + str(lag - 1) if lag > 1 else ''
        suffix2 = '_lag' + '_' + str(lag)

        df['diff_' + value + suffix2] = (df[value + suffix1] - df[value + suffix2])

        df['diff_' + value + '_rank' + suffix2] = df.groupby(['period'])['diff_' + value + suffix2] \
                   .transform('rank', pct=True, method='dense')
                   
# dropping inapt observations

df = df[df.shares_num.gt(0) & ~df.change_predict.isna()]
df = df[~df.share.eq(-df.change_predict)]

# splitting data

train_index = df[df.period.lt(df.period.max())].index
val_index = df[df.period.eq(df.period.max())].index

X_train = df.loc[train_index].drop(['change_predict'], axis=1)
y_train = df.change_predict.loc[train_index]

X_val = df.loc[val_index].drop(['change_predict'], axis=1)
y_val = df.change_predict.loc[val_index]

# plots of dependent variable distributions

plt.figure(figsize=(10,6));
plt.xlabel('сhange_predict Box Plot')
sns.boxplot(y=df.change_predict);

plt.figure(figsize=(10,6));
plt.xlabel('сhange_predict Box Plot Train Data')
sns.boxplot(y=train_y);

plt.figure(figsize=(10,6));
plt.xlabel('сhange_predict Box Plot Validation set')
sns.boxplot(y=val_y);

# externalizing setups

X_train_norm = X_train.copy(deep=True)
for col in X_train_norm.columns:
  if col not in macro_columns:
    X_train_norm[col] = (X_train[col] - X_train[col].mean()) \
    / X_train[col].std()

X_train_winsor = X_train.copy(deep=True)
for col in X_train_winsor.columns:
  if col not in macro_columns:
    X_train_winsor[col] = winsor(X_train_winsor[col])

X_train_norm_winsor = X_train_norm.copy(deep=True)
for col in X_train_norm_winsor.columns:
  if col not in macro_columns:
    X_train_norm_winsor[col] = winsor(X_train_norm_winsor[col])
    
X_val_norm = X_val.copy(deep=True)
for col in X_val_norm.columns:
  if col not in macro_columns:
    X_val_norm[col] = (X_val_norm[col] - X_train[col].mean())\
    / X_train[col].std()

X_val_winsor = X_val.copy(deep=True)
for col in X_val_winsor.columns:
  if col not in macro_columns:
    X_val_winsor[col] = winsor(X_val_winsor[col])

X_val_norm_winsor = X_val_norm.copy(deep=True)
for col in X_train_norm_winsor.columns:
  if col not in macro_columns:
    X_val_norm_winsor[col] = winsor(X_val_norm_winsor[col])

data = {True: {True: [X_train_norm_winsor.replace([-np.inf, np.inf], 
                                              np.nan).fillna(0), X_val_norm_winsor.loc[val_index]],
               False: [X_train_norm.replace([-np.inf, np.inf], 
                                              np.nan).fillna(0), X_val_norm.loc[val_index]]}}#,
        False: {True: [X_train_winsor.replace([-np.inf, np.inf], 
                                              np.nan).fillna(0), X_val_winsor.loc[val_index]],
                False: [X_train.replace([-np.inf, np.inf], 
                                              np.nan).fillna(0), X_val.loc[val_index]]}}

# sampling train and valid data
    
val_index = X_val.sample(frac=0.2, random_state=0).index
y_val = y_val.loc[val_index]
                                             

perc_75_train = np.nanpercentile(y_train, 75, interpolation='lower')
perc_25_train = np.nanpercentile(y_train, 25, interpolation='lower')

train_index = y_train[y_train.ge(perc_75_train) | y_train.le(perc_25_train)] \
              .sample(frac=0.1, random_state=10).index.tolist() + \
            y_train[y_train.lt(perc_75_train) | y_train.gt(perc_25_train)] \
              .sample(frac=0.1, random_state=10).index.tolist()

# pipeline of training and evaluating

models = ['linreg', 'forest', 'boost']

experiments_results = []
for norm, winsor in product([True, False], repeat=2):
  for model in models:

    experiment_dict = {'norm': norm, 'winsor': winsor, 'model': model}
    if model == 'boost':
      model = CatBoostRegressor(depth=10, verbose=False, random_seed=10)
    elif model == 'forest':
      model = RandomForestRegressor(n_estimators=10, random_state=10)
    else:
      model = LinearRegression()

    X = data[norm][winsor][0].loc[train_index].values
    
    model.fit(X, y_train.loc[train_index].values)

    _X = data[norm][winsor][1].replace([-np.inf, np.inf], 
                                        np.nan).fillna(0).values
    y_pred = model.predict(_X)

    total_mse = mse(y_val, y_pred, squared=False)
    
    experiment_dict['mse'] = total_mse
    print(experiment_dict)
    experiments_results.append(experiment_dict)

# obtaining results
    
print(results = pd.DataFrame(experiments_results).pivot(index=['norm', 'winsor'], columns=['model'], values='mse').to_latex())

# plots of dependent variables sample distribution

plt.figure(figsize=(10,6));
plt.xlabel('сhange_predict Box Plot Train Sample')
sns.boxplot(y=y_train[train_index]);

plt.figure(figsize=(10,6));
plt.xlabel('сhange_predict Box Plot Validation Sample')
sns.boxplot(y=y_val);