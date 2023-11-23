import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import date2num
from itertools import cycle
markers = ['o', '.', '^', 'v', '<', '>', '8', 's', 'p', '*', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def match_recommendations_prices(fn_recommendations="analysis.csv", fn_prices="prices.csv", verbose=True):
    # Read recommendations
    recommendations = pd.read_csv(fn_recommendations, parse_dates=[2])
    # add recommendation date (after close -> next day)
    recommendations['PvmSuositus'] = pd.to_datetime(np.where(recommendations['created'].dt.time > datetime.time(18,25),
                                              recommendations['created'].dt.date + pd.Timedelta(1,'D'),
                                              recommendations['created'].dt.date))
    # Capitalize columns and recommendations
    recs = {'lisää':'Lisää', 'vähennä':'Vähennä','osta':'Osta', 'myy':'Myy', 'pidä':'-', '-':'-'}
    recommendations.rename(columns={'suositus':'Suositus', 'analyytikko':'Analyytikko'}, inplace=True)
    recommendations['Suositus'] = recommendations['Suositus'].str.lower().map(recs)

    # Read Inderes <-> Kauppalehti links
    inderes_to_ticker = pd.read_csv('key_merge.csv')
    kauppalehti_tickers = pd.read_csv('key_kauppalehti.csv')
    # has price information but missing from the merge file
    no_merge = set(kauppalehti_tickers['ticker']).difference(set(inderes_to_ticker['ticker']))
    if verbose and len(no_merge) > 0: print("Kauppalehti tickers missing from merge file: %s" % no_merge)
    # has ticker but missing price history information
    no_price = set(inderes_to_ticker['ticker'].dropna()).difference(set(kauppalehti_tickers['ticker']))
    if verbose and len(no_price) > 0: print("Kauppalehti tickers missing price information: %s" % no_price)
    # Which inderes identifiers are missing from the merge file?
    no_merge = set(recommendations['stock']).difference(set(inderes_to_ticker['stock']))
    if verbose and len(no_merge) > 0: print("Inderes stock missing from the merge file: %s" % no_merge)
    # These inderes identifiers have no matching kauppalehti ticker
    no_ticker = set(inderes_to_ticker.loc[inderes_to_ticker['stock'].isin(set(recommendations['stock'])) & 
                                          inderes_to_ticker['ticker'].isnull(), 'stock'])
    if verbose and len(no_ticker) > 0: print("Inderes stock missing kauppalehti ticker: %s" % no_ticker)

    # merge inderes 'stock' to kauppalehti 'name' by 'ticker' column in both merge files
    stock_name = inderes_to_ticker.merge(kauppalehti_tickers).set_index('stock')['name']
    recommendations['Osake'] = recommendations['stock'].map(dict(stock_name))
    recommendations['url'] = recommendations['url'].apply(lambda url: 'https://www.inderes.fi/fi/{}'.format(url))
    recommendations = recommendations.loc[recommendations['Osake'].notnull(),
                                          ['Osake', 'PvmSuositus', 'Suositus', 'Analyytikko', 'url']]

    # Read prices
    prices = pd.read_csv(fn_prices, parse_dates=[1])

    # Find the next trading day (Päivämäärä) following every recommendation day (PvmSuositus)
    new_recommendations = recommendations.merge(prices[['Osake', 'Päivämäärä']], how='outer', 
                                    left_on=['Osake', 'PvmSuositus'],
                                    right_on=['Osake', 'Päivämäärä'], 
                                    sort=True)
    # Fill the recommendation forward to all subsequent trading days, return first trading day
    fill_columns = recommendations.columns.drop('Osake')
    new_recommendations[fill_columns] = new_recommendations.groupby('Osake')[fill_columns].ffill()#.fillna(method='ffill')
    new_recommendations.dropna(axis=0, subset=['Päivämäärä'], inplace=True) 
    new_recommendations = new_recommendations.groupby(['Osake', 'PvmSuositus'], as_index=False).first()
    # Issue a warning for recommendations for which trading day was not found
    temp = recommendations.merge(new_recommendations, how='left')
    temp = temp[temp['Päivämäärä'].isnull()]
    if verbose and len(temp) > 0:
        print('Recommendations without matching closing price:')
        print([(stock, pvm.strftime('%Y-%m-%d')) for i, (stock, pvm) in temp[["Osake", "PvmSuositus"]].iterrows()])
        
    # Add 'end of follow-up' at last trading day in data
    last_close = prices.groupby('Osake',as_index=False)['Päivämäärä'].max()#.last()
    last_close['PvmSuositus'] = last_close['Päivämäärä'] 
    last_close['Suositus'] = '-'
    last_close['Analyytikko'] = '-'
    last_close = last_close.loc[last_close['Osake'].isin(recommendations['Osake'].unique())]
    new_recommendations = pd.concat([new_recommendations, last_close], axis=0)
    new_recommendations.sort_values(['Osake', 'PvmSuositus'], inplace=True)
    return(new_recommendations)

def get_prices(fn_prices="prices.csv"):
    prices = pd.read_csv(fn_prices, parse_dates=[1])
    
    # add profit each day for every stock
    previous_price = prices.groupby('Osake')['Päätöskurssi + Osinko'].shift(1)
    prices['Tuotto (%)'] = prices['Päätöskurssi + Osinko'] / previous_price - 1
    prices['Alin (%)'] = prices['Alin'] / prices['Päätöskurssi'] - 1
    prices['Ylin (%)'] = prices['Ylin'] /prices['Päätöskurssi'] - 1
    
    # calculate index return and append to prices
    index = get_index_return(prices)
    prices = prices.merge(index, how='left')
    return(prices)

def create_categories(values, n_categories=4):
    values_interval = pd.qcut(values.round(0), n_categories, precision=0)
    values_interval = values_interval.apply(lambda x: pd.Interval(left=int(x.left), right=int(x.right)))
    return(values_interval)

def add_information(recommendations, prices, extras=True, categories=True):
    
    # Previous recommendation -> Current recommendation category
    recommendations['Edellinen'] =  recommendations.groupby('Osake')['Suositus'].shift(1).fillna('-')
    #recommendations['Muutos'] = recommendations['Edellinen'] + "->" + recommendations['Suositus']
    
    if extras:
        # Calculate mean trading volume in total and at recommendation date (use median?)
        volume_stock = prices.groupby('Osake')['Vaihto €'].agg('mean')
        if categories: volume_stock = create_categories(volume_stock)
        volume_365d = prices.groupby('Osake').apply(lambda s: s.set_index('Päivämäärä')['Vaihto €'].rolling('356D').mean())
        if categories: volume_365d = create_categories(volume_365d)
        # Add to recommendations
        recommendations['Vaihto'] = recommendations['Osake'].map(volume_stock)
        recommendations['Vaihto 365d'] = recommendations[['Osake', 'Päivämäärä']].merge(volume_365d.reset_index(),
                                                                                        how = 'left')['Vaihto €']

        # Count how many recommendations analyst has given in total and at recommendation date
        experience_total = recommendations["Analyytikko"].value_counts(dropna=False)
        if categories: experience_total = create_categories(experience_total)
        experience_date = recommendations.groupby(["Analyytikko", "Päivämäärä"])["Osake"].count().groupby(level=0).cumsum()
        if categories: experience_date = create_categories(experience_date)

        # Add to recommendations
        recommendations['Kokemus'] = recommendations['Analyytikko'].map(experience_total)
        recommendations['Kokemus 365d'] = recommendations[['Analyytikko', 'Päivämäärä']].merge(experience_date.reset_index(),
                                                                                               how = 'left')['Osake']
        recommendations['Vuosi'] = recommendations['Päivämäärä'].dt.year

    return(recommendations)

def get_index_return(prices):
    # Some trading days are missing for a given stock. Maybe the trading was halted, no trade, or the data has problem
    trading_days = pd.Index(pd.to_datetime(np.sort(prices['Päivämäärä'].unique())), name = 'Päivämäärä')
    # Fill these as 'the stock did not change from previous day', this makes it easy to calculate portfolio return (unbiased)
    def fill_price_between(df_stock):
        df_stock = df_stock.set_index('Päivämäärä').drop(columns='Osake')
        trading_first, trading_last = df_stock.index[0], df_stock.index[-1]
        df_stock = df_stock.reindex(trading_days)[trading_first:trading_last].ffill()#.fillna(method='ffill')
        return(df_stock)

    # version of the data with prices filled between first and last trading day
    prices_filled = prices.groupby('Osake').apply(fill_price_between).reset_index()
    previous_price = prices_filled.groupby('Osake')['Päätöskurssi + Osinko'].shift(1)
    prices_filled['Tuotto (%)'] = prices_filled['Päätöskurssi + Osinko'] / previous_price - 1

    # Percentage change of equal weight index
    index = prices_filled.groupby('Päivämäärä')["Tuotto (%)"].mean().reset_index()
    index.rename(columns={'Tuotto (%)': 'Index (%)'}, inplace=True)
    index['Index'] = (1 + index['Index (%)']).cumprod().fillna(1.0)
    return(index)

def to_counting_process(recommendations, prices, id_vars='Osake', buy='Buy', sell='Sell'):
    # Change in stock observable 
    dY = pd.melt(recommendations, id_vars=id_vars, value_vars=[buy, sell], var_name='dY', value_name='Päivämäärä')
    dY['dY'] = dY['dY'].map({buy: 1, sell: -1})
    # Combine with price information
    returns = recommendations.merge(dY, how='left')
    returns = prices.merge(returns, on=['Osake', 'Päivämäärä'], how='left')
    returns['dY'] = returns['dY'].fillna(0.0)
    fill_columns = returns.columns.drop(['Osake'])
    returns[fill_columns] = returns.groupby('Osake')[fill_columns].ffill()#.fillna(method='ffill')
    # Add time since recommendation
    returns['Päivä'] = (returns['Päivämäärä'] - returns['Buy']).dt.days
    returns.reset_index(drop=True, inplace=True)
    return(returns)

range_like = lambda s: pd.Series(np.arange(len(s))+1, index=s.index)
rec_change = lambda rec: (rec != rec.shift(1)).cumsum()

# Add buy/sell dates for i'th recommendation
def recommendation_ith(recommendations):
    recommendations = recommendations.copy()
    recommendations['N'] = recommendations.groupby('Osake')['Suositus'].apply(range_like).droplevel(0)
    recommendations['Buy'] = recommendations['Päivämäärä']
    recommendations['Sell'] = recommendations.groupby('Osake')['Päivämäärä'].shift(-1)
    recommendations.dropna(axis=0, subset=['Sell'], inplace=True) 
    recommendations.drop(columns='Päivämäärä', inplace=True)
    return(recommendations)

# Add buy/sell dates for N'th recommendation
def recommendation_nth(recommendations):
    # Add buy/sell date
    recommendations = recommendations.copy()
    recommendations['N'] = recommendations.groupby('Osake')['Suositus'].apply(rec_change).droplevel(0)
    recommendations = recommendations.groupby(['Osake', 'N'], as_index=False).first()
    recommendations['Buy'] = recommendations['Päivämäärä']
    recommendations['Sell'] = recommendations.groupby('Osake')['Päivämäärä'].shift(-1)
    recommendations.dropna(axis=0, subset=['Sell'], inplace=True) 
    recommendations.drop(columns='Päivämäärä', inplace=True)
    return(recommendations)

# Trading days may not neatly match 30d, 60d, 90d, ... so fill between and calculate stock returns
def get_stock_returns(df_stock, days=(30,90,180,365,365*2,365*3), close='Päätöskurssi + Osinko'):
    df_stock = df_stock.set_index('Päivämäärä')[[close]]
    trading_days = df_stock.index
    df_stock = df_stock.reindex(pd.date_range(trading_days[0],  trading_days[-1])).ffill()#.fillna(method='ffill')
    for day in days:
        df_stock['Tuotto Inderes {}d (%)'.format(day)] = df_stock[close].shift(-day)/df_stock[close] - 1
    df_stock = df_stock.reindex(trading_days)
    df_stock.drop(columns=close, inplace=True)
    return(df_stock)

# Calculate 30d, 60d, 90d, ... index returns
def get_index_returns(df_index, days=(30,90,180,365,365*2,365*3), close='Index'):
    df_index = df_index.set_index('Päivämäärä')[[close]]
    all_days = pd.date_range(df_index.index.min(), df_index.index.max(), name='Päivämäärä')
    df_index = df_index.reindex(all_days).ffill()
    for day in days:
        df_index['Tuotto Index {}d (%)'.format(day)] = df_index[close].shift(-day) / df_index[close] - 1
    df_index.reset_index(inplace=True)
    df_index.drop(columns=close, inplace=True)
    return(df_index)

# Add recommendation interval (old recommendation -> new recommendation) returns
def add_interval_returns(recommendations, prices):
    # At every recommendation, add first day stock and index return, stock and index buy&sell prices
    inderes_effect = prices[['Osake', 'Päivämäärä', 'Tuotto (%)', 'Index (%)']].rename(
        columns={'Päivämäärä':'Buy', 'Tuotto (%)': 'Inderes (%)'})
    buy = prices[['Osake', 'Päivämäärä', 'Päätöskurssi + Osinko', 'Index']].rename(
        columns={'Päivämäärä':'Buy', 'Päätöskurssi + Osinko': 'Inderes (Buy)', 'Index': 'Index (Buy)'})
    sell = prices[['Osake', 'Päivämäärä', 'Päätöskurssi + Osinko', 'Index']].rename(
        columns={'Päivämäärä':'Sell', 'Päätöskurssi + Osinko': 'Inderes (Sell)', 'Index': 'Index (Sell)'})
    intervals = recommendations.merge(inderes_effect, how='left').merge(buy, how='left').merge(sell, how='left')

    # add stock and index return from buy&sell prices
    intervals['Tuotto Inderes (%)'] = intervals['Inderes (Sell)'] / intervals['Inderes (Buy)'] - 1
    intervals['Tuotto Index (%)'] = intervals['Index (Sell)'] / intervals['Index (Buy)'] - 1
    intervals['Time'] = (intervals['Sell'] - intervals['Buy']).dt.days
    return(intervals)

# Add recommendation X day returns
def add_Xday_returns(recommendations, prices):
    # Calculate stock and index return X days from recommendation date
    returns_stock = prices.groupby('Osake').apply(get_stock_returns).reset_index().rename(columns={"Päivämäärä":"Buy"})
    returns_index = get_index_returns(get_index_return(prices)).rename(columns={"Päivämäärä":"Buy"})
    # Combine stock and index returns
    intervals = recommendations.merge(returns_stock, how='left').merge(returns_index, how='left')
    return(intervals)

from scipy.optimize import newton, bisect

def compute_profit(cashflow, time):
    dcf = lambda d: (cashflow / (1.0 + d)**time).mean() - 1
    try:
        z = newton(dcf, 0.00, maxiter=1000)
    except (ValueError, RuntimeError) as e:
        z = np.nan
    return(z)

def bootstrap_mean(x):
    x = x[~np.isnan(x)]
    bootstrap_means = [np.random.choice(x, len(x)).mean() for i in range(100)] 
    mean = np.mean(x)
    median = np.median(x)
    lci = np.quantile(x, 0.025)
    uci = np.quantile(x, 0.975)
    mean_lci = np.quantile(bootstrap_means, 0.025)
    mean_uci = np.quantile(bootstrap_means, 0.975)
    return(pd.Series({'mean': mean, 'median': median, '95% lower': lci, '95% upper':uci, 
                      'mean 95% lower': mean_lci, 'mean 95% upper': mean_uci}))


def mcf_align(df_mcf1, df_mcf2):
    # The event times may differ, reindex both to all event times
    df_mcf1 = df_mcf1.set_index('Time')
    df_mcf2 = df_mcf2.set_index('Time')
    idx = df_mcf1.index.union(df_mcf2.index)
    # Fill forward overlapping MCFs
    # MCF1
    df_mcf1 = df_mcf1.reindex(idx)
    df_mcf1['dY'].fillna(value=0, inplace=True)
    df_mcf1['dC'].fillna(value=0, inplace=True)
    df_mcf1['Y'].fillna(method='ffill', inplace=True)
    df_mcf1['Y'].fillna(0, inplace=True)
    df_mcf1['C'].fillna(method='ffill', inplace=True)
    df_mcf1.loc[df_mcf1['Y'] == 0, 'C'] = np.nan #end of MCF
    df_mcf1.reset_index(inplace=True)
    # MCF 2
    df_mcf2 = df_mcf2.reindex(idx)
    df_mcf2['dY'].fillna(value=0, inplace=True)
    df_mcf2['dC'].fillna(value=0, inplace=True)
    df_mcf2['Y'].fillna(method='ffill', inplace=True)
    df_mcf2['Y'].fillna(0, inplace=True)
    df_mcf2['C'].fillna(method='ffill', inplace=True)
    df_mcf2.loc[df_mcf2['Y'] == 0, 'C'] = np.nan #end of MCF
    df_mcf2.reset_index(inplace=True)
    return(df_mcf1, df_mcf2)

def mcf_diff(fr_sample1, fr_sample2, 
             sample1='Sample', time1='Time', events1='dY', values1='dC', 
             sample2='Sample', time2='Time', events2='dY', values2='dC', 
             include_lowest=False, bootstrap=None, save_bootstrap=False, level=0.05):
    
    df_mcf1 = mcf(fr_sample1, time=time1, sample=sample1, events=events1, values=values1, include_lowest=include_lowest)
    df_mcf2 = mcf(fr_sample2, time=time2, sample=sample2, events=events2, values=values2, include_lowest=include_lowest)
    df_mcf1, df_mcf2 = mcf_align(df_mcf1, df_mcf2)

    # Calculate difference in returns and total observable
    time = df_mcf1['Time'] # should be equal
    dC_diff = df_mcf1['dC'] - df_mcf2['dC']
    C_diff = df_mcf1['C'] - df_mcf2['C']
    dY = df_mcf1['dY'] + df_mcf2['dY'] 
    Y = np.minimum(df_mcf1['Y'], df_mcf2['Y'])
    df_diff = pd.DataFrame({'Time': time, 'dY':dY, 'Y': Y, 'dC':dC_diff, 'C': C_diff})
    
    # Calculate MCF diffs from bootstrap replicates
    if not bootstrap is None:
        mcfs = {}
        instances1 = fr_sample1[sample1].drop_duplicates()
        instances2 = fr_sample2[sample2].drop_duplicates()
        for i in range(bootstrap):
            # Take an equal size samples from originals with replacement
            fr_bootstrap1 = instances1.sample(frac=1, replace=True)
            fr_bootstrap1['BootstrapSample'] = np.arange(len(fr_bootstrap1))
            fr_bootstrap1 = fr_bootstrap1.merge(fr_sample1)
            fr_bootstrap2 = instances2.sample(frac=1, replace=True)
            fr_bootstrap2['BootstrapSample'] = np.arange(len(fr_bootstrap2))
            fr_bootstrap2 = fr_bootstrap2.merge(fr_sample2)            
            df_bootstrap = mcf_diff(fr_bootstrap1, fr_bootstrap2, include_lowest=include_lowest,
                                    time1=time1, sample1='BootstrapSample', events1=events1, values1=values1,
                                    time2=time2, sample2='BootstrapSample', events2=events2, values2=values2)
            _, df_bootstrap = mcf_align(df_diff, df_bootstrap)
            mcfs['C.{}'.format(i)] = df_bootstrap.set_index('Time')['C']
        mcfs = pd.DataFrame(mcfs)
        # Calculate quantiles
        df_diff['C Lower'] = mcfs.apply(lambda means: np.quantile(means, level/2), axis=1) 
        df_diff['C Upper'] = mcfs.apply(lambda means: np.quantile(means, 1 - level/2), axis=1)
        if save_bootstrap:
            df_diff = pd.concat((df_diff, mcfs), axis=1)
        
    return(df_diff)

def mcf(fr_sample, sample='Sample', time='Time', events='dY', values='dC', extras=[],
         include_lowest=False, bootstrap=None, save_bootstrap=False, level=0.05, add_cost=0.00):
    df_sample = pd.DataFrame(fr_sample[sample])
    # Calculate if the sample is observable at each time points
    df_sample['Time'] = fr_sample[time]
    df_sample['dY'] = fr_sample[events]
    df_sample['Yi'] = df_sample.groupby(sample)['dY'].cumsum()
    df_sample['Yi'] = df_sample.groupby(sample)['Yi'].shift(1).fillna(0)
    # Whether to include the return at entry: left interval open or closed
    df_sample['dY0i'] = (fr_sample[values].notnull() & (fr_sample[events] == 1)).astype(int)
    if include_lowest:
        df_sample['Yi'] = df_sample['Yi'] + df_sample['dY0i']
    # Caclulate the return of the sample at each time point
    df_sample['dC'] = df_sample['Yi'] * fr_sample[values]
    for col in extras:
        df_sample[col] = df_sample['Yi'] * fr_sample[col]
    
    if add_cost > 0.00:
        df_sample.loc[fr_sample[events] > 0, 'dC'] = 0.0
        df_sample.loc[fr_sample[events] != 0, 'dC'] -= add_cost
    
    # Calculate the total return and the total number of observable at each time point
    df = df_sample.groupby('Time')[['dC', 'dY', 'dY0i'] + extras].agg('sum').sort_index().reset_index()
    df['Y'] = df['dY'].cumsum().shift(1).fillna(0)
    if include_lowest:
        df['Y'] = df['Y'] + df['dY0i']
    # MCF: Calculate the mean return and cumulative return at each time point 
    df['dC'] = np.where(df['Y'] > 0, df['dC']/df['Y'], 0.0)
    df['C'] = (1+df['dC']).cumprod()
    for col in extras:
        df[col] = np.where(df['Y'] > 0, df[col]/df['Y'], 0.0)
    #df.drop('dY0i', axis=1, inplace=True)
    df = df[['Time', 'dY', 'dC', 'Y', 'C'] + extras]
    
    # Recalculate MCFs from bootstrap replicates
    if not bootstrap is None:
        mcfs = {}
        instances = fr_sample[sample].drop_duplicates()
        for i in range(bootstrap):
            # Take an equal size sample from original with replacement
            fr_bootstrap = instances.sample(frac=1, replace=True)
            fr_bootstrap['BootstrapSample'] = np.arange(len(fr_bootstrap))
            fr_bootstrap = fr_bootstrap.merge(fr_sample)
            # Calculate and save MCF from this sample
            df_bootstrap = mcf(fr_bootstrap, sample='BootstrapSample', time=time, events=events, values=values,
                               include_lowest=include_lowest, bootstrap=None, extras=extras)
            _, df_bootstrap = mcf_align(df, df_bootstrap)
            mcfs['C.{}'.format(i)] = df_bootstrap.set_index('Time')['C']
        mcfs = pd.DataFrame(mcfs)
        # Calculate quantiles
        df['C Lower'] = mcfs.apply(lambda means: np.quantile(means, level/2), axis=1) 
        df['C Upper'] = mcfs.apply(lambda means: np.quantile(means, 1 - level/2), axis=1) 
        if save_bootstrap:
            df = pd.concat((df, mcfs), axis=1)
    
    #extra_cols = [] if bootstrap is None else ['C Lower', 'C Upper']
    return(df)

def plot_mcf(fr, axs=None, label='', color='black', marker='.', drawstyle='steps-post',
              CI=True, xmin=None, xmax=None, ymin=None, ymax=None):

    # If no axis were given, create a new figure
    if axs is None:
        fig = plt.figure(figsize=(8,6))
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        ax = fig.add_subplot(gs[0])
        axx = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax, axx = axs

    # Plot MCF
    has_values, has_censor = fr['dC'] > 0, fr['dY'] < 0
    ax.plot(fr.loc[has_values, 'Time'].values, fr.loc[has_values, 'C'].values, marker=marker, linestyle='', color=color, label=label)
    ax.plot(fr.loc[has_censor, 'Time'].values, fr.loc[has_censor, 'C'].values, marker='+' if marker else '', linestyle='', color=color, label='')
    ax.plot(fr['Time'].values, fr['C'].values, marker='', linestyle='-', drawstyle=drawstyle, color=color, label='')
    if CI and ('C Lower' in fr) and ('C Upper' in fr):
        ax.plot(fr['Time'].values, fr['C Lower'].values, marker='', linestyle='--', drawstyle=drawstyle, color=color, label='')
        ax.plot(fr['Time'].values, fr['C Upper'].values, marker='', linestyle='--', drawstyle=drawstyle, color=color, label='')

    ax.set_title('Mean Cumulative Function Plot')
    ax.set_ylabel('E[C]')
    if label:
        ax.legend()
    ax.grid(True)

    # Plot Observable
    axx.plot(fr['Time'].values, fr['Y'].values, drawstyle=drawstyle, color=color, label='')
    axx.set_ylabel('Observable')
    axx.set_xlabel('Time')
    axx.grid()

    # By default expand the plot to match the data
    ax_xmin = fr['Time'].min()
    ax_xmax = fr['Time'].max()
    ax_ymin = (fr['C Lower'].min() if 'C Lower' in fr else fr['C'].min()) - 0.01
    ax_ymax = (fr['C Upper'].max() if 'C Upper' in fr else fr['C'].max()) + 0.01
    ax_yobsmin = fr['Y'].min() 
    ax_yobsmax = fr['Y'].max()  + 1
    # update
    if not axs is None:
        # mcf x axis
        old_xmin, old_xmax = ax.get_xlim()
        ax.set_xlim(ax_xmin, ax_xmax)
        new_xmin, new_xmax = ax.get_xlim()
        ax_xmin, ax_xmax = min(old_xmin, new_xmin), max(old_xmax, new_xmax)
        # mcf y axis
        old_ymin, old_ymax = ax.get_ylim()
        ax.set_ylim(ax_ymin, ax_ymax)
        new_ymin, new_ymax = ax.get_ylim()
        ax_ymin, ax_ymax = min(old_ymin, new_ymin), max(old_ymax, new_ymax)
        # observable
        old_yobsmin, old_yobsmax = axx.get_ylim()
        axx.set_ylim(ax_yobsmin, ax_yobsmax)
        new_yobsmin, new_yobsmax = axx.get_ylim()
        ax_yobsmin, ax_yobsmax = min(old_yobsmin, new_yobsmin), max(old_yobsmax, new_yobsmax)
    # Overwrite if given
    ax_xmin = xmin if not xmin is None else ax_xmin
    ax_xmax = xmax if not xmax is None else ax_xmax
    ax_ymin = ymin if not ymin is None else ax_ymin
    ax_ymax = ymax if not ymax is None else ax_ymax
    # set
    ax.set_xlim(ax_xmin, ax_xmax)
    ax.set_ylim(ax_ymin, ax_ymax)
    axx.set_ylim(ax_yobsmin, ax_yobsmax)
    
    
    return (ax,axx)

def plot_mcfs(group_fr, axs=None, subplots=False, **kwargs):
    # Plot timelines grouped by cohorts
    idx, marker, color = 0, cycle(markers), cycle(colors)
    for cohort, cohort_fr in group_fr:
        mcf_kwargs = kwargs
        if not 'color' in kwargs: mcf_kwargs = {**mcf_kwargs, **{'color': next(color)}}
        if not 'marker' in kwargs: mcf_kwargs = {**mcf_kwargs, **{'marker': next(marker)}}
        axs = plot_mcf(cohort_fr, axs, label=str(cohort) if cohort else '', **mcf_kwargs)
        if subplots: axs = None

    return axs