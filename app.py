import numpy as np
import pandas as pd
import os
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, callback, Output, Input, dash_table
import dash_bootstrap_components as dbc

from utils import *

order = ["Indeksi", "Osta", "Lisää", "Vähennä", "Myy"]
colors = {"Indeksi" : '#000000', 
          "Kaikki" : '#000000', 
          "Osta"    : '#2ca02c', 
          "Lisää"   : '#1f77b4', 
          "Vähennä" : '#ff7f0e', 
          "Myy"     : '#d62728', 
          "-"       : '#7f7f7f', }

# MCF for portfolio return
portfolio = pd.read_csv('assets/mcf_portfolio.csv', parse_dates=[3])
start_date, end_date = datetime.date(2013, 1, 1), portfolio['Time'].max().date()
# MCF for recommendation return
mcfs = pd.read_csv('assets/mcf_recommendation.csv')
mcfs['Edellinen'].fillna('', inplace=True)
# Recommendation -> Next Recommendation intervals
interval = pd.read_csv('assets/interval1.csv')
stocks = interval["Osake"].unique()
# Stock name to ticker
stock_ticker = pd.read_csv('key_kauppalehti.csv').set_index('name')['ticker']

external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Img(src=app.get_asset_url('somuchstocksmall.png'), style={'height':'128px', 'width':'128px'})
        ]),
        dbc.Col([
            html.H1("OMXH Small Cap Dashboard", style={'margin': 'auto', 'textAlign':'left', 'padding': '25px'})
        ]),
    ]),
    
    # Toteutunut tuotto
    dbc.Row([
        html.H2(children='Toteutunut tuotto'),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=start_date,
                max_date_allowed=end_date,
                initial_visible_month=start_date,
                end_date=end_date
            ),
            html.Div(children=[html.Button(id='date-reset', n_clicks=0, children='Reset')])
        ]),
        dbc.Col([
            dbc.RadioItems(options=[
                {'label': 'Inderes-efekti', 'value': True},
                {'label': 'Ei inderes-efektiä', 'value': False}
            ], value=False, id='date-inderes', inline=False)
        ]),
        dbc.Col([
            html.Div(children=[html.Div("Rullaava tuotto:"), 
                               dbc.RadioItems([
                                    {'label': '1kk', 'value': 30},
                                    {'label': '3kk', 'value': 90},
                                    {'label': '6kk', 'value': 180},
                                    {'label': '1v', 'value': 365},
                                    {'label': '2v', 'value': 365*2},
                                    {'label': '3v', 'value': 365*3}
                               ],value=365, inline=True, id='date-rolling')
                              ]),
        ])
    ]),
    
    ## Salkun tuotto
    #dbc.Row([
    #    html.H3(children='Salkun tuotto'),
    #]),
    dbc.Row([
        dbc.Col([
            html.Div(["""Tasapainotettu indeksi jossa salkun sisältönä Osta/Lisää/Vähennä/Myy-suosituksen osakkeet.
            Vertailuindeksi sisältää kaikki OMXH ja FNFI osakkeet. Indeksit on rebalansoitu joka kaupankäyntipäivä 
            ja osingot uudelleensijoitettu käyttäen päivän päätöskurssia. Ei Inderes-efektiä tarkoittaa että osake
            ostetaan vasta analyysin julkaisun jälkeiseen päätöskurssiin."""])
        ]),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='return-portfolio'))]),
    dbc.Row([dbc.Col(dcc.Graph(id='stocks-portfolio'))]),
    dbc.Row([html.Div(children=[html.Div("""Portfolion tuotto indeksiä seuraamalla ja sitä vastaava
                                            vuosituotto (Compound Annual Growth Rate, CAGR %):"""), 
                                dash_table.DataTable(id="return-tbl")])]),
    dbc.Row([dbc.Col(dcc.Graph(id='return-portfolio-range'))]),
    dbc.Row([dbc.Col(dcc.Graph(id='return-portfolio-range-diff'))]),

    ## Tuotto suosituksesta
    #dbc.Row([
    #    html.H3(children='Suosituksen tuotto'),
    #]),
    dbc.Row([
        dbc.Col([
            html.Div(["""Kassavirta suosituksesta lasketaan olettamalla että 1€ sijoitetaan osakkeeseen kun suositus
            annetaan ja osake myydään kun seuraava suositus annetaan, tuottaen X€."""])
        ]),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='return-dcf'))]),
    dbc.Row([html.Div("Kumulatiivinen kassavirta, vastaava vuosituotto (Discounted Cash Flow, DCF %):"), dash_table.DataTable(id="return-dcf-tbl")]),
    dbc.Row([dbc.Col(dcc.Graph(id='year-dcf'))]),
    dbc.Row([html.Div("Suosituksen tuotto (%):"), dash_table.DataTable(id="year-dcf-tbl")]),

    ## Inderes-efekti
    dbc.Row([
        html.H2(children='Inderes-efekti')
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(["""Suosituksen jälkeisten päivien keskimääräinen tuotto visualisoitu kumulatiivisena tuottona. """])
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div("Piirrä:"),
            dbc.RadioItems(options=[
                {'label': 'Keskiarvo', 'value': ''},
                {'label': '+Ylin/Alin', 'value': 'high-low'}
            ], value='', id='mcf-error', inline=False)
        ]),
        dbc.Col([
            dbc.RadioItems(options=[
                {'label': 'Inderes-efekti', 'value': True},
                {'label': 'Ei inderes-efektiä', 'value': False}
            ], value=False, id='mcf-inderes', inline=False)
        ]),
        dbc.Col([
            html.Div(children=[html.Div("Edellinen suositus:"),
                               dcc.Dropdown(['Kaikki', '-', 'Osta', 'Lisää', 'Vähennä', 'Myy'], 
                                            value='Kaikki', id='mcf-previous')
                               ]),
        ])
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='return-recommendation'))]),
    dbc.Row([dbc.Col(dcc.Graph(id='stocks-recommendation'))]),
    dbc.Row([html.Div("Tuotto suosituspäivänä eli Inderes-efekti (%):"), dash_table.DataTable(id="effect-tbl")]),
    
    # Individual stock return
    dbc.Row([
        html.H2(children='Osakkeen tuotto'),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(["""Osakkeen kurssihistoria verrattuna suositushistoriaan.
            Osakkeen tuotto huomioi osingot uudelleensijoittamalla irtoamispäivän päätöskurssiin."""])
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.RadioItems(options=[
                {'label': 'Inderes-efekti', 'value': True},
                {'label': 'Ei inderes-efektiä', 'value': False}
            ], value=False, id='stock-inderes', inline=False)
        ]),
        dbc.Col([
            html.Div(children=[html.Div("Osake:"),
                               dcc.Dropdown(stocks, value=stocks[0], id='stock-select')
                               ]),
        ])
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id='return-stock'))]),
    dbc.Row([html.Div("Osakkeen tuotto ajanjaksona:"), dash_table.DataTable(id="stock-cagr")]),

])


@callback(
    Output('date-picker-range', 'start_date'),
    Output('date-picker-range', 'end_date'),
    Input('date-reset', 'n_clicks')
)
def date_reset(n_clicks):
    return (start_date, end_date)
    
@callback(
    Output('return-portfolio', 'figure'),
    Output('stocks-portfolio', 'figure'),
    Output('return-portfolio-range', 'figure'),
    Output('return-portfolio-range-diff', 'figure'),
    Output('return-tbl', 'data'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('date-inderes', 'value'),
    Input('date-rolling', 'value')
)
def update_graph1(start, end, include, days):
    
    # Limit to given dates and Inderes effect
    mcfs_new = portfolio[(portfolio['Time'] >= start) & (portfolio['Time'] <= end) & (portfolio['Inderes'] == include)].copy()

    # Recalculate cumulative return
    mcfs_new['dC'] = 1 + mcfs_new['dC']
    mcfs_new['C'] = mcfs_new.groupby('Suositus')['dC'].cumprod()

    # Calculate rolling return
    time_str = "{}D".format(days)
    prod_all = lambda s: s.prod() - 1
    dC_365d = mcfs_new.groupby("Suositus").rolling(time_str, on="Time", min_periods=1)["dC"].apply(prod_all).reset_index()
    dC_365d = dC_365d.merge(dC_365d.loc[dC_365d["Suositus"] == "Indeksi"], on="Time", how="left", suffixes=("", ".Indeksi"))
    dC_365d['dC.diff'] = dC_365d['dC'] - dC_365d['dC.Indeksi']
    
    # CAGR table
    def get_cagr(mcf):
        start = mcf.iloc[0,]
        end = mcf.iloc[-1,]
        profit = end["C"]/start["C"]
        days = (end["Time"]-start["Time"]).days
        index = pd.Series({
            'CAGR (%)': np.round((profit**(365.0/days)-1)*100,2),
            'Tuotto': np.round(profit,2),#np.round((profit-1)*100,2),
            'Pitoaika': days
        })
        return(index)
    cagr = mcfs_new.groupby("Suositus").apply(get_cagr).loc[order].reset_index()

    # Figure: portfolio return
    fig1 = px.line(mcfs_new, x="Time", y="C", color="Suositus", 
                  category_orders={"Suositus": order}, color_discrete_map=colors,
                  title="Suositukseen perustuva tasapainotettu portfolio")
    fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'height':640})
    fig1.update_xaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey')
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig1.update_yaxes(title=None)
    
    # Figure: stocks in portfolio return
    fig2 = px.line(mcfs_new, x="Time", y="Y", color="Suositus", 
                  category_orders={"Suositus": order}, color_discrete_map=colors,
                  title="Osakkeiden lukumäärä salkussa")
    fig2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'height':320})
    fig2.update_xaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey')
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig2.update_yaxes(title=None)
    
    # Figure: rolling return
    fig3 = px.line(dC_365d, x="Time", y="dC", color="Suositus", 
                  category_orders={"Suositus": order}, color_discrete_map=colors,
                  title="Rullaava tuotto {} päivää".format(days))
    fig3.add_hline(y=0.0, line=dict(color="black", width=1, dash="dot"))
    fig3.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'height':400})
    fig3.update_xaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey')
    fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig3.update_yaxes(title=None)
    
    # Figure: rolling return difference to index
    fig4 = px.line(dC_365d, x="Time", y="dC.diff", color="Suositus", 
                  category_orders={"Suositus": order}, color_discrete_map=colors,
                  title="Rullaava tuotto {} päivää verrattuna indeksiin".format(days))
    fig4.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'height':400})
    fig4.update_xaxes(showgrid=False, gridwidth=1, gridcolor='lightgrey')
    fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig4.update_yaxes(title=None)

    return(fig1, fig2, fig3, fig4, cagr.to_dict('records'))


@callback(
    Output('return-dcf', 'figure'),
    Output('year-dcf', 'figure'),
    Output('return-dcf-tbl', 'data'),
    Output('year-dcf-tbl', 'data'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('date-inderes', 'value'),
    Input('date-rolling', 'value')
)
def update_graph2(start, end, include, days):
    
    interval_new = interval[(interval['Buy'] >= start) & (interval['Buy'] <= end)].copy()
    
    # cashflow with 1€ invested
    if include:
        interval_new['DCF Index (%)'] =  (1 + interval_new['Tuotto Index (%)']) * (1 + interval_new['Index (%)'])
        interval_new['DCF Inderes (%)'] = (1 + interval_new['Tuotto Inderes (%)']) * (1 + interval_new['Inderes (%)'])
    else:
        interval_new['DCF Index (%)'] =  1 + interval_new['Tuotto Index (%)']
        interval_new['DCF Inderes (%)'] = 1 + interval_new['Tuotto Inderes (%)']

    # Index cashflow over time
    n = interval_new["DCF Index (%)"].count()
    cashflow = interval_new.groupby(['Time'])['DCF Index (%)'].sum() / n
    cashflow = cashflow.rename('DCF').reset_index()
    cashflow.insert(0, "Suositus", "Indeksi")

    # Recommendations cashflow over time 
    ns = interval_new.groupby("Suositus")["DCF Inderes (%)"].count()
    cashflows = interval_new.groupby(['Suositus', 'Time'])['DCF Inderes (%)'].sum() / ns
    cashflows = cashflows.rename('DCF').reset_index()

    # Combine
    cashflows = pd.concat([cashflow, cashflows], axis=0)
    cashflows['CDCF'] = cashflows.groupby('Suositus')['DCF'].cumsum()
    
    # MCF plot of recommendation intervals
    fig0 = px.line(cashflows, x=cashflows["Time"], y=cashflows["CDCF"], color=cashflows["Suositus"], 
                  category_orders={"Suositus": order}, color_discrete_map=colors,
                 title="Kumulatiivinen kassavirta suosituksen jälkeen")
    fig0.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'height':640})
    fig0.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',range=[0,365])
    fig0.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig0.update_yaxes(title=None)
    
    # Calculate DCF from every recommendation
    dcfs = interval_new.groupby('Suositus').apply(lambda df: compute_profit(df['DCF Inderes (%)'], df['Time']/365.0))
    dcfs["Indeksi"] = compute_profit(interval_new['DCF Index (%)'], interval_new['Time']/365.0)
    dcfs = (pd.DataFrame({'DCF': dcfs}).transpose()[order]*100).round(2)
    
    # Return 365 days for every recommendation
    key1 = 'Tuotto Index {}d (%)'.format(days)
    key2 = 'Tuotto Inderes {}d (%)'.format(days)
    if include:
        interval_new['DCF Index (%)'] =  (1 + interval_new[key1]) * (1 + interval_new['Index (%)']) - 1
        interval_new['DCF Inderes (%)'] = (1 + interval_new[key2]) * (1 + interval_new['Inderes (%)']) - 1
    else:
        interval_new['DCF Index (%)'] = interval_new[key1]
        interval_new['DCF Inderes (%)'] = interval_new[key2]

    
    # 1 year returns and their confidence interval from the recommendation day
    stat = interval_new.groupby('Suositus').apply(lambda df: bootstrap_mean(df['DCF Inderes (%)']))
    stat.loc["Indeksi",] = bootstrap_mean(interval_new['DCF Index (%)'])
    stat = (stat.loc[order,]*100).round(2).reset_index()
    
    # Scatter plot of 365d returns
    fig1 = px.scatter(stat, x=stat["Suositus"], y=stat["mean"], color=stat["Suositus"],
                     error_y=stat["mean 95% upper"]-stat["mean"], error_y_minus=stat["mean"]-stat["mean 95% lower"],
                  category_orders={"Suositus": order}, color_discrete_map=colors)
    fig1.update_traces(marker=dict(size=6, line=dict(width=2, color='DarkSlateGrey')), 
                       selector=dict(mode='markers'),  error_y=dict(color='DarkSlateGrey'))
    temp = pd.concat([
        pd.DataFrame({'Suositus'  : 'Indeksi', 
                      'Tuotto (%)': (100*interval_new[key1]).round(2)}),
        pd.DataFrame({'Suositus'  : interval_new['Suositus'], 
                      'Tuotto (%)': (100*interval_new[key2]).round(2)}),
    ], axis=0)
    fig2 = px.strip(temp, x=temp["Suositus"], y=temp["Tuotto (%)"], color=temp["Suositus"], 
                  category_orders={"Suositus": order}, color_discrete_map=colors)
    fig = go.Figure(data=fig1.data + fig2.data)
    fig.update_layout({"title":"Yksittäisten osakkeiden tuotto {} päivää suosituksesta".format(days),
                       'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'height':640})
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(title=None)

    return(fig0, fig, dcfs.to_dict('records'), stat.to_dict('records'))


@callback(
    Output('return-recommendation', 'figure'),
    Output('stocks-recommendation', 'figure'),
    Output('effect-tbl', 'data'),
    Input('mcf-error', 'value'),
    Input('mcf-inderes', 'value'),
    Input('mcf-previous', 'value')
)
def update_graph3(error, include, previous):
    
    # Select previous recommendation and inderes effect
    mcf_new = mcfs[(mcfs['Edellinen'] == previous) & (mcfs['Inderes'] == include)].copy()

    # Bootstrap / Low-High / no errors
    if error == "high-low":
        ymin, ymax = -mcf_new['Alin (%)'], mcf_new['Ylin (%)']
    else:
        ymin, ymax = None, None

    # Figure MCF
    fig1 = px.line(mcf_new, x="Time", y="C", color="Suositus", 
                  error_y_minus=ymin, error_y=ymax,
                  category_orders={"Suositus": order}, color_discrete_map=colors, line_shape='hv',
                 title="Kumulatiivinen tuotto suosituksen jälkeen")
    fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'height':640})
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')#range=[0,120]
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')#range=[0.8,1.2]
    fig1.update_yaxes(title=None)

    # Observable counts
    fig2 = px.line(mcf_new, x="Time", y="Y", color="Suositus", 
                  category_orders={"Suositus": order}, color_discrete_map=colors, line_shape='hv', 
                 title="Osakkeiden lukumäärä joilla suositus")
    fig2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)', 'height':320})
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig2.update_yaxes(title=None)
    
    start_day = (mcfs["Time"] == 0) & (mcfs['Inderes'] == True)
    effect = mcfs[start_day].pivot(index="Edellinen", columns="Suositus", values="dC")
    effect = np.round(effect*100,2).loc[["Osta", "Lisää", "Vähennä", "Myy", "Kaikki"],
                                       ["Osta", "Lisää", "Vähennä", "Myy"]].reset_index()

    return(fig1, fig2, effect.to_dict('records'))

@callback(
    Output('return-stock', 'figure'),
    Output('stock-cagr', 'data'),
    Input('stock-inderes', 'value'),
    Input('stock-select', 'value')
)
def update_graph4(include, stock):
    
    #Price time series
    ticker = stock_ticker[stock]
    filename = os.path.join('dls', '{}.csv'.format(ticker))
    stock_prices = pd.read_csv(filename, sep=';', decimal=',', parse_dates=[0])
    stock_prices = stock_prices.loc[~stock_prices['Päivämäärä'].duplicated(keep='first'),]
    # Recommendation intervals
    stock_interval = interval[interval['Osake'] == stock].copy()

    # OHLC stock price
    fig = go.Figure(data=go.Ohlc(x=stock_prices['Päivämäärä'],
                        open=stock_prices['Keskimäärin'],
                        high=stock_prices['Ylin'],
                        low=stock_prices['Alin'],
                        close=stock_prices['Päätöskurssi']))
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(title=dict(text=stock))

    # Plot recommendation intervals
    for suositus in ['Lisää', 'Vähennä', 'Osta', 'Myy', '-']:
        intervals_suositus = stock_interval[stock_interval["Suositus"] == suositus]
        for (x0, x1) in zip(intervals_suositus["Buy"], intervals_suositus["Sell"]):
            fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor=colors[suositus], opacity=0.2)

    def get_statistics(df, inderes=False):
        if inderes:
            profit = (1+df['Tuotto Inderes (%)'] + stock_interval['Inderes (%)']).prod()
        else:
            profit = (1+df['Tuotto Inderes (%)']).prod()
        
        index = (1+df['Tuotto Index (%)']).prod()
        days = df['Time'].sum()
        stats = pd.Series({
            'Tuotto Suositus': profit,
            'Tuotto Index': index,
            'Pitoaika ': days,
            'CAGR Suositus (%)': ((profit**(365/days)-1)*100).round(2) if days > 0 else 1.0,
            'CAGR Index (%)': ((index**(365/days)-1)*100).round(2) if days > 0 else 1.0,
        })
        return(stats)

    cagr = pd.concat([
        stock_interval.groupby('Suositus').apply(lambda df: get_statistics(df, inderes=include)),
        pd.DataFrame({"Buy&Hold": get_statistics(stock_interval)}).transpose()],
    axis=0).round(2).reindex(index=['Buy&Hold', 'Osta', 'Lisää', 'Vähennä', 'Myy']).reset_index()
    return(fig, cagr.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)
