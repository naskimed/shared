
import os
from mesa.space import MultiGrid
from mesa import Agent, Model
from mesa.time import RandomActivation,BaseScheduler
import random
from mesa.datacollection import DataCollector
from datetime import datetime  
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd 
import seaborn as sns
from dash import Dash, dcc, html, Input, Output, callback, State, no_update, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
from dash_bootstrap_components.themes import BOOTSTRAP
import time
import dash_bootstrap_components as dbc
import copy
from dash.exceptions import PreventUpdate\
### run .py
import pandas as pd
import datetime
import time
from dash_app import *
import math
from statistics import mean, median
from layout import create_layout
from backend_dash import *
from perf_sheet import *



#########################
#########       #########
######### input #########
#########       #########
#########################

path=""# "C:/CloudStation/Hanon/CloudStation/Prototypes/ABM/"
filename=path+ "SP500_us_blend1.xlsx"
filename=path+ "for_alfie.xlsx"


print("opening file", filename)
xl = pd.ExcelFile(filename)
sheets_name="prices" #xl.sheet_names[0]
all_prices=xl.parse(sheets_name)
all_prices['date']=all_prices['date'].dt.date
bmk="^GSPC"
bmk=all_prices.columns[1]
prices_bmk=all_prices[['date',bmk]].set_index('date')
prices_df=all_prices.drop(bmk, axis=1)
prices_df['Cash']=1.
prices_df=prices_df.ffill().set_index('date')
stock_universe=prices_df.columns

sheets_name="starting_holdings"
starting_holdings=xl.parse(sheets_name).set_index('security_id')['nominal']


sheets_name="dates"
simulation_dates_df=xl.parse(sheets_name)
simulation_dates=simulation_dates_df['date'].dt.date
#print(simulation_dates_df.columns)

###
#dates_ref = simulation_dates_df['date']
ptf_id_ref = simulation_dates_df['ptf_id']
nav_ref = simulation_dates_df['nav']
performance_ref = simulation_dates_df['performance']
hit_ratio_ref = simulation_dates_df['hit_ratio']
win_loss_ratio_ref = simulation_dates_df['win_loss_ratio']
daily_performance_ref = simulation_dates_df['daily_performance']

""" sheets_name = "reference_bets"
reference_bets = xl.parse(sheets_name) """


xl.close()
start_date=simulation_dates.min()
end_date=simulation_dates.max()


list_available=prices_df.loc[start_date, prices_df.columns[:-1]].T.to_frame(name='price')
stock_universe=list_available.loc[list_available['price'].notnull()].index

#########################
#########       #########
####### simulate ########
#########       #########
#########################


bets_cols=["security_id","last_decision", "next_decision", "start_date",  "end_date", "performance"]
start_time= time.time()

portfolio_manager_behaviour={"style":
                                        {
                                        "buy":
                                            {
                                            "number_of_days":20,
                                            "momentum1":
                                                        {
                                                        "momentum_level_min":-0.05,
                                                        "momentum_level_max":0.025,
                                                        "percentage":0.5
                                                        },
                                            "momentum2":
                                                        {
                                                        "momentum_level_min":0.025,
                                                        "momentum_level_max":0.1,
                                                        "percentage":0.5
                                                        }
                                            },
                                        "sell":
                                            {
                                            "number_of_days":20,
                                            "momentum1":
                                                        {
                                                        "momentum_level_min":-0.05,
                                                        "momentum_level_max":0.025,
                                                        "percentage":0.5
                                                        },
                                            "momentum2":
                                                        {
                                                        "momentum_level_min":0.025,
                                                        "momentum_level_max":0.1,
                                                        "percentage":0.5
                                                        }
                                            },
                                        },
                                        
                    
                            "buy_behaviour":
                                            {"min_cash":0.01,
                                            "max_wght_buy":0.01,  # not a max 
                                            "nb_max_building_stock":1000,
                                            "buy_every_days":2},
                            "sell_behaviour":
                                            {"max_cash":0.5,
                                            "sell_every_days":1,
                                            },
                            "scale_up_behaviour":
                                            {"min_cash":0.01,
                                            "max_weight":0.01,
                                            "increment":0.025,
                                            "nb_max_building_stock":10,
                                            "scale_up_every_days":2,
                                            "weight_to_scale_up":0.0025
                                            },
        
                            "scale_down_behaviour":
                                            {"scale_down_every_days":2,
                                             "weight_to_scale_down":0.0025
                                            }
                            }

debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True


app = Dash(__name__,url_base_pathname = '/abm_dash/', external_stylesheets=[BOOTSTRAP], prevent_initial_callbacks='initial_duplicate')


server = app.server

if (os.environ["DASH_AUTH_MODE"]== "True"):
    auth = dash_auth.BasicAuth(
        app,
        USERNAME_PASSWORD_PAIRS
    )


# Function to generate performance graph
def generate_performance_graph(data):
    fig = go.Figure()
    #print(data.columns)
    for p in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[p], name=f'Portfolio {p}'))

    fig.update_layout(title='Portfolio Performance', title_x=0.5)
    return fig


def generate_performance_graph_ref(data, df):
    fig = go.Figure()
    df1 = df.set_index('date').loc[data.index]
     
    # Add traces for other portfolios
    for p in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[p], name=f'Portfolio {p}'))
    
    # Add trace for the reference portfolio with a thicker line
    fig.add_trace(go.Scatter(x=df1.index, y=df1.performance, name='Reference Portfolio', line=dict(width=3)))
   
    fig.update_layout(title='Portfolio Performance', title_x=0.5)
    return fig

def generate_buy_pipeline_size_graph(data):
    fig = go.Figure()
    for p in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[p], name=f'Portfolio {p}'))

    fig.update_layout(title='Buy Pipeline Size', title_x=0.5)
    return fig

def generate_sell_pipeline_size_graph(data):
    fig = go.Figure()
    for p in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[p], name=f'Portfolio {p}'))

    fig.update_layout(title='Sell Pipeline Size', title_x=0.5)
    return fig

def generate_cash_graph(data):
    fig = go.Figure()
    for p in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[p], name=f'Portfolio {p}'))

    fig.update_layout(title='Cash Weights', title_x=0.5)
    return fig



# Function to generate holdings weights data and figures
def generate_holdings_weights_data_figures(data, nb_ptf):
    holdings_weights_figures = []
    for i in range(0, nb_ptf):
        variable = 'holdings_weights'
        wghts_hist = pd.DataFrame.from_dict(list(data[data.index.get_level_values('AgentID') == i][variable]))
        wghts_hist['date'] = data[data.index.get_level_values('AgentID') == i]['date'].values
        wghts_hist = wghts_hist.set_index('date')

        fig = px.area(wghts_hist, title='Weights p=' + str(i + 1), color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_yaxes(title='Percent (%)')
        fig.update_layout(title=dict(x=0.5))

        graph = dcc.Graph(figure=fig)
        holdings_weights_figures.append(graph)

    return holdings_weights_figures


# Function to generate performance attribution data and figures
def generate_performance_attributions_data_figures(data, number_p):
    performance_attributions_figures = []
    for i in range(0, number_p):
        variable = 'performance_attribution_active_bets'
        wghts_hist = pd.DataFrame.from_dict(list(data[data.index.get_level_values('AgentID') == i][variable]))
        wghts_hist['date'] = data[data.index.get_level_values('AgentID') == i]['date'].values
        wghts_hist = wghts_hist.set_index('date')

        fig = px.line(wghts_hist, title=f'Performance Attribution for Portfolio {i + 1}')
        fig.update_layout(yaxis_title='Percent (%)')
        fig.update_layout(title=dict(x=0.5))

        graph = dcc.Graph(figure=fig)
        performance_attributions_figures.append(graph)

    return performance_attributions_figures


def generate_hit_ratio_performance_scatter(behave):
    last_row = behave.iloc[-1]  # Get the last row of data
    marker_color = ["blue"] * (len(behave) - 1) + ["green"]  # Set marker color for all data points except the last one

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=behave["hit_ratio"], y=behave['performance']*100, mode='markers', marker=dict(color=marker_color)))
    fig.update_layout(title='Hit Ratio vs Performance', title_x=0.5, xaxis_title='Hit Ratio', yaxis_title='Performance')
    return fig

def generate_win_loss_ratio_performance_scatter(behave):
    last_row = behave.iloc[-1]  # Get the last row of data
    marker_color = ["red"] * (len(behave) - 1) + ["green"]  # Set marker color for all data points except the last one

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=behave["win_loss_ratio"], y=behave['performance']*100, mode='markers', marker=dict(color=marker_color)))
    fig.update_layout(title='Win/Loss Ratio vs Performance', title_x=0.5, xaxis_title='Win/Loss Ratio', yaxis_title='Performance')
    return fig


def generate_hit_ratio_win_loss_ratio_scatter(behave):
    last_row = behave.iloc[-1]  # Get the last row of data
    ratio = 100 / behave['performance'].abs().max()
    
    # Set marker color for all data points except the last one
    couleur = ['blue' if x < 0 else 'green' for x in behave['performance']]
    marker_color = couleur[:-1] + ['red']  # Assign 'red' color to the marker of the last data point

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=behave["hit_ratio"],
        y=behave['win_loss_ratio'],
        mode='markers',
        marker=dict(color=marker_color, size=behave['performance'].abs()*ratio, opacity=0.25)
    ))
    fig.update_layout(title='Hit Ratio vs Win/Loss Ratio', title_x=0.5, xaxis_title='Hit Ratio', yaxis_title='Win/Loss Ratio')
    return fig






####################
## Code for Table ##
####################
'''
tableData = {
        "Metric": ["Average", "Median", "Max", "Min"],
        "Total Return": [10.5, 9.8, 12.3, 8.1],
        "Volatility": [0.15, 0.12, 0.18, 0.09],
        "Information Ratio": [1.2, 1.1, 1.5, 0.9],
        "Max Drawdown": [0.2, 0.15, 0.25, 0.12],
        "Hit Ratio": [0.6, 0.55, 0.65, 0.5],
        "Win Loss Ratio": [2.5, 2.2, 3.0, 2.0]
    }
dfTableData = pd.DataFrame(tableData)
'''


def IR(data):
    stdev=data.std()*math.sqrt(255)
    perf=data.sum()
    return (perf/stdev)
 

def MaxDrawdown(data):
    dd_series=data.cumsum() - data.cumsum().cummax()
    max_dd=dd_series.min()
    return(max_dd)
 

def MaxDrawdownDuration(data):
    dd_series=data.cumsum() - data.cumsum().cummax()
    #max_dd=dd_series.min()
    end_dd=dd_series.argmin()
    start_dd=data.cumsum()[:end_dd].argmax()
    return ((end_dd-start_dd)/datetime.timedelta (days=1))
 

def MaxDrawdownDurationRecovery(data):
    dd_series=data.cumsum() - data.cumsum().cummax()
    #max_dd=dd_series.min()
    end_dd=dd_series.argmin()
    start_dd=data.cumsum()[:end_dd].argmax()
    recovery=dd_series[end_dd:].loc[dd_series[end_dd:]>=dd_series[start_dd]]
    if len(recovery)>0:
        first_recovery=recovery.index.values[0]
        return ((first_recovery-end_dd)/datetime.timedelta (days=1))
    else:
        return('NaN')

def calculateMetrics(data):
    avg = mean(data)
    med = median(data)
    maximum = max(data)
    minimum = min(data)
    return avg, med, maximum, minimum
'''
def generateTable(data, totralreturndata, volatilitydata, informationratiodata,maxdrawdowndata, hitratiodata, winlossratiodata):
    tableData = {
        "Metric": ["Average", "Median", "Max", "Min"],
        "Total Return": calculateMetrics(totralreturndata),
        "Volatility": calculateMetrics(volatilitydata),
        "Information Ratio": calculateMetrics(informationratiodata),
        "Max Drawdown": calculateMetrics(maxdrawdowndata),
        "Hit Ratio": calculateMetrics(hitratiodata),
        "Win Loss Ratio": calculateMetrics(winlossratiodata)
    }
    df = pd.DataFrame(tableData)
    return df
'''

def generateTable(hitRatiodata, winLossRatioData):
    tableData = {
        "Metric": ["Average", "Median", "Max", "Min"],
        "Hit Ratio": calculateMetrics(hitRatiodata),
        "Win Loss Ratio": calculateMetrics(winLossRatioData)
    }
    df = pd.DataFrame(tableData)
    return df

# Define layout
app.layout = create_layout()

portfolio = None

simulation_results = {}

# Define the run_simulation callback
@app.callback(
    Output('clientside-stored-nav', 'data'),
    Output('clientside-stored-performance', 'data'),
    Output('clientside-stored-buy-pipeline-size', 'data'),
    Output('clientside-stored-sell-pipeline-size', 'data'),
    Output('clientside-stored-cash', 'data'),
    Output('clientside-stored-weights', 'data'),
    Output('clientside-stored-performance-attribution', 'data'),
    Output('clientside-stored-hit-ratio-performance', 'data'),
    Output('clientside-stored-win-loss-performance', 'data'),
    Output('clientside-stored-hit-ratio-win-loss', 'data'),
    Output('clientside-stored-data-table', 'data'),
    Output('clientside-stored-steps', 'data'),
    Output('clientside-stored-interval-component', 'data'),
    Output('clientside-stored-total-return', 'data'),
    Output('clientside-stored-volatility', 'data'),
    Output('clientside-stored-IR', 'data'),
    Output('clientside-stored-max-dropdown', 'data'),
    Output('clientside-stored-hit-ratio', 'data'),
    Output('clientside-stored-win-loss-ratio', 'data'),
    Output('clientside-stored-stored', 'data'),
    Output('clientside-stored-line-chart', 'data'),
    Output('clientside-stored-line-chart2', 'data'),

    Input('run-button', 'n_clicks'),
    State('max-cash-input', 'value'),
    State('sell-every-days-input', 'value'),
    State('min-cash-input', 'value'),
    State('buy-every-days-input', 'value'),
    State('max-weight-buy-input', 'value'),
    State('nb-max-building-stock-buy', 'value'),
    State('weight-to-scale-down-input', 'value'),
    State('scale-down-every-days-input', 'value'),
    State('scale-up-min-cash-input', 'value'),
    State('max-weight-scale-up-input', 'value'),
    State('increment-input', 'value'),
    State('nb-max-building-stock-scale-up', 'value'),
    State('scale-up-every-days-input', 'value'),
    State('weight-to-scale-up-input', 'value'),
    State('simulation-days-input', 'value'),
    State('number-portfolios-input', 'value'),
    State('style-one-percentage', 'value'),
    State('style-two-percentage', 'value'),
    State('style-num-days', 'value'),
    State('style-one-min', 'value'),
    State('style-one-max', 'value'),
    State('style-two-min', 'value'),
    State('style-two-max', 'value'),
    State('sell-style-one-percentage', 'value'),
    State('sell-style-two-percentage', 'value'),
    State('sell-style-num-days', 'value'),
    State('sell-style-one-min', 'value'),
    State('sell-style-one-max', 'value'),
    State('sell-style-two-min', 'value'),
    State('sell-style-two-max', 'value'),
    State('stored', 'value')
)
def run_simulation(n_clicks, max_cash, sell_every_days, min_cash_buy, buy_every_days, max_wght_buy, nb_max_building_stock,
                    weight_to_scale_down, scale_down_every_days, min_cash_scale, max_weight, increment,
                    nb_max_building_stock_scale, scale_up_every_days, weight_to_scale_up, simulation_days, number_portfolios, 
                    percentage_one, percentage_two, style_days, min_momentum_one, max_momentum_one, min_momentum_two, max_momentum_two,
                    sell_percentage_one, sell_percentage_two, sell_style_days, sell_min_momentum_one, sell_max_momentum_one, sell_min_momentum_two, sell_max_momentum_two,
                    stored
                    ):
    # Check if the button was clicked
    if n_clicks is None:
        # Button not clicked yet, return initial figures

        raise PreventUpdate
    
    print("Simulation 1: post call")
    begin = time.time()
    steps = 0

    # Update the portfolio_manager_behaviour dictionary with the new input values
    portfolio_manager_behaviour['sell_behaviour']['max_cash'] = max_cash
    portfolio_manager_behaviour['sell_behaviour']['sell_every_days'] = sell_every_days
    portfolio_manager_behaviour['buy_behaviour']['min_cash'] = min_cash_buy
    portfolio_manager_behaviour['buy_behaviour']['buy_every_days'] = buy_every_days
    portfolio_manager_behaviour['buy_behaviour']['max_wght_buy'] = max_wght_buy
    portfolio_manager_behaviour['buy_behaviour']['nb_max_building_stock'] = nb_max_building_stock
    portfolio_manager_behaviour['scale_down_behaviour']['weight_to_scale_down'] = weight_to_scale_down
    portfolio_manager_behaviour['scale_down_behaviour']['scale_down_every_days'] = scale_down_every_days
    portfolio_manager_behaviour['scale_up_behaviour']['min_cash'] = min_cash_scale
    portfolio_manager_behaviour['scale_up_behaviour']['max_weight'] = max_weight
    portfolio_manager_behaviour['scale_up_behaviour']['increment'] = increment
    portfolio_manager_behaviour['scale_up_behaviour']['nb_max_building_stock'] = nb_max_building_stock_scale
    portfolio_manager_behaviour['scale_up_behaviour']['scale_up_every_days'] = scale_up_every_days
    portfolio_manager_behaviour['scale_up_behaviour']['weight_to_scale_up'] = weight_to_scale_up
    portfolio_manager_behaviour['style']['buy']['momentum1']['percentage'] = percentage_one / 100
    portfolio_manager_behaviour['style']['buy']['momentum2']['percentage'] = percentage_two / 100
    portfolio_manager_behaviour['style']['buy']['number_of_days'] = style_days
    portfolio_manager_behaviour['style']['buy']['momentum1']['momentum_level_min'] = min_momentum_one
    portfolio_manager_behaviour['style']['buy']['momentum1']['momentum_level_max'] = max_momentum_one
    portfolio_manager_behaviour['style']['buy']['momentum2']['momentum_level_min'] = min_momentum_two
    portfolio_manager_behaviour['style']['buy']['momentum2']['momentum_level_max'] = max_momentum_two
    portfolio_manager_behaviour['style']['sell']['momentum1']['percentage'] = sell_percentage_one / 100
    portfolio_manager_behaviour['style']['sell']['momentum2']['percentage'] = sell_percentage_two / 100
    portfolio_manager_behaviour['style']['sell']['number_of_days'] = sell_style_days
    portfolio_manager_behaviour['style']['sell']['momentum1']['momentum_level_min'] = sell_min_momentum_one
    portfolio_manager_behaviour['style']['sell']['momentum1']['momentum_level_max'] = sell_max_momentum_one
    portfolio_manager_behaviour['style']['sell']['momentum2']['momentum_level_min'] = sell_min_momentum_two
    portfolio_manager_behaviour['style']['sell']['momentum2']['momentum_level_max'] = sell_max_momentum_two

    # Create a new instance of the Portfolio class with the updated input values
    nb_ptf = number_portfolios
    global portfolio

    if portfolio_manager_behaviour['style']['buy']['number_of_days'] == 200:
        portfolio = Portfolio(nb_pt, starting_holdings, portfolio_manager_behaviour, simulation_dates, prices_df, prices_bmk, diff)
    else:
        diff = get_perf(prices_df, starting_holdings, prices_bmk, simulation_dates,portfolio_manager_behaviour)
        portfolio = Portfolio(nb_ptf, starting_holdings, portfolio_manager_behaviour, simulation_dates, prices_df, prices_bmk, diff)


    # Run the simulation loop
    for date in simulation_dates[:simulation_days]:
        portfolio.step()
        steps += 1
        if steps == 10:
            break

    portfolio_copy = copy.deepcopy(portfolio)
    
    # Get the updated data for the graphs
    data = portfolio_copy.datacollector.get_agent_vars_dataframe()
  
    if (steps == simulation_days):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp)
        simulation_results[timestamp] = {
            'portfolio': portfolio_copy,
            'data': data,
            'steps': steps,
            'nb_ptf': nb_ptf
        }
        stored += 1
        print(stored)
    end = time.time()
    totalTime = end - begin
    print(f"Time taken before generating graphs: {totalTime:.6f} seconds")


    #Alphie
    nav_graph, perf_graph, cash_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable, gauges, survival_graph, PN_survival_graph = update_data_and_generate_outputs(portfolio_copy, data, steps, performance_ref, daily_performance_ref, hit_ratio_ref, win_loss_ratio_ref, simulation_dates_df, ptf_id_ref, end_date)
    
    #Naski
    #nav_graph, perf_graph, cash_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable,  = update_data_and_generate_outputs(portfolio_copy, data)

    # Generate holdings weights data and figures
    if nb_ptf <= 10:
        holdings_weights_figures = generate_holdings_weights_data_figures(data, nb_ptf)

        # Generate performance attributions data and figures
        performance_attributions_figures = generate_performance_attributions_data_figures(data, nb_ptf)
    else:
        holdings_weights_figures = []
        performance_attributions_figures = []

    print("Simulation 1: pre return")

    time.sleep(3)


    #Alphie
    return nav_graph, perf_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, cash_graph, holdings_weights_figures, performance_attributions_figures, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable, steps, False, gauges['totalReturnGauge'], gauges['volatilityGauge'], gauges['IRGauge'], gauges['maxDrawdownGauge'], gauges['hitRatioGauge'], gauges['winLossRatioGauge'], stored,survival_graph, PN_survival_graph
    
    #Naski
    #return nav_graph, perf_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, cash_graph, holdings_weights_figures, performance_attributions_figures, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable, steps, False, survival_graph, PN_survival_graph

# Clientside callback
app.clientside_callback(
    """
    function(nav_graph) {
        const nav_fig = Object.assign({}, nav_graph, {
                'layout': {
                    ...nav_graph.layout,
                }
        });
        
        return nav_fig
    }

    """,
    Output('nav-graph-container', 'children', allow_duplicate=True),
    Input('clientside-stored-nav', 'data'),

)

app.clientside_callback(
    """
    function(perf_graph) {
        
        const perf_fig = Object.assign({}, perf_graph, {
                'layout': {
                    ...perf_graph.layout,
                }
        });
        return perf_fig
    }
    """,
    Output('performance-graph-container', 'children', allow_duplicate=True),
    Input('clientside-stored-performance', 'data'),
),
    

app.clientside_callback(
    """
    function(buy_pipeline_size_graph) {

        const buy_pipeline_size_fig = Object.assign({}, buy_pipeline_size_graph, {
                'layout': {
                    ...buy_pipeline_size_graph.layout,
                }
        });

        return buy_pipeline_size_fig
    }
    """,
    Output('buy-pipeline-size-graph-container', 'children', allow_duplicate=True),
    Input('clientside-stored-buy-pipeline-size', 'data'),
),

app.clientside_callback(
    """
    function(sell_pipeline_size_graph) {

        const sell_pipeline_size_fig = Object.assign({}, sell_pipeline_size_graph, {
                'layout': {
                    ...sell_pipeline_size_graph.layout,
                }
        });

        return sell_pipeline_size_fig
    }
    """,
    Output('sell-pipeline-size-graph-container', 'children', allow_duplicate=True),
    Input('clientside-stored-sell-pipeline-size', 'data'),
),

app.clientside_callback(
    """
    function(cash_graph) {

        const cash_fig = Object.assign({}, cash_graph, {
                'layout': {
                    ...cash_graph.layout,
                }
        });

        return cash_fig
    }
    """,
    Output('cash-graph-container', 'children', allow_duplicate=True),
    Input('clientside-stored-cash', 'data'),
),

app.clientside_callback(
    """
    function(holdings_weights_figures) {
        // Create an array to hold the generated graphs
        const graphs = [];

        // Loop through the data and create dcc.Graph components dynamically
        for (const graphKey in holdings_weights_figures) {
            graphs.push(
                graphKey
            );
            
        }

        return graphs;
    }
    """,
    Output('weights-graph-container', 'children', allow_duplicate=True),
    Input('clientside-stored-weights', 'data'),
),

app.clientside_callback(
    """
    function(performance_attributions_figures) {
        // Create an array to hold the generated graphs
        const graphs = [];

        // Loop through the data and create dcc.Graph components dynamically
        for (const graphKey in performance_attributions_figures) {
            graphs.push(
            graphKey
        );
        }

        return graphs;
    }
    """,
    Output('performance-attribution-container', 'children', allow_duplicate=True),
    Input('clientside-stored-performance-attribution', 'data'),
),

app.clientside_callback(
    """
    function(hit_ratio_performance_graph) {

        const hit_ratio_performance_fig = Object.assign({}, hit_ratio_performance_graph, {
            'layout': {
                ...hit_ratio_performance_graph.layout,
            }
        });

        return hit_ratio_performance_fig
    }
    """,
    Output('hit-ratio-performance-container', 'children', allow_duplicate=True),
    Input('clientside-stored-hit-ratio-performance', 'data'),
),

app.clientside_callback(
    """
    function(win_loss_performance_graph) {

        const win_loss_performance_fig = Object.assign({}, win_loss_performance_graph, {
            'layout': {
                ...win_loss_performance_graph.layout,
            }
        });

        return win_loss_performance_fig
    }
    """,
    Output('win-loss-performance-container', 'children', allow_duplicate=True),
    Input('clientside-stored-win-loss-performance', 'data'),
),

app.clientside_callback(
    """
    function(hit_ratio_win_loss_graph) {

        const hit_ratio_win_loss_fig = Object.assign({}, hit_ratio_win_loss_graph, {
            'layout': {
                ...hit_ratio_win_loss_graph.layout,
            }
        });

        return hit_ratio_win_loss_fig
    }
    """,
    Output('hit-ratio-win-loss-container', 'children', allow_duplicate=True),
    Input('clientside-stored-hit-ratio-win-loss', 'data'),
),

app.clientside_callback(
    """
    function(dataTable) {

        const dataTable_fig = Object.assign({}, dataTable, {
            'layout': {
                ...dataTable.layout,
            }
        });

        return dataTable_fig
    }
    """,
    Output('data-table-container', 'children', allow_duplicate=True),
    Input('clientside-stored-data-table', 'data'),
),

app.clientside_callback(
    """
    function(steps) {

        const steps_fig = steps;

        return steps_fig
    }
    """,
    Output('steps', 'value', allow_duplicate=True),
    Input('clientside-stored-steps', 'data'),
),

app.clientside_callback(
    """
    function(FalseVar) {

        const FalseVar_fig = false
    
        return FalseVar
    }
    """,
    Output('interval-component', 'disabled', allow_duplicate=True),
    Input('clientside-stored-interval-component', 'data'),
),

app.clientside_callback(
    """
    function(totalReturnGauge) {

        const totalReturnGauge_fig = Object.assign({}, totalReturnGauge, {
            'layout': {
                ...totalReturnGauge.layout,
            }
        });

        return totalReturnGauge_fig
    }
    """,
    Output('total-return-gauge', 'children', allow_duplicate=True),
    Input('clientside-stored-total-return', 'data'),
),

app.clientside_callback(
    """
    function (volatilityGauge) {

        const volatilityGauge_fig = Object.assign({}, volatilityGauge, {
            'layout': {
                ...volatilityGauge.layout,
            }
        });

        return volatilityGauge_fig
    }
    """,
    Output('volatility-gauge', 'children', allow_duplicate=True),
    Input('clientside-stored-volatility', 'data'),
),

app.clientside_callback(
    """
    function(IRGauge) {
    
        const IRGauge_fig = Object.assign({}, IRGauge, {
            'layout': {
                ...IRGauge.layout,
            }
        });

        return IRGauge_fig
    }
    """,
    Output('IR-gauge', 'children', allow_duplicate=True),
    Input('clientside-stored-IR', 'data'),
),

app.clientside_callback(
    """
    function(maxDrawdownGauge) {

        const  maxDrawdownGauge_fig = Object.assign({}, maxDrawdownGauge, {
            'layout': {
                ...maxDrawdownGauge.layout,
            }
        });
    
        return maxDrawdownGauge_fig
    }
    """,
    Output('max-dropdown-gauge', 'children', allow_duplicate=True),
    Input('clientside-stored-max-dropdown', 'data'),
),

app.clientside_callback(
    """
    function(hitRatioGauge) {

        const hitRatioGauge_fig = Object.assign({}, hitRatioGauge, {
            'layout': {
                ...hitRatioGauge.layout,
            }
        });

        return hitRatioGauge_fig
    }
    """,
    Output('hit-ratio-gauge', 'children', allow_duplicate=True),
    Input('clientside-stored-hit-ratio', 'data'),
),

app.clientside_callback(
    """
    function(winLossRatioGauge) {

        const winLossRatioGauge_fig = Object.assign({}, winLossRatioGauge, {
            'layout': {
                ...winLossRatioGauge.layout,
            }
        });

        return winLossRatioGauge_fig
    }
    """,
    Output('win-loss-ratio-gauge', 'children', allow_duplicate=True),
    Input('clientside-stored-win-loss-ratio', 'data'),
),

app.clientside_callback(
    """
    function(stored) {
        
        const stored_fig = stored;

        return stored_fig
    }
    """,
    Output('stored', 'value', allow_duplicate=True),
    Input('clientside-stored-stored', 'data'),
),

app.clientside_callback(
    """
    function(survival_graph) {

        const survival_graph_fig = Object.assign({}, survival_graph, {
            'layout': {
                ...survival_graph.layout,
            }
        });

        return survival_graph_fig
    }
    """,
    Output('line-chart', 'children',allow_duplicate=True),
    Input('clientside-stored-line-chart', 'data'),
),

app.clientside_callback(
    """
    function(PN_survival_graph) {

        const PN_survival_graph_fig = Object.assign({}, PN_survival_graph, {
            'layout': {
                ...PN_survival_graph.layout,
            }
        });

        return PN_survival_graph_fig
    }
    """,

    #Outputs

    Output('line-chart2', 'children', allow_duplicate=True),

    # Inputs

    Input('clientside-stored-line-chart2', 'data'),
)




@app.callback(

    Output('clientside-stored-nav', 'data', allow_duplicate=True),
    Output('clientside-stored-performance', 'data', allow_duplicate=True),
    Output('clientside-stored-buy-pipeline-size', 'data', allow_duplicate=True),
    Output('clientside-stored-sell-pipeline-size', 'data', allow_duplicate=True),
    Output('clientside-stored-cash', 'data', allow_duplicate=True),
    Output('clientside-stored-weights', 'data', allow_duplicate=True),
    Output('clientside-stored-performance-attribution', 'data', allow_duplicate=True),
    Output('clientside-stored-hit-ratio-performance', 'data', allow_duplicate=True),
    Output('clientside-stored-win-loss-performance', 'data', allow_duplicate=True),
    Output('clientside-stored-hit-ratio-win-loss', 'data', allow_duplicate=True),
    Output('clientside-stored-data-table', 'data', allow_duplicate=True),
    Output('clientside-stored-total-return', 'data', allow_duplicate=True),
    Output('clientside-stored-volatility', 'data', allow_duplicate=True),
    Output('clientside-stored-IR', 'data', allow_duplicate=True),
    Output('clientside-stored-max-dropdown', 'data', allow_duplicate=True),
    Output('clientside-stored-hit-ratio', 'data', allow_duplicate=True),
    Output('clientside-stored-win-loss-ratio', 'data', allow_duplicate=True),
    Output('clientside-stored-line-chart', 'data', allow_duplicate=True),
    Output('clientside-stored-line-chart2', 'data', allow_duplicate=True),
    Output('clientside-stored-steps', 'data', allow_duplicate=True),
    Output('clientside-stored-interval-component', 'data', allow_duplicate=True),
    Output('clientside-stored-stored', 'data', allow_duplicate=True),


    Input('interval-component', 'n_intervals'),
    Input('simulation-days-input', 'value'),
    State('number-portfolios-input', 'value'),
    State('steps', 'value'),
    State('stored', 'value'),
    prevent_initial_call=True
)
def update_portfolio_data(n_intervals, simulation_days, number_portfolios, num_steps, stored):
    # Update the portfolio data for subsequent steps
    print("Simulation 2: post call")
    begin = time.time()
    
    global portfolio
    if (portfolio == None):
        print("Simulation 2: Portfolio == None")
        raise PreventUpdate

    nb_ptf = number_portfolios
    steps = num_steps
    if steps == simulation_days:
        print("Steps == simulation days")


        #Alphie
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, no_update
        
        #Naski
        #return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, True 
    
   # Run the simulation loop
    for date in simulation_dates[steps:simulation_days]:
        portfolio.step()
        steps += 1
        if steps % 10 == 0:
            break
    
    portfolio_copy = copy.deepcopy(portfolio)
    
    # Get the updated data for the graphs
    data = portfolio_copy.datacollector.get_agent_vars_dataframe()

    if (steps == simulation_days):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestamp)
        simulation_results[timestamp] = {
            'portfolio': portfolio_copy,
            'data': data,
            'steps': steps,
            'nb_ptf': nb_ptf
        }
        stored += 1
    end = time.time()
    totalTime = end - begin
    print(f"Time taken before generating graphs: {totalTime:.6f} seconds")

    begin = time.time()
    #Alphie    
    nav_graph, perf_graph, cash_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable, gauges,survival_graph, PN_survival_graph = update_data_and_generate_outputs(portfolio_copy, data, steps, performance_ref, daily_performance_ref, hit_ratio_ref, win_loss_ratio_ref, simulation_dates_df, ptf_id_ref, end_date)

    #Naski
    #nav_graph, perf_graph, cash_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable, survival_graph, PN_survival_graph = update_data_and_generate_outputs(portfolio_copy, data)

    # Generate holdings weights data and figures
    if nb_ptf <= 10:
        holdings_weights_figures = generate_holdings_weights_data_figures(data, nb_ptf)

        # Generate performance attributions data and figures
        performance_attributions_figures = generate_performance_attributions_data_figures(data, nb_ptf)
    else:
        holdings_weights_figures = []
        performance_attributions_figures = []
    
    end = time.time()
    totalTime = end - begin
    print(f"Time taken to generate graphs: {totalTime:.6f} seconds")
        
    print("Simulation 2: pre return")

    return nav_graph, perf_graph, cash_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, holdings_weights_figures, performance_attributions_figures, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable, gauges['totalReturnGauge'], gauges['volatilityGauge'], gauges['IRGauge'], gauges['maxDrawdownGauge'], gauges['hitRatioGauge'], gauges['winLossRatioGauge'], survival_graph, PN_survival_graph, steps, False, stored

# Clientside Callbacks
# Clientside callbacks are used to update the graphs without having to wait for the server to respond
# This is useful for the graphs that are not updated by the simulation loop
# The clientside callbacks are triggered by the data stored in the hidden divs
# The data is updated by the server-side callbacks

app.clientside_callback(
    """
    function(nav_graph) {
        const nav_fig = Object.assign({}, nav_graph, {
                'layout': {
                    ...nav_graph.layout,
                }
        });
        
        return nav_fig
    }

    """,
    Output('nav-graph-container', 'children'),
    Input('clientside-stored-nav', 'data'),
    #State('clientside-interval','n_intervals'),

)

app.clientside_callback(
    """
    function(perf_graph) {
        
        const perf_fig = Object.assign({}, perf_graph, {
                'layout': {
                    ...perf_graph.layout,
                }
        });
        return perf_fig
    }
    """,
    Output('performance-graph-container', 'children'),
    Input('clientside-stored-performance', 'data'),
    #State('clientside-interval','n_intervals'),
),
    

app.clientside_callback(
    """
    function(buy_pipeline_size_graph) {

        const buy_pipeline_size_fig = Object.assign({}, buy_pipeline_size_graph, {
                'layout': {
                    ...buy_pipeline_size_graph.layout,
                }
        });

        return buy_pipeline_size_fig
    }
    """,
    Output('buy-pipeline-size-graph-container', 'children'),
    Input('clientside-stored-buy-pipeline-size', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(sell_pipeline_size_graph) {

        const sell_pipeline_size_fig = Object.assign({}, sell_pipeline_size_graph, {
                'layout': {
                    ...sell_pipeline_size_graph.layout,
                }
        });

        return sell_pipeline_size_fig
    }
    """,
    Output('sell-pipeline-size-graph-container', 'children'),
    Input('clientside-stored-sell-pipeline-size', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(cash_graph) {

        const cash_fig = Object.assign({}, cash_graph, {
                'layout': {
                    ...cash_graph.layout,
                }
        });

        return cash_fig
    }
    """,
    Output('cash-graph-container', 'children'),
    Input('clientside-stored-cash', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(holdings_weights_figures) {
        // Create an array to hold the generated graphs
        const graphs = [];

        // Loop through the data and create dcc.Graph components dynamically
        for (const graphKey in holdings_weights_figures) {
            graphs.push(
                graphKey
            );
            
        }

        return graphs;
    }
    """,
    Output('weights-graph-container', 'children'),
    Input('clientside-stored-weights', 'data'),
   # State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(performance_attributions_figures) {
        // Create an array to hold the generated graphs
        const graphs = [];

        // Loop through the data and create dcc.Graph components dynamically
        for (const graphKey in performance_attributions_figures) {
            graphs.push(
            graphKey
        );
        }

        return graphs;
    }
    """,
    Output('performance-attribution-container', 'children'),
    Input('clientside-stored-performance-attribution', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(hit_ratio_performance_graph) {

        const hit_ratio_performance_fig = Object.assign({}, hit_ratio_performance_graph, {
            'layout': {
                ...hit_ratio_performance_graph.layout,
            }
        });

        return hit_ratio_performance_fig
    }
    """,
    Output('hit-ratio-performance-container', 'children'),
    Input('clientside-stored-hit-ratio-performance', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(win_loss_performance_graph) {

        const win_loss_performance_fig = Object.assign({}, win_loss_performance_graph, {
            'layout': {
                ...win_loss_performance_graph.layout,
            }
        });

        return win_loss_performance_fig
    }
    """,
    Output('win-loss-performance-container', 'children'),
    Input('clientside-stored-win-loss-performance', 'data'),
   # State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(hit_ratio_win_loss_graph) {

        const hit_ratio_win_loss_fig = Object.assign({}, hit_ratio_win_loss_graph, {
            'layout': {
                ...hit_ratio_win_loss_graph.layout,
            }
        });

        return hit_ratio_win_loss_fig
    }
    """,
    Output('hit-ratio-win-loss-container', 'children'),
    Input('clientside-stored-hit-ratio-win-loss', 'data'),
   # State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(dataTable) {

        const dataTable_fig = Object.assign({}, dataTable, {
            'layout': {
                ...dataTable.layout,
            }
        });

        return dataTable_fig
    }
    """,
    Output('data-table-container', 'children'),
    Input('clientside-stored-data-table', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(FalseVar) {

        const FalseVar_fig = false
    
        return FalseVar
    }
    """,
    Output('interval-component', 'disabled'),
    Input('clientside-stored-interval-component', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(totalReturnGauge) {

        const totalReturnGauge_fig = Object.assign({}, totalReturnGauge, {
            'layout': {
                ...totalReturnGauge.layout,
            }
        });

        return totalReturnGauge_fig
    }
    """,
    Output('total-return-gauge', 'children'),
    Input('clientside-stored-total-return', 'data'),
   # State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function (volatilityGauge) {

        const volatilityGauge_fig = Object.assign({}, volatilityGauge, {
            'layout': {
                ...volatilityGauge.layout,
            }
        });

        return volatilityGauge_fig
    }
    """,
    Output('volatility-gauge', 'children'),
    Input('clientside-stored-volatility', 'data'),
   # State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(IRGauge) {
    
        const IRGauge_fig = Object.assign({}, IRGauge, {
            'layout': {
                ...IRGauge.layout,
            }
        });

        return IRGauge_fig
    }
    """,
    Output('IR-gauge', 'children'),
    Input('clientside-stored-IR', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(maxDrawdownGauge) {

        const  maxDrawdownGauge_fig = Object.assign({}, maxDrawdownGauge, {
            'layout': {
                ...maxDrawdownGauge.layout,
            }
        });
    
        return maxDrawdownGauge_fig
    }
    """,
    Output('max-dropdown-gauge', 'children'),
    Input('clientside-stored-max-dropdown', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(hitRatioGauge) {

        const hitRatioGauge_fig = Object.assign({}, hitRatioGauge, {
            'layout': {
                ...hitRatioGauge.layout,
            }
        });

        return hitRatioGauge_fig
    }
    """,
    Output('hit-ratio-gauge', 'children'),
    Input('clientside-stored-hit-ratio', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(winLossRatioGauge) {

        const winLossRatioGauge_fig = Object.assign({}, winLossRatioGauge, {
            'layout': {
                ...winLossRatioGauge.layout,
            }
        });

        return winLossRatioGauge_fig
    }
    """,
    Output('win-loss-ratio-gauge', 'children'),
    Input('clientside-stored-win-loss-ratio', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(stored) {
        
        const stored_fig = stored;

        return stored_fig
    }
    """,
    Output('stored', 'value'),
    Input('clientside-stored-stored', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(steps) {

        const steps_fig = steps;

        return steps_fig
    }
    """,
    Output('steps', 'value'),
    Input('clientside-stored-steps', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(survival_graph) {

        const survival_graph_fig = Object.assign({}, survival_graph, {
            'layout': {
                ...survival_graph.layout,
            }
        });

        return survival_graph_fig
    }
    """,
    Output('line-chart', 'children'),
    Input('clientside-stored-line-chart', 'data'),
    #State('clientside-interval','n_intervals'),
),

app.clientside_callback(
    """
    function(PN_survival_graph) {

        const PN_survival_graph_fig = Object.assign({}, PN_survival_graph, {
            'layout': {
                ...PN_survival_graph.layout,
            }
        });

        return PN_survival_graph_fig
    }
    """,

    Output('line-chart2', 'children'),
    Input('clientside-stored-line-chart2', 'data'),
    #State('clientside-interval','n_intervals'),
)


# Callback for generating buttons of the stored data
@app.callback(
    Output('button-container', 'children'),
    Input('stored', 'value'),
    prevent_initial_call=True
)
def display_simulation_buttons(stored):
    # Check if the simulation_results dictionary is available
    if not simulation_results:
        raise PreventUpdate

    # Get the keys (timestamps) from the simulation_results dictionary
    keys = list(simulation_results.keys())

    # Create a list of buttons based on the keys
    buttons = [html.Button(key, id={'type': 'simulation-button', 'index': key}, n_clicks=0) for key in keys]

    return buttons



# Update the 'steps-display' element with the steps value
@app.callback(
    Output('steps-display', 'children'),
    Input('steps', 'value')
)
def update_steps_display(steps):
    stepnum = str(steps)
    return f'Steps: {stepnum}'

# Style % fillers
@app.callback(
    Output('style-two-percentage', 'value'),
    Input('style-one-percentage', 'value')
)
def style_auto_one(val):
    if (val == None):
        raise PreventUpdate
    return 100 - val

@app.callback(
    Output('style-one-percentage', 'value'),
    Input('style-two-percentage', 'value')
)
def style_auto_two(val):
    if (val == None):
        raise PreventUpdate
    return 100 - val

@app.callback(
    Output('sell-style-two-percentage', 'value'),
    Input('sell-style-one-percentage', 'value')
)
def sell_style_auto_one(val):
    if (val == None):
        raise PreventUpdate
    return 100 - val

@app.callback(
    Output('sell-style-one-percentage', 'value'),
    Input('sell-style-two-percentage', 'value')
)
def sell_style_auto_two(val):
    if (val == None):
        raise PreventUpdate
    return 100 - val

# Run the app
#if __name__ == '__main__':
#    app.run_server(debug=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8150", debug=debug)





