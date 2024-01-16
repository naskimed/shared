import plotly.graph_objects as go
from dash import dcc, dash_table
from datetime import datetime
import pandas as pd
import math
from statistics import mean, median
from dash_app import Portfolio, PortfolioManager
import plotly.express as px
import time
from lifelines import KaplanMeierFitter
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

path=""# "C:/CloudStation/Hanon/CloudStation/Prototypes/ABM/"
filename=path+ "for_alfie.xlsx"


print("opening file", filename)
xl = pd.ExcelFile(filename)

sheets_name = "reference_bets"
reference_bets = xl.parse(sheets_name)
xl.close()


# Survival graph
def generate_survival_graph_ref(dataList, df):

    fig = go.Figure()
    df1 = df
    for i, df in enumerate(dataList):
        fig.add_trace(go.Scatter(x=df.index, y=df['All Proba'], mode='lines', name=f'Portfolio {i}'))
    fig.add_trace(go.Scatter(x=df1.index, y=df1['All Proba'], mode='lines', name='Reference Portfolio', line=dict(width=3)))

    fig.update_layout(title='Survival Graph', title_x=0.5, xaxis_title='Timeline (days)', yaxis_title='Percent (%)')

    return fig

# Survival graph positive and negative
def generate_surv_positive_negative_graph(posData,negData,posRef,negRef):

    fig = make_subplots(rows=1, cols=2,subplot_titles=("POSITIVE", "NEGATIVE"))
    #Positive Graph
    for i, df in enumerate(posData):
        fig.add_trace(go.Scatter(x=df.index, y=df['All Proba'], mode='lines', name=f'+ Portfolio {i}'),row=1,col=1)
    fig.add_trace(go.Scatter(x=posRef.index, y=posRef['All Proba'], mode='lines', name='Positive Reference Portfolio', line=dict(width=3)),row=1,col=1)
    #Negative Graph
    for i, df in enumerate(negData):
        fig.add_trace(go.Scatter(x=df.index, y=df['All Proba'], mode='lines', name=f'- Portfolio {i}'),row=1,col=2)
    fig.add_trace(go.Scatter(x=negRef.index, y=negRef['All Proba'], mode='lines', name='Negative Reference Portfolio', line=dict(width=3)),row=1,col=2)

    fig.update_xaxes(title_text="Timeline (days)", row=1, col=1)
    fig.update_xaxes(title_text="Timeline (days)", row=1, col=2)
    fig.update_yaxes(title_text="Percent (%)", row=1, col=1)
    fig.update_yaxes(title_text="Percent (%)", row=1, col=2)
    
    fig.update_layout(title_text ='Survival Graph', title_x=0.5)

    return fig



# Survival Function 
def survival(data):
    data['age'] = (pd.to_datetime(data['end_date']) - pd.to_datetime(data['start_date'])).dt.days

    data.drop(data.loc[data['next_decision'] == 'Still Alive'].index, axis=0, inplace=True)
    data.drop(data.loc[data['age'] == 0].index, axis=0, inplace=True)
    
    if data.empty:
        data['age'] = [0] * 50
        data['All Proba'] = [0] * 50
        data.loc[:, 'unit'] = 1
        return (data)

    data.loc[:, 'unit'] = 1
    kmf = KaplanMeierFitter()
    kmf.fit(pd.to_numeric(data['age']), pd.to_numeric(data['unit']), label='All Proba')

    return kmf.survival_function_
    

# Cash, buy/sell, performance, nav graphs
def generate_graphs(nav_data, perf_ref_data, buy_pipeline_data, sell_pipeline_data, cash_data, df):
    navFig = go.Figure()
    perfFig = go.Figure()
    df1 = df.set_index('date').loc[perf_ref_data.index]
    buyFig = go.Figure()
    sellFig = go.Figure()
    cashFig = go.Figure()
    for p in range(len(nav_data.columns)) :
        navFig.add_trace(go.Scatter(x=nav_data.index, y=nav_data[p], name=f'Portfolio {p}'))
        perfFig.add_trace(go.Scatter(x=perf_ref_data.index, y=perf_ref_data[p], name=f'Portfolio {p}'))
        buyFig.add_trace(go.Scatter(x=buy_pipeline_data.index, y=buy_pipeline_data[p], name=f'Portfolio {p}'))
        sellFig.add_trace(go.Scatter(x=sell_pipeline_data.index, y=sell_pipeline_data[p], name=f'Portfolio {p}'))
        cashFig.add_trace(go.Scatter(x=cash_data.index, y=cash_data[p], name=f'Portfolio {p}'))
    navFig.update_layout(title='Portfolio NAV', title_x=0.5)
    perfFig.add_trace(go.Scatter(x=df1.index, y=df1.performance, name='Reference Portfolio', line=dict(width=3)))
    perfFig.update_layout(title='Portfolio Performance', title_x=0.5)
    buyFig.update_layout(title='Buy Pipeline Size', title_x=0.5)
    sellFig.update_layout(title='Sell Pipeline Size', title_x=0.5)
    cashFig.update_layout(title='Cash Weights', title_x=0.5)
    return navFig, perfFig, buyFig, sellFig, cashFig


# Function to generate NAV graph
def generate_nav_graph(data):
    fig = go.Figure()
    for p in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[p], name=f'Portfolio {p}'))
    fig.update_layout(title='Portfolio NAV', title_x=0.5)
    return fig

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

def generate_gauge_chart(percentile, text):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentile,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': text},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'blue'},
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 100], 'color': 'gray'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': percentile
            }
        }
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0, pad=0, autoexpand=False),
    )
    return fig

def Percentile(data, number):
    return int(len([num for num in data if num < number]) / len(data) * 100)


def Volatility(data):
    return data.std()*math.sqrt(255)

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

def calculateMetrics(data, percentBoolean):
    avg = mean(data)
    med = median(data)
    minimum = min(data)
    maximum = max(data)
    if percentBoolean:
        return toPercentAndRound([minimum, med, avg, maximum], True)
    return toPercentAndRound([minimum, med, avg, maximum], False)

def toPercentAndRound(nums, percentBoolean):
    if percentBoolean:
        for i in range(len(nums)):
            nums[i] = round(nums[i] * 100, 2)
    else:
        for i in range(len(nums)):
            nums[i] = round(nums[i], 2)
    return nums

def formatPercentages(nums):
    for i in range(len(nums)):
            nums[i] = f"{nums[i]}%"
    return nums

def generateTable(hitRatiodata, winLossRatioData, gg_perf, totalReturn, referenceRow, percentileRow):
    tableData = {
        "Metric": ["Min", "Median", "Average", "Max"],
        "Total Return": formatPercentages(calculateMetrics(totalReturn, True)),
        "Volatility": formatPercentages(calculateMetrics(gg_perf['Volatility'], True)),
        "IR": formatPercentages(calculateMetrics(gg_perf['IR'], True)),
        "Max Drawdown": formatPercentages(calculateMetrics(gg_perf['MaxDrawdown'], True)),
        "Hit Ratio": calculateMetrics(hitRatiodata, False),
        "Win Loss Ratio": calculateMetrics(winLossRatioData, False),
    }
    df = pd.DataFrame(tableData)
    df1 = pd.DataFrame([referenceRow], columns=df.columns)
    df2 = pd.DataFrame([percentileRow], columns=df.columns)
    df = pd.concat([df, df1, df2]) 

    return df

    # Function shared between the two callbacks to get the data and update the graphs
def update_data_and_generate_outputs(portfolio_copy, data, steps, performance_ref, daily_performance_ref, hit_ratio_ref, win_loss_ratio_ref, simulation_dates_df, ptf_id_ref, end_date):
    
    nav_data = pd.pivot_table(data[~data['date'].isna()], index=['date'], columns=data.index.get_level_values('AgentID'), values='NAV')
    perf_data = data[~data['date'].isna()]
    perf = pd.pivot_table(perf_data, index=['date'], columns=perf_data.index.get_level_values('AgentID'), values='performance')
    gg_perf = perf.diff(-1).agg(['count','sum','std', IR, MaxDrawdown, Volatility]).transpose()

    most_recent_date = perf_data['date'].max()
    most_recent_perf = perf[perf.index == most_recent_date]
    totalReturn = most_recent_perf.iloc[-1]
    survivalData = []
    positive_survival_data = []
    negative_survival_data = []
    
    #Reference Portfolio
    referenceRow = {
        "Metric": "Reference Portfolio",
        "Total Return": f"{round(performance_ref[steps - 1] * 100, 2)}%",
        "Volatility": f"{round(Volatility(daily_performance_ref[0:steps]) * 100, 2)}%",
        "IR": f"{round(IR(daily_performance_ref[0:steps]) * 100, 2)}%",
        "Max Drawdown": f"{round(MaxDrawdown(daily_performance_ref[0:steps]) * 100, 2)}%",
        "Hit Ratio": round(hit_ratio_ref[steps - 1], 2),
        "Win Loss Ratio": round(win_loss_ratio_ref[steps - 1], 2),
    }

    all_ptfs = [obj for obj in portfolio_copy.schedule.agents if ((isinstance(obj, Portfolio)))]
    cols=["pft_id","number_bets", 'nav', "performance", "hit_ratio","win_loss_ratio", "nb_stop_loss"]
    behave=pd.DataFrame([],columns=cols)
    for ptf in all_ptfs:
        
        bets=pd.DataFrame.from_dict(ptf.bets['closed'])
        still_active_bets=pd.DataFrame([{"security_id":sec,
                                                    "start_date": ptf.bets['active'][sec],
                                                    "end_date": end_date, 
                                                    "performance":ptf.performance_attribution_active_bets[sec]
                                                    } for sec in ptf.holdings.keys()])
        bets=bets.append(still_active_bets, ignore_index=True)
        bets = bets.query("security_id != 'Cash'")
        hit_ratio=len(bets[bets['performance']>0])/len(bets)
        win_loss_ratio=-bets[bets['performance']>0]['performance'].mean() /bets[bets['performance']<0]['performance'].mean()
        tab=[ptf.unique_id,len(bets),ptf.nav, ptf.performance,  hit_ratio,win_loss_ratio, ptf.nb_stop_loss]
        behave=behave.append(pd.DataFrame([tab],columns=cols), ignore_index=True)
        #print(ptf.buy_pipeline)

    all_ptfs = [obj for obj in portfolio_copy.schedule.agents if ((isinstance(obj, PortfolioManager)))]
    cols=["pft_id","number_bets", 'nav', "performance", "hit_ratio","win_loss_ratio"]
    behave=pd.DataFrame([],columns=cols)
    cash_weights=pd.DataFrame()
    buy_pipeline_data=pd.DataFrame()
    sell_pipeline_data=pd.DataFrame()
    hitRatioData = []
    winLossRatioData = []

    i=0
    max_days = 0
    begin = time.time()
    timespent = 0
    for ptf in all_ptfs:
        bets=ptf.bets['closed'].copy()
        still_active_bets=ptf.bets['active']
        still_active_bets['security_id']=ptf.bets['active'].index
        still_active_bets['end_date']=end_date
        still_active_bets['next_decision']="Still Alive"
        bets=bets._append(still_active_bets, ignore_index=True)
        bets = bets.query("security_id != 'Cash'")
        hit_ratio=len(bets[bets['performance']>0])/len(bets)
        win_loss_ratio=-bets[bets['performance']>0]['performance'].mean() /bets[bets['performance']<0]['performance'].mean()
        tab=[ptf.unique_id,len(bets),ptf.nav, ptf.performance,  hit_ratio,win_loss_ratio]
        behave=behave._append(pd.DataFrame([tab],columns=cols), ignore_index=True)

        variable = 'performance_attribution_active_bets'
        wghts_hist = pd.DataFrame.from_dict(list(data[data.index.get_level_values('AgentID') == i][variable]))
        wghts_hist['date'] = data[data.index.get_level_values('AgentID') == i]['date'].values
        wghts_hist = wghts_hist.set_index('date')
        cash_weights[i]=wghts_hist['Cash']
        
        variable = 'buy_pipeline'
        buy_pipeline = pd.DataFrame.from_dict(list(data[data.index.get_level_values('AgentID') == i][variable]))
        buy_pipeline['date'] = data[data.index.get_level_values('AgentID') == i]['date'].values
        buy_pipeline = buy_pipeline.set_index('date')
        buy_pipeline_data[i] = buy_pipeline.apply(lambda row: len(row.dropna()), axis=1)

        variable = 'sell_pipeline'
        sell_pipeline = pd.DataFrame.from_dict(list(data[data.index.get_level_values('AgentID') == i][variable]))
        sell_pipeline['date'] = data[data.index.get_level_values('AgentID') == i]['date'].values
        sell_pipeline = sell_pipeline.set_index('date')
        sell_pipeline_data[i] = sell_pipeline.apply(lambda row: len(row.dropna()), axis=1)

        winLossRatioData.append(win_loss_ratio)
        hitRatioData.append(hit_ratio)

        begin1 = time.time()
        #survival parameters
        survival_data = survival(bets)
        survivalData.append(survival_data)

        end1 = time.time()
        #For the refrence portfolio
        if max(survival_data.index) > max_days:
            max_days = max(survival_data.index)

        #positive survival parameters
        PsurvivalData = survival(bets[bets['performance'] > 0])
        NsurvivalData = survival(bets[bets['performance'] < 0])
        positive_survival_data.append(PsurvivalData)
        negative_survival_data.append(NsurvivalData)
        
        
        timespent+= end1 - begin1
        i += 1

    end = time.time()
    execution_time2 = end - begin
    


    #Bets refrerence
    ref_bets = survival(reference_bets)
    ref_bets = ref_bets[ref_bets.index <= max_days]
    positive_ref = survival(reference_bets[reference_bets['performance'] > 0])
    positive_ref = positive_ref[positive_ref.index <= max_days]
    negative_ref = survival(reference_bets[reference_bets['performance'] < 0])
    negative_ref = negative_ref[negative_ref.index <= max_days]


    tab = [ptf_id_ref[steps - 1], performance_ref[steps - 1], hit_ratio_ref[steps - 1], win_loss_ratio_ref[steps - 1]]
    cols=["pft_id", "performance", "hit_ratio","win_loss_ratio"]
    behave=behave._append(pd.DataFrame([tab],columns=cols), ignore_index=True)

    start_time = time.time()
    nav_graph_figure, perf_graph_figure, buy_pipeline_size_graph_figure, sell_pipeline_size_graph_figure, cash_graph_figure = generate_graphs(nav_data, perf, buy_pipeline_data.dropna(), sell_pipeline_data.dropna(), cash_weights, simulation_dates_df)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time spent: {timespent:.6f} seconds")
    print(f"Second loop: {execution_time2:.6f} seconds")
    print(f"Single function execution time: {execution_time:.6f} seconds")

    
    start_time1 = time.time()
    nav_graph = dcc.Graph(figure=generate_nav_graph(nav_data))
    perf_graph = dcc.Graph(figure=generate_performance_graph_ref(perf, simulation_dates_df))
    cash_graph = dcc.Graph(figure=generate_cash_graph(cash_weights))
    buy_pipeline_size_graph = dcc.Graph(figure=generate_buy_pipeline_size_graph(buy_pipeline_data.dropna()))
    sell_pipeline_size_graph = dcc.Graph(figure=generate_sell_pipeline_size_graph(sell_pipeline_data.dropna()))
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1
    print(f"Multiple function execution time: {execution_time1:.6f} seconds")
    
    nav_graph = dcc.Graph(figure=nav_graph_figure)
    perf_graph = dcc.Graph(figure=perf_graph_figure)
    cash_graph = dcc.Graph(figure=cash_graph_figure)
    buy_pipeline_size_graph = dcc.Graph(figure=buy_pipeline_size_graph_figure)
    sell_pipeline_size_graph = dcc.Graph(figure=sell_pipeline_size_graph_figure)
    hit_ratio_performance_graph = dcc.Graph(figure=generate_hit_ratio_performance_scatter(behave))
    win_loss_performance_graph = dcc.Graph(figure=generate_win_loss_ratio_performance_scatter(behave))
    hit_ratio_win_loss_graph = dcc.Graph(figure=generate_hit_ratio_win_loss_ratio_scatter(behave))
    survival_graph = dcc.Graph(figure=generate_survival_graph_ref(survivalData,ref_bets))
    survival_positive_negative = dcc.Graph(figure= generate_surv_positive_negative_graph(positive_survival_data, negative_survival_data,positive_ref,negative_ref))

    #Percentiles
    percentileData = {
        "Metric": "Percentile",
        "Total Return": Percentile(totalReturn, performance_ref[steps - 1]),
        "Volatility": Percentile(gg_perf['Volatility'], Volatility(daily_performance_ref[0:steps])),
        "IR": Percentile(gg_perf['IR'], IR(daily_performance_ref[0:steps])),
        "Max Drawdown": Percentile(gg_perf['MaxDrawdown'], MaxDrawdown(daily_performance_ref[0:steps])),
        "Hit Ratio": Percentile(hitRatioData, hit_ratio_ref[steps - 1]),
        "Win Loss Ratio": Percentile(winLossRatioData, hit_ratio_ref[steps - 1]),
    }

    gauges = {
        "totalReturnGauge": dcc.Graph(figure=generate_gauge_chart(percentileData['Total Return'], "Total Return Percentile")),
        "volatilityGauge": dcc.Graph(figure=generate_gauge_chart(percentileData['Volatility'], "Volatility Percentile")),
        "IRGauge": dcc.Graph(figure=generate_gauge_chart(percentileData['IR'], "IR Percentile")),
        "maxDrawdownGauge": dcc.Graph(figure=generate_gauge_chart(percentileData['Max Drawdown'], "Max Drawdown Percentile")),
        "hitRatioGauge": dcc.Graph(figure=generate_gauge_chart(percentileData['Hit Ratio'], "Hit Ratio Percentile")),
        "winLossRatioGauge": dcc.Graph(figure=generate_gauge_chart(percentileData['Win Loss Ratio'], "Win Loss Percentile")),
    }

    

    dfTableData = generateTable(hitRatioData, winLossRatioData, gg_perf, totalReturn, referenceRow, percentileData)
    dataTable = dash_table.DataTable(
        data=dfTableData.to_dict('records'),
        columns=[{"name": i, "id": i} for i in dfTableData.columns],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'font_family': 'Arial',
            'font_size': '14px',
            'text_align': 'left',
            'padding': '5px'
        },
        style_table={
            'overflowX': 'auto',
            'border': '1px solid grey',
            'border_radius': '5px',
            'margin': 'auto'
        },
        style_header={
            'background': 'lightgrey',
            'fontWeight': 'bold',
            'font_family': 'Arial',
            'font_size': '16px'
        },
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'center'
            } for c in dfTableData.columns
        ]
    )
    
    # Return the relevant data and outputs
    return nav_graph, perf_graph, cash_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable, gauges,survival_graph, survival_positive_negative 





 
    nav_data = pd.pivot_table(data[~data['date'].isna()], index=['date'], columns=data.index.get_level_values('AgentID'), values='NAV')
    perf_data = data[~data['date'].isna()]
    perf = pd.pivot_table(perf_data, index=['date'], columns=perf_data.index.get_level_values('AgentID'), values='performance')


    all_ptfs = [obj for obj in portfolio_copy.schedule.agents if ((isinstance(obj, Portfolio)))]
    cols=["pft_id","number_bets", 'nav', "performance", "hit_ratio","win_loss_ratio", "nb_stop_loss"]
    behave=pd.DataFrame([],columns=cols)
    for ptf in all_ptfs:
        
        bets=pd.DataFrame.from_dict(ptf.bets['closed'])
        still_active_bets=pd.DataFrame([{"security_id":sec,
                                                    "start_date": ptf.bets['active'][sec],
                                                    "end_date": end_date, 
                                                    "performance":ptf.performance_attribution_active_bets[sec]
                                                    } for sec in ptf.holdings.keys()])
        bets=bets.append(still_active_bets, ignore_index=True)
        bets = bets.query("security_id != 'Cash'")
        hit_ratio=len(bets[bets['performance']>0])/len(bets)
        win_loss_ratio=-bets[bets['performance']>0]['performance'].mean() /bets[bets['performance']<0]['performance'].mean()
        tab=[ptf.unique_id,len(bets),ptf.nav, ptf.performance,  hit_ratio,win_loss_ratio, ptf.nb_stop_loss]
        behave=behave.append(pd.DataFrame([tab],columns=cols), ignore_index=True)
        #print(ptf.buy_pipeline)

    all_ptfs = [obj for obj in portfolio_copy.schedule.agents if ((isinstance(obj, PortfolioManager)))]
    cols=["pft_id","number_bets", 'nav', "performance", "hit_ratio","win_loss_ratio"]
    behave=pd.DataFrame([],columns=cols)
    cash_weights=pd.DataFrame()
    buy_pipeline_data=pd.DataFrame()
    sell_pipeline_data=pd.DataFrame()
    informationRatioData = []
    totalReturnData = []
    hitRatioData = []
    winLossRatioData = []
    volatilityData = []
    maxDrawdownData = []
    survivalData = []
    positive_survival_data = []
    negative_survival_data = []

    i=0
    max_days = 0
    for ptf in all_ptfs:
        bets=ptf.bets['closed'].copy()
        still_active_bets=ptf.bets['active']
        still_active_bets['security_id']=ptf.bets['active'].index
        still_active_bets['end_date']=end_date
        still_active_bets['next_decision']="Still Alive"
        bets=bets._append(still_active_bets, ignore_index=True)
        bets = bets.query("security_id != 'Cash'")
        hit_ratio=len(bets[bets['performance']>0])/len(bets)
        win_loss_ratio=-bets[bets['performance']>0]['performance'].mean() /bets[bets['performance']<0]['performance'].mean()
        tab=[ptf.unique_id,len(bets),ptf.nav, ptf.performance,  hit_ratio,win_loss_ratio]
        behave=behave._append(pd.DataFrame([tab],columns=cols), ignore_index=True)

        variable = 'performance_attribution_active_bets'
        wghts_hist = pd.DataFrame.from_dict(list(data[data.index.get_level_values('AgentID') == i][variable]))
        wghts_hist['date'] = data[data.index.get_level_values('AgentID') == i]['date'].values
        wghts_hist = wghts_hist.set_index('date')
        cash_weights[i]=wghts_hist['Cash']
        
        variable = 'buy_pipeline'
        buy_pipeline = pd.DataFrame.from_dict(list(data[data.index.get_level_values('AgentID') == i][variable]))
        buy_pipeline['date'] = data[data.index.get_level_values('AgentID') == i]['date'].values
        buy_pipeline = buy_pipeline.set_index('date')
        buy_pipeline_data[i] = buy_pipeline.apply(lambda row: len(row.dropna()), axis=1)

        variable = 'sell_pipeline'
        sell_pipeline = pd.DataFrame.from_dict(list(data[data.index.get_level_values('AgentID') == i][variable]))
        sell_pipeline['date'] = data[data.index.get_level_values('AgentID') == i]['date'].values
        sell_pipeline = sell_pipeline.set_index('date')
        sell_pipeline_data[i] = sell_pipeline.apply(lambda row: len(row.dropna()), axis=1)

        winLossRatioData.append(win_loss_ratio)
        hitRatioData.append(hit_ratio)

        #survival parameters
        survival_data = survival(bets)
        survivalData.append(survival_data)

        #For the refrence portfolio
        if max(survival_data.index) > max_days:
            max_days = max(survival_data.index)

        #positive survival parameters
        PsurvivalData = survival(bets[bets['performance'] > 0])
        NsurvivalData = survival(bets[bets['performance'] < 0])
        positive_survival_data.append(PsurvivalData)
        negative_survival_data.append(NsurvivalData)


        i += 1

    #Bets refrerence
    ref_bets = survival(reference_bets)
    ref_bets = ref_bets[ref_bets.index <= max_days]
    positive_ref = survival(reference_bets[reference_bets['performance'] > 0])
    positive_ref = positive_ref[positive_ref.index <= max_days]
    negative_ref = survival(reference_bets[reference_bets['performance'] < 0])
    negative_ref = negative_ref[negative_ref.index <= max_days]

    nav_graph = dcc.Graph(figure=generate_nav_graph(nav_data))
    perf_graph = dcc.Graph(figure=generate_performance_graph_ref(perf, simulation_dates_df))
    cash_graph = dcc.Graph(figure=generate_cash_graph(cash_weights))
    buy_pipeline_size_graph = dcc.Graph(figure=generate_buy_pipeline_size_graph(buy_pipeline_data.dropna()))
    sell_pipeline_size_graph = dcc.Graph(figure=generate_sell_pipeline_size_graph(sell_pipeline_data.dropna()))
    hit_ratio_performance_graph = dcc.Graph(figure=generate_hit_ratio_performance_scatter(behave))
    win_loss_performance_graph = dcc.Graph(figure=generate_win_loss_ratio_performance_scatter(behave))
    hit_ratio_win_loss_graph = dcc.Graph(figure=generate_hit_ratio_win_loss_ratio_scatter(behave))
    survival_graph = dcc.Graph(figure=generate_survival_graph_ref(survivalData,ref_bets))
    survival_positive_negative = dcc.Graph(figure= generate_surv_positive_negative_graph(positive_survival_data, negative_survival_data,positive_ref,negative_ref))


    ## dfTableData = generateTable(data, totalReturnData, volatilityData, informationRatioData, maxDrawdownData, hitRatioData, winLossRatioData)
    # dataTable = dash_table.DataTable(dfTableData.to_dict('records'), [{"name": i, "id": i} for i in dfTableData.columns]),
    dfTableData = generateTable(hitRatioData, winLossRatioData)
    dataTable = dash_table.DataTable(dfTableData.to_dict('records'), [{"name": i, "id": i} for i in dfTableData.columns]),
    
    # Return the relevant data and outputs
    return nav_graph, perf_graph, cash_graph, buy_pipeline_size_graph, sell_pipeline_size_graph, hit_ratio_performance_graph, win_loss_performance_graph, hit_ratio_win_loss_graph, dataTable, survival_graph, survival_positive_negative 
