from dash import dcc, html
import dash_bootstrap_components as dbc

def create_layout():
    layout = html.Div(className="container", children=[
    html.H1('Portfolio Performance'),

    html.Div(className="input-section", children=[

        html.Div(className="behavior-section", children=[
            html.Div(className="section-row", children=[
                html.Div(className="input-column", children=[
                    html.H4('Buy Behavior:'),

                    html.Div(className="input-row", children=[
                        html.Label('Min Cash:'),
                        dcc.Input(id='min-cash-input', type='number', value=0.01),
                    ]),

                    html.Div(className="input-row", children=[
                        html.Label('Buy Every Days:'),
                        dcc.Input(id='buy-every-days-input', type='number', value=2),
                    ]),

                    html.Div(className="input-row", children=[
                        html.Label('Max Weight Buy:'),
                        dcc.Input(id='max-weight-buy-input', type='number', value=0.01),
                    ]),

                    html.Div(className="input-row", children=[
                        html.Label('# Max Building Stock:'),
                        dcc.Input(id='nb-max-building-stock-buy', type='number', value=1000),
                    ]),
                ]),

                html.Div(className="input-column", children=[
                    html.H5('Momentum:'),

                    html.Div(className="input-row", children=[
                        html.Label('Number of Days'),
                        dcc.Input(id='style-num-days', type='number', value=20),
                    ]),

                    html.Div(className="input-row", children=[
                        html.Label('Style 1 Min: '),
                        dcc.Input(id='style-one-min', type='number', value=0),
                        html.Label('Style 1 Max: '),
                        dcc.Input(id='style-one-max', type='number', value=0),
                        html.Label('Style 1 %: '),
                        dcc.Input(id='style-one-percentage', type='number', value=50),
                    ]),

                    html.Div(className="input-row", children=[
                        html.Label('Style 2 Min:'),
                        dcc.Input(id='style-two-min', type='number', value=0),
                        html.Label('Style 2 Max:'),
                        dcc.Input(id='style-two-max', type='number', value=0),
                        html.Label('Style 2 %:'),
                        dcc.Input(id='style-two-percentage', type='number', value=50),
                    ])
                ])
            ])
            
        ]),


        html.Div(className="behavior-section", children=[
            html.H4('Scale Up Behavior:'),
            html.Div(className="input-row", children=[
                html.Label('Min Cash:'),
                dcc.Input(id='scale-up-min-cash-input', type='number', value=0.01),
            ]),

            html.Div(className="input-row", children=[
                html.Label('Max Weight Scale Up:'),
                dcc.Input(id='max-weight-scale-up-input', type='number', value=0.01),
            ]),

            html.Div(className="input-row", children=[
                html.Label('Increment:'),
                dcc.Input(id='increment-input', type='number', value=0.025),
            ]),

            html.Div(className="input-row", children=[
                html.Label('# Max Building Stock:'),
                dcc.Input(id='nb-max-building-stock-scale-up', type='number', value=10),
            ]),

            html.Div(className="input-row", children=[
                html.Label('Scale Up Every Days:'),
                dcc.Input(id='scale-up-every-days-input', type='number', value=2),
            ]),

            html.Div(className="input-row", children=[
                html.Label('Weight To Scale Up:'),
                dcc.Input(id='weight-to-scale-up-input', type='number', value=0.0025),
            ]),    
        ]),

        html.Div(className="behavior-section", children=[
            html.H4('Scale Down Behavior:'),
            html.Div(className="input-row", children=[
                html.Label('Weight To Scale Down:'),
                dcc.Input(id='weight-to-scale-down-input', type='number', value=0.0025),
            ]),

            html.Div(className="input-row", children=[
                html.Label('Scale Down Every Days:'),
                dcc.Input(id='scale-down-every-days-input', type='number', value=2),
            ]),
        ]),


        html.Div(className="behavior-section", children=[
            html.Div(className="section-row", children=[
                html.Div(className="input-column", children=[
                    html.H4('Sell Behavior:'),

                    html.Div(className="input-row", children=[
                        html.Label('Max Cash:'),
                        dcc.Input(id='max-cash-input', type='number', value=0.1),
                    ]),

                    html.Div(className="input-row", children=[
                        html.Label('Sell Every Days:'),
                        dcc.Input(id='sell-every-days-input', type='number', value=2),
                    ])
                ]),

                html.Div(className="input-column", children=[
                    html.H5('Momentum:'),

                    html.Div(className="input-row", children=[
                        html.Label('Number of Days'),
                        dcc.Input(id='sell-style-num-days', type='number', value=20),
                    ]),

                    html.Div(className="input-row", children=[
                        html.Label('Style 1 Min: '),
                        dcc.Input(id='sell-style-one-min', type='number', value=0),
                        html.Label('Style 1 Max: '),
                        dcc.Input(id='sell-style-one-max', type='number', value=0),
                        html.Label('Style 1 %: '),
                        dcc.Input(id='sell-style-one-percentage', type='number', value=50),
                    ]),

                    html.Div(className="input-row", children=[
                        html.Label('Style 2 Min:'),
                        dcc.Input(id='sell-style-two-min', type='number', value=0),
                        html.Label('Style 2 Max:'),
                        dcc.Input(id='sell-style-two-max', type='number', value=0),
                        html.Label('Style 2 %:'),
                        dcc.Input(id='sell-style-two-percentage', type='number', value=50),
                    ])
                ])
            ])
            
        ]),

        html.Div(className="input-section", children=[
        html.Div(className="simulation-section", children=[
            html.H4('Simulation:'),
            html.Div(className="input-row", children=[
                html.Label('Simulation Days:'),
                dcc.Input(id='simulation-days-input', type='number', value=20),
            ]),
            html.Div(className="input-row", children=[
                html.Label('Number of Portfolios:'),
                dcc.Input(id='number-portfolios-input', type='number', value=2),
            ]),
            html.Button('Run Simulation', id='run-button', className='run-button')
        ])
    ]),

    ]),

    html.Div(id='button-container'),

    dcc.Input(id='stored', style={'display': 'none'}, type='number', value=0),

    # Add a div element to display the steps value
    html.Div(id='steps-display'),

    html.Div(id='data-table-container'),

    dbc.Row(
            [
                dbc.Col(html.Div(id='total-return-gauge')),
                dbc.Col(html.Div(id='volatility-gauge')),
                dbc.Col(html.Div(id='IR-gauge'))
            ]
        ),

    dbc.Row(
            [
                dbc.Col(html.Div(id='max-dropdown-gauge')),
                dbc.Col(html.Div(id='hit-ratio-gauge')),
                dbc.Col(html.Div(id='win-loss-ratio-gauge'))
            ]
        ),

    dbc.Row(
            [
                dbc.Col(html.Div(id="nav-graph-container", className="graph")),
                dbc.Col(html.Div(id="hit-ratio-performance-container")),
            ]
        ),
    dbc.Row(
        [
            dbc.Col(html.Div(id="performance-graph-container", className="graph")),
            dbc.Col(html.Div(id="win-loss-performance-container")),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(html.Div(id="cash-graph-container", className="graph")),
            dbc.Col(html.Div(id="hit-ratio-win-loss-container")),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(html.Div(id="buy-pipeline-size-graph-container", className="graph")),
            dbc.Col(html.Div(id="sell-pipeline-size-graph-container", className="graph")),
        ]
    ),

    html.Div(id='weights-graph-container'),

    html.Div(id='performance-attribution-container'),

    html.Div(id='line-chart'),
    
    html.Div(id='line-chart2'),

    dcc.Input(id='steps', style={'display': 'none'}, type='number', value=0),

    dcc.Interval(id='interval-component', interval=10000, n_intervals=0, disabled=True)
    
])

    return layout