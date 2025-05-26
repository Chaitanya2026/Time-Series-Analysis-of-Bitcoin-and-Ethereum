import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import joblib

# Load data
btc_df = pd.read_csv('../data/btc_with_sentiment.csv')
eth_df = pd.read_csv('../data/eth_with_sentiment.csv')
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
eth_df['Date'] = pd.to_datetime(eth_df['Date'])

btc_model = joblib.load('../models/rf_btc_model.pkl')
eth_model = joblib.load('../models/rf_eth_model.pkl')

features_btc = ['Open_BTC-USD', 'High_BTC-USD', 'Low_BTC-USD', 'Close_BTC-USD',
                'Volume_BTC-USD', 'Volatility', 'MA_7', 'MA_30', 'Daily_Return', 'Sentiment_Score']
features_eth = ['ETH-USD_Open', 'ETH-USD_High', 'ETH-USD_Low', 'ETH-USD_Close',
                'ETH-USD_Volume', 'Volatility', 'MA_7', 'MA_30', 'Daily_Return', 'Sentiment_Score']

app = dash.Dash(__name__)
app.title = "Crypto Sentiment Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Crypto Sentiment Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Select Cryptocurrency"),
            dcc.Dropdown(
                id='coin-dropdown',
                options=[
                    {'label': 'Bitcoin (BTC)', 'value': 'BTC'},
                    {'label': 'Ethereum (ETH)', 'value': 'ETH'}
                ],
                value='BTC',
                clearable=False,
                style={'width': '200px'}
            )
        ]),
        
        html.Div([
            html.Label("Sentiment Score Range"),
            dcc.RangeSlider(
                id='sentiment-slider',
                min=-1,
                max=1,
                step=0.1,
                value=[-1, 1],
                marks={i: f'{i:.1f}' for i in np.arange(-1, 1.1, 0.5)},
                tooltip={"always_visible": True}
            )
        ], style={'marginTop': '20px', 'width': '60%'}),
        
        html.Div([
            html.Label("Select Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=btc_df['Date'].min(),
                end_date=btc_df['Date'].max()
            )
        ], style={'marginTop': '20px'})
    ], style={'padding': '20px'}),
    
    dcc.Store(id='current-index', data=0),
    dcc.Graph(id='live-price-chart'),
    html.Div(id='forecast-output', style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '18px'}),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0, max_intervals=-1)
])

# 🔮 Simplified forecast message with trend + sentiment
def get_forecast_message(pred, last_price, sentiment_recent):
    diff = pred - last_price
    rel_change = diff / last_price

    if rel_change > 0.01:
        trend = "📈 **Trend:** Likely to go high"
    elif rel_change < -0.01:
        trend = "📉 **Trend:** Likely to go low"
    else:
        trend = "⚖️ **Trend:** Likely to stay stable"

    sentiment_trend = sentiment_recent.diff().mean()
    if sentiment_trend > 0.01:
        sentiment_msg = "📊 Sentiment improving. Possible positive momentum."
    elif sentiment_trend < -0.01:
        sentiment_msg = "😟 Sentiment weakening. Caution advised."
    else:
        sentiment_msg = "🤔 Sentiment steady. Market may consolidate."

    return f"{trend}\n\n{sentiment_msg}"

@app.callback(
    [Output('live-price-chart', 'figure'),
     Output('forecast-output', 'children'),
     Output('current-index', 'data')],
    [Input('coin-dropdown', 'value'),
     Input('sentiment-slider', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('interval-component', 'n_intervals')],
    [State('current-index', 'data')]
)
def update_dashboard(coin, sentiment_range, start_date, end_date, n_intervals, current_idx):
    df = btc_df.copy() if coin == 'BTC' else eth_df.copy()
    model = btc_model if coin == 'BTC' else eth_model
    features = features_btc if coin == 'BTC' else features_eth
    close_col = 'Close_BTC-USD' if coin == 'BTC' else 'ETH-USD_Close'

    df = df[(df['Sentiment_Score'] >= sentiment_range[0]) &
            (df['Sentiment_Score'] <= sentiment_range[1])]
    df = df[(df['Date'] >= pd.to_datetime(start_date)) &
            (df['Date'] <= pd.to_datetime(end_date))]
    df.sort_values('Date', inplace=True)

    if df.empty or len(df) < 10:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig, "No data to predict.", 0

    max_idx = len(df) - 1
    new_idx = current_idx + 1 if current_idx < max_idx else max_idx
    df_slice = df.iloc[:new_idx + 1]
    X = df_slice[features].dropna()

    if X.empty:
        return dash.no_update, "Insufficient data to make forecast.", new_idx

    last_features = X.iloc[-1].values.reshape(1, -1)
    last_close = df_slice[close_col].iloc[-1]

    try:
        pred = model.predict(last_features)[0]
    except Exception:
        pred = last_close

    recent_sentiment = df_slice['Sentiment_Score'].tail(5)
    forecast_msg = get_forecast_message(pred, last_close, recent_sentiment)

    trace_price = go.Scatter(
        x=df_slice['Date'],
        y=df_slice[close_col],
        mode='lines+markers',
        name='Price'
    )
    trace_sentiment = go.Scatter(
        x=df_slice['Date'],
        y=df_slice['Sentiment_Score'],
        mode='lines',
        name='Sentiment',
        yaxis='y2'
    )

    layout = go.Layout(
        title=f'{coin} Price & Sentiment',
        yaxis=dict(title='Price (USD)'),
        yaxis2=dict(title='Sentiment Score', overlaying='y', side='right', range=[-1, 1]),
        legend=dict(x=0, y=1),
        margin=dict(l=40, r=40, t=50, b=50)
    )

    fig = go.Figure(data=[trace_price, trace_sentiment], layout=layout)
    return fig, forecast_msg, new_idx

if __name__ == '__main__':
    app.run(debug=True)
