from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from kafka import KafkaConsumer
import plotly.express as px
import pandas as pd
import threading
import json

app = Dash(__name__)
app.title = "Real-Time Predictions"

consumer = KafkaConsumer(
    "predicted_data",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

data = []

def consume_data():
    global data
    for message in consumer:
        data.append(message.value)
        if len(data) > 100:
            data = data[-100:]

thread = threading.Thread(target=consume_data, daemon=True)
thread.start()

app.layout = html.Div([
   
    html.Div([
        html.Div([
            dcc.Graph(id="cluster-graph-1"),
        ], style={"width": "33%", "display": "inline-block"}),

        html.Div([
            dcc.Graph(id="cluster-graph-2"),
        ], style={"width": "33%", "display": "inline-block"}),

        html.Div([
            dcc.Graph(id="cluster-graph-3"),
        ], style={"width": "33%", "display": "inline-block"}),
    ]),
    dcc.Interval(id="interval", interval=1000, n_intervals=0),  
])

@app.callback(
    [
        Output("cluster-graph-1", "figure"),
        Output("cluster-graph-2", "figure"),
        Output("cluster-graph-3", "figure"),
    ],
    [Input("interval", "n_intervals")]
)
def update_graphs(n):
    global data
    if not data:
        return (
            px.scatter(title="Waiting for data..."),
            px.scatter(title="Waiting for data..."),
            px.scatter(title="Waiting for data...")
        )

    df = pd.DataFrame(data)

    fig1 = px.scatter(
        df,
        x="amount",
        y="duration",
        color="prediction",
        title="Cluster by Amount and Duration",
        labels={"amount": "Loan Amount", "duration": "Loan Duration (months)", "prediction": "Cluster"}
    )

    fig2 = px.scatter(
        df,
        x="amount",
        y="age",
        color="prediction",
        title="Cluster by Amount and Age",
        labels={"amount": "Loan Amount", "age": "Customer Age", "prediction": "Cluster"}
    )

    fig3 = px.scatter(
        df,
        x="duration",
        y="age",
        color="prediction",
        title="Cluster by Duration and Age",
        labels={"duration": "Loan Duration (months)", "age": "Customer Age", "prediction": "Cluster"}
    )

    return fig1, fig2, fig3

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
