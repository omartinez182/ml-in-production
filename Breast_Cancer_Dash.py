import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

URL = 'https://raw.githubusercontent.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/master/data.csv'
df = pd.read_csv(URL)

variables = df.columns

app = dash.Dash()

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='xaxis',
            value='perimeter_mean',
            options=[{'label': i.title(), 'value': i} for i in variables]
        )
    ],
    style={'width': '48%', 'display':'inline-block'}),

    html.Div([
        dcc.Dropdown(
            id='yaxis',
            value='concavity_mean',
            options=[{'label': i.title(), 'value': i} for i in variables]
        )
    ],
    style={'width': '48%', 'display':'inline-block'}),
    dcc.Graph(id='grafica')
])

@app.callback(Output('grafica', 'figure'),
              [Input('xaxis', 'value'),
               Input('yaxis', 'value')]
)
def update_graph(xaxis_name, yaxis_name):
    df.diagnosis[df.diagnosis == 'M'] = 1
    df.diagnosis[df.diagnosis == 'B'] = 0
    data = [go.Scatter(
        x=df[xaxis_name],
        y=df[yaxis_name],
        mode='markers',
        text=df['id'],
       marker={
            'size':15,
            'color':df['diagnosis'],
            'opacity':0.5,
            'line':{'width':0.5, 'color':'white'}
        }
    )]

    layout = go.Layout(
        title='Grafica de mtars',
        xaxis={'title': xaxis_name.title()},
        yaxis={'title': yaxis_name.title()},
        hovermode='closest'
    )

    return {'data': data, 'layout':layout}


if __name__ == '__main__':
    app.run_server(debug=True)