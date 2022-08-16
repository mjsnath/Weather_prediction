import pandas as pd
import datetime
import keras
import tensorflow as tf
from keras.models import load_model
model = load_model(r'model.h5',compile=True)
pred = pd.read_csv("https://raw.githubusercontent.com/mjsnath/Time_Series/main/ValFeatures.csv")
anom = pd.read_csv("https://raw.githubusercontent.com/mjsnath/Time_Series/main/ValLabels.csv")
import numpy as np
dataset_val = tf.keras.preprocessing.timeseries_dataset_from_array(
    np.array(pred),
     np.array(anom),
    sequence_length=(720/6),
    sampling_rate=6,
    batch_size=256)  
initial = datetime.datetime.now()
last = datetime.datetime.strptime('01.07.2017 00:00:00', '%d.%m.%Y %H:%M:%S')
sub =  initial - last
proc = sub.total_seconds()//3600
import plotly.graph_objects as go
import base64
import plotly.express as px
import dash_table as dt
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash
import dash_bootstrap_components as dbc
import urllib.request
image = urllib.request.urlretrieve("https://image.freepik.com/free-vector/family-wearing-face-masks_52683-38547.jpg", "gender.jpg")
encimg = base64.b64encode(open(image[0], 'rb').read())
wind1 = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
image2 = urllib.request.urlretrieve("https://static01.nyt.com/images/2014/12/11/technology/personaltech/11machin-illo/11machin-illo-articleLarge-v3.jpg?quality=75&auto=webp&disable=upscale", "mask.jpg")
encimg2 = base64.b64encode(open(image2[0], 'rb').read())
wind2 = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

show = html.Div([
    dbc.Row([
               dbc.Col(html.Div(dbc.Alert("** The model is trained on Jena Climate dataset recorded by the Max Planck Institute for Biogeochemistry. The predictions are for next hours from 26.12.2016 12:00.", color="info",style={'height':'90px','font-size':16,'font-style':'italic','fontWeight': 'bold','font-family':"Arial"}))),
               
                dbc.Col(dcc.Input(id="input1", type="text", placeholder="Enter the number of hours to be predicted", style={'marginRight':'10px','width':'450px', 'height':50}),)
              ],className="mt-2"),
        dbc.Row([
            dbc.Col([
                     
                     dbc.Row([dbc.Col(html.Div(id="grp1")), dbc.Col(html.Div([
    html.Img(src='data:image/jpg;base64,{}'.format(encimg2.decode()), 
             style={'height': '300px','height': '500px',"margin-left": "20px","margin-right":'10-px'})]))])], className="mt-2")])])


wind1.layout = html.Div([show])

show1 = html.Div([
    
               
    dbc.Row([dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Pressure (mpbar)",style={
                    'color':'black','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input2", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='warning',style={'height':'13vh','width':500,'margin-left':30})))
              ,
             dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Temperature (deg C)",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input3", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='secondary',style={'height':'13vh','width':500,'margin-left':30}))),
             
            dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Saturation Pressure (mpbar)",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input4", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                    color='secondary',style={'height':'13vh','width':500,'margin-left':30})))],className="mt-2"),
    
    dbc.Row([dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Vapor pressure deficit",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input5", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='secondary',style={'height':'13vh','width':500,'margin-left':30}))),
             
            dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Specific Humidity",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input6", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='secondary',style={'height':'13vh','width':500,'margin-left':30}))),
              
             dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Airtight",style={
                    'color':'white','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input7", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],
                                       color='secondary',style={'height':'13vh','width':500,'margin-left':30}))),
              
             
           
            ],className="mt-2"),
    
    dbc.Row([dbc.Col(html.Div(dbc.Card([dbc.CardHeader("Wind speed (m/s)",style={
                    'color':'black','font-weight': 'bold'
                }),dbc.CardBody(dcc.Input(id="input8", type="text", style={'marginRight':'10px','width':'450px', 'height':50}))],color='warning',style={'height':'13vh','width':500,'margin-left':30})))],className="mt-2"),
    
  
   
      
   
   
    dbc.Row([dbc.Col(dbc.Col(html.Div(id="grp2")))],className="mt-2"),
    


      ])
wind2.layout = html.Div([show1])

import urllib.request
display_image = urllib.request.urlretrieve("https://raw.githubusercontent.com/mllover5901/dat/main/gender-equality.jpg", "gender.jpg")
img_enc = base64.b64encode(open(display_image[0], 'rb').read())
program = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])