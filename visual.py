import numpy as np 
import pandas as pd 
import plotly.express as px

dfv= pd.read_csv('machine-learning/GTCO2.csv')

fig_line = px.line(dfv, x='Year', y='Emissions', title='Emissions Over Time',
                   width=800, height=500, labels={'Emissions': 'Emissions (GtCO₂)'})
fig_line.update_traces(mode='lines+markers', marker=dict(color='red'))
fig_line.update_layout(hoverlabel=dict(bgcolor="white", font_size=16))
fig_line.show()