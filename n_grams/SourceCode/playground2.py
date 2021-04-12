import plotly.express as px
import pandas as pd

sample = { "a" : 10, "b": 20}
df = pd.Series(sample).to_frame()

fig = px.bar(df, width=960, height=540)

fig.write_image("./bla.png")