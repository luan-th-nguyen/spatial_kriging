import pandas as pd
import numpy as np
#from scipy import interpolate as _interp #import griddata
import scipy
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.spatial_kriging import SpatialKriging

def main_variogram(st):
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    st.header('Known data import')
    uploaded_file = st.file_uploader("Choose a CSV file of known data points")
    if uploaded_file is not None:
        # read data
        col1, col2 = st.columns([3, 7])
        data = pd.read_csv(uploaded_file)
        data = data.rename(columns=lambda c: c.strip()) # remove trailing whitespaces in column names
        col1.write(data)

        # visualize data
        trace = go.Scatter3d(
           x = data['X'], y = data['Y'], z = data['Z'], text = data['Name'], mode='markers+text', marker = dict(
              size = 5.0,
              color = data['Z'], # set color to an array/list of desired values
              colorscale = 'Viridis'
              )
           )
        layout = go.Layout(title = '3D Scatter plot')
        fig = go.Figure(data = [trace], layout = layout)
        col2.plotly_chart(fig, use_container_width=False, sharing='streamlit')

        trace2d = go.Scatter(
            x=data['X'], y=data['Y'], text=data['Name'], mode='markers+text', marker = dict(
              size = 10.0,
              color = data['Z'], # set color to an array/list of desired values
              colorbar=dict(
                title="Elevation"
              ),
              colorscale = 'Viridis'
              )
        )
        layout2d = go.Layout(title='2D Scatter plot top view')
        fig2d = go.Figure(data=[trace2d], layout=layout2d)
        fig2d.update_traces(textposition='top center')
        range_x = data['X'].max() - data['X'].min()
        range_y = data['Y'].max() - data['Y'].min()
        ratio_yx = range_y/range_x
        fig2d.update_layout(autosize=False, width=800, height=800*ratio_yx)
        st.plotly_chart(fig2d, use_container_width=False, sharing='streamlit')

        st.header('Experimental Variogram (Variogram from measured data)')
        my_kriging = SpatialKriging(data)
        V = my_kriging.build_experimental_variogram()
        col1, col2, col3 = st.columns(3)
        model_options = ['spherical', 'exponential', 'gaussian', 'matern']
        V.model = col1.selectbox('Model', model_options, index=model_options.index(V.model.__name__), key='model')
        V.n_lags = col2.number_input('n_lags', value=V.n_lags)
        maxlag = col3.text_input('maxlag', value=V.maxlag)
        if V.maxlag is None:
            pass
        else:
            self.maxlag = float(maxlag)

        print(V)
        fig_variogram = V.plot()
        st.write(fig_variogram)

        st.header('Suggestions for variogram parameters for the model {0}:'.format(V.model.__name__))
        st.write(V)

        st.header('User settings for Variogram')
        col1, col2, col3, col4 = st.columns(4)
        model = col1.selectbox('Model', model_options, index=model_options.index(V.model.__name__), key='user_model_variogram')
        range_variogram = col2.number_input('Range', value=30.0, step=1.0)
        sill_variogram = col3.number_input('Sill', value=1.0, step=1.0)
        nugget_variogram = col4.number_input('Nugget', value=0.0, step=1.0)
        my_kriging.set_variogram_model_parameters(range_variogram, sill_variogram, nugget_variogram, model)

        # calculate
        st.header('Kriging data preparation')
        h_variogram = my_kriging.dist_matrix[:,:].flatten()
        gamma_variogram = my_kriging.variogram_matrix[:-1,:-1].flatten()
        fig, ax = plt.subplots()
        ax.scatter(h_variogram, gamma_variogram)
        ax.set_xlabel('h')
        ax.set_ylabel('$\gamma(h)$')
        col1, col2, col3 = st.columns([3, 4, 3])
        col1.markdown('Variogram')
        col1.write(fig)

        col2.markdown('Semivariance matrix from known data')
        col2.write(my_kriging.variogram_matrix)
        fig, ax = plt.subplots()
        sns.heatmap(my_kriging.variogram_matrix, ax=ax)
        col3.markdown('Heatmap plot of the semivariance matrix')
        col3.write(fig)
