import pandas as pd
import numpy as np
#from scipy import interpolate as _interp #import griddata
import scipy
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.spatial_kriging import SpatialKriging

def main_spatial_kriging(st):
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

        st.header('User settings for Variogram')
        my_kriging = SpatialKriging(data)
        #V = my_kriging.build_experimental_variogram()
        col1, col2, col3, col4 = st.columns(4)
        model_options = ['spherical', 'exponential', 'gaussian', 'matern']
        model = col1.selectbox('Model', model_options, index=0, key='user_model_spatial_kriging')
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

    st.header('Unknown data import')
    uploaded_file_unknown = st.file_uploader("Choose a CSV file of unknown data points (to be estimated by Kriging)")
    if uploaded_file_unknown is not None:
        # read data
        data_unknown = pd.read_csv(uploaded_file_unknown)
        data_unknown = data_unknown.rename(columns=lambda c: c.strip()) # remove trailing whitespaces in column names
        st.write(data_unknown)

        st.header('Kriging results')
        col1, col2 = st.columns([3, 7])
        #p0 = [5, 5]
        points_unknown = [[xi, yi] for xi, yi in zip(data_unknown['X'], data_unknown['Y'])]
        z_est = [my_kriging.estimate_with_ordinary_kriging(point)[0] for point in points_unknown]
        var_est = [my_kriging.estimate_with_ordinary_kriging(point)[1] for point in points_unknown]
        data_unknown['Z'] = np.array(z_est)
        data_unknown['Var'] = np.array(var_est)
        col1.write(data_unknown)

        # visualize data
        trace_known = go.Scatter3d(
           x = data['X'], y = data['Y'], z = data['Z'], mode='markers', marker = dict(
              size = 5.0,
              color = 'grey', # set color to an array/list of desired values
              #colorscale = 'Viridis'
              )
           )
        trace_unknown = go.Scatter3d(
           x = data_unknown['X'], y = data_unknown['Y'], z = data_unknown['Z'], text=data_unknown['Name'], mode='markers+text', marker = dict(
              size = 5.0,
              color = data_unknown['Z'], # set color to an array/list of desired values
              colorscale = 'Viridis'
              )
           )
        layout = go.Layout(title = '3D Scatter plot (grey: known points, colored: estimated points)')
        fig = go.Figure(data = [trace_known, trace_unknown], layout = layout)
        col2.plotly_chart(fig, use_container_width=False, sharing='streamlit')

        # visualize grid data
        st.header('Generate dense data points within range of known data, perform kriging on each of the points and plot results')
        x_min, x_max = np.min(data['X']), np.max(data['X'])
        y_min, y_max = np.min(data['Y']), np.max(data['Y'])
        x_points_grid = np.linspace(x_min, x_max, 30)
        y_points_grid = np.linspace(y_min, y_max, 30)

        xx, yy = np.meshgrid(x_points_grid, y_points_grid)
        zz = np.zeros_like(xx)         # estimated mean
        zz_var = np.zeros_like(xx)     # estimation variance
        for i in range(zz.shape[0]):
          for j in range(zz.shape[1]):
            zz[i,j], zz_var[i,j], _, _ = my_kriging.estimate_with_ordinary_kriging((xx[i,j], yy[i,j]))

        layout = go.Layout(title = 'Estimated Mean (grey points: known points, surface: estimated elevation surface)',
                           autosize=False,
                           width=800,
                           height=800)
        trace_known = go.Scatter3d(
           x = data['X'], y = data['Y'], z = data['Z'], text=data['Name'], mode='markers+text', marker = dict(
              size = 5.0,
              color = 'grey', # set color to an array/list of desired values
              #colorscale = 'Viridis'
              )
           )

        trace2d_known = go.Scatter(
           x = data['X'], y = data['Y'], text=data['Name'], mode='markers+text', 
                  marker = dict(
                  size = 5.0,
                  color = 'black', # set color to an array/list of desired values
                  #colorscale = 'Viridis'
                     ),
              textfont=dict(
                  family="sans serif",
                  size=18,
                  color="black"
              )
           )

        fig1 = go.Figure(data=[trace_known, go.Surface(x=xx, y=yy, z=zz, colorscale = 'Viridis')], layout=layout)
        st.plotly_chart(fig1, use_container_width=False, sharing='streamlit')
        fig2 = go.Figure(data=[trace2d_known, go.Contour(x=x_points_grid, y=y_points_grid, z=zz, colorscale = 'Viridis')], layout=layout)
        st.plotly_chart(fig2, use_container_width=False, sharing='streamlit')

        layout_var = go.Layout(title = 'Estimation Variance (grey points: known points, surface: estimation variance)',
                           autosize=False,
                           width=800,
                           height=800)
        fig3 = go.Figure(data=[trace2d_known, go.Contour(x=x_points_grid, y=y_points_grid, z=zz_var, colorscale = 'Viridis')], layout=layout_var)
        st.plotly_chart(fig3, use_container_width=False, sharing='streamlit')

        # download results
        st.download_button(
        label="Download results as CSV",
        data=convert_df(data_unknown),
        file_name='Kriging_results.csv',
        mime='text/csv',
        )

