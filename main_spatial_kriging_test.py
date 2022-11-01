import pandas as pd
import numpy as np
#from scipy import interpolate as _interp #import griddata
import scipy
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.spatial_kriging import SpatialKriging

def main_spatial_kriging_test(st):

   st.header('Known data')
   d = {'X': [2.0, 3.0, 9.0, 6.0, 5.0], 'Y': [2.0, 7.0, 9.0, 5.0, 3.0], 'Z': [3.0, 4.0, 2.0, 4.0, 6.0]}
   data = pd.DataFrame(data =d)

   col1, col2 = st.columns([3, 7])
   col1.write(data)

   # visualize data
   trace = go.Scatter3d(
      x = data['X'], y = data['Y'], z = data['Z'], mode='markers', marker = dict(
         size = 5.0,
         color = data['Z'], # set color to an array/list of desired values
         colorscale = 'Viridis'
         )
      )
   layout = go.Layout(title = '3D Scatter plot')
   fig = go.Figure(data = [trace], layout = layout)
   col2.plotly_chart(fig, use_container_width=False, sharing='streamlit')

   st.header('Settings for Variogram')
   col1, col2, col3 = st.columns(3)
   range_variogram = col1.number_input('Range', value=10.0, step=1.0)
   sill_variogram = col2.number_input('Sill', value=7.5, step=1.0)
   nugget_variogram = col3.number_input('Nugget', value=2.5, step=1.0)

   # calculate
   st.header('Kriging data preparation')
   my_kriging = SpatialKriging(data, range_variogram, sill_variogram, nugget_variogram)
   h_variogram = my_kriging.dist_matrix[:,1]
   gamma_variogram = my_kriging.variance_dist_matrix[:-1,1]
   fig, ax = plt.subplots()
   ax.scatter(h_variogram, gamma_variogram)
   ax.set_xlabel('h')
   ax.set_ylabel('$\gamma(h)$')
   col1, col2, col3 = st.columns([3, 4, 3])
   col1.markdown('Variogram')
   col1.write(fig)

   col2.markdown('Semivariance matrix from known data')
   col2.write(my_kriging.variance_dist_matrix)
   fig, ax = plt.subplots()
   sns.heatmap(my_kriging.variance_dist_matrix, ax=ax)
   col3.markdown('Heatmap plot of the semivariance matrix')
   col3.write(fig)



   st.header('Unknown data')
   d1 = {'X': [5.0], 'Y': [5.0]}
   data_unknown = pd.DataFrame(data=d1)
   st.write(data_unknown)

   st.header('Kriging results')
   col1, col2 = st.columns([3, 7])
   #p0 = [5, 5]
   points_unknown = [[xi, yi] for xi, yi in zip(data_unknown['X'], data_unknown['Y'])]
   z_est = [my_kriging.estimate_with_ordinary_kriging(point)[0] for point in points_unknown]
   data_unknown['Z'] = np.array(z_est)
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
      x = data_unknown['X'], y = data_unknown['Y'], z = data_unknown['Z'], mode='markers', marker = dict(
         size = 5.0,
         color = data_unknown['Z'], # set color to an array/list of desired values
         colorscale = 'Viridis'
         )
      )
   layout = go.Layout(title = '3D Scatter plot (grey: known points, colored: estimated points)')
   fig = go.Figure(data = [trace_known, trace_unknown], layout = layout)
   col2.plotly_chart(fig, use_container_width=False, sharing='streamlit')

   # weights
   st.header('Verifications')
   col1, col2, col3 = st.columns(3)
   col1.markdown('Semivariance vector')
   variance_vector = [my_kriging.estimate_with_ordinary_kriging(point)[2] for point in points_unknown]
   col1.write(variance_vector[0][:-1])

   col2.markdown('Weights')
   weights = [my_kriging.estimate_with_ordinary_kriging(point)[1] for point in points_unknown]
   col2.write(weights[0][:-1])
   col3.markdown('Sum of weights (must be ~1.0)')
   col3.write(np.sum(weights[0][:-1]))

   # visualize grid data
   st.header('Generate dense data points within range of known data, perform kriging on each of the points and plot results')
   x_min, x_max = np.min(data['X']), np.max(data['X'])
   y_min, y_max = np.min(data['Y']), np.max(data['Y'])
   x_points_grid = np.linspace(x_min, x_max, 50)
   y_points_grid = np.linspace(y_min, y_max, 50)
   points_grid = [[xi, yi] for xi, yi in zip(x_points_grid, y_points_grid)]
   z_points_grid = [my_kriging.estimate_with_ordinary_kriging(point)[0] for point in points_grid]

   X, Y = np.meshgrid(x_points_grid, y_points_grid)
   xy_points = np.array((x_points_grid, y_points_grid)).transpose()
   z_values = np.array((z_points_grid))
   XY_points = (X, Y) # points at which to interpolate data

   Z = scipy.interpolate.griddata(xy_points, z_values, XY_points, method='nearest')
   layout = go.Layout(title = 'Estimated Elevation (grey points: known points, surface: estimated elevation surface)')
   trace_known = go.Scatter3d(
      x = data['X'], y = data['Y'], z = data['Z'], mode='markers', marker = dict(
         size = 5.0,
         color = 'grey', # set color to an array/list of desired values
         #colorscale = 'Viridis'
         )
      )
   fig1 = go.Figure(data=[trace_known, go.Surface(x=x_points_grid, y=y_points_grid, z=Z)], layout=layout)
   st.plotly_chart(fig1, use_container_width=False, sharing='streamlit')


