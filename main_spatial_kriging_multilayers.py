import pandas as pd
import numpy as np
#from scipy import interpolate as _interp #import griddata
import scipy
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.spatial_kriging import SpatialKriging

def main_spatial_kriging_multilayers(st):
    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    st.header('Known data import')
    uploaded_file = st.file_uploader("Choose a CSV file of known data points")
    if uploaded_file is not None:
         # read data
         data = pd.read_csv(uploaded_file)
         data = data.rename(columns=lambda c: c.strip()) # remove trailing whitespaces in column names
         st.write(data)

         # determine the number of layers
         number_layers = len(data.columns) - 3

         # experimental variograms
         st.header('Experimental variograms')
         my_krigings = [None]*number_layers
         for i in range(number_layers):
             data_column_z = data[data.columns[[3+i]]]
             data_i = data[["X", "Y"]]
             data_i["Z"] = data_column_z
             my_kriging_i = {}
             my_kriging_i['model'] = SpatialKriging(data_i) # variogram function
             V = my_kriging_i['model'].build_experimental_variogram()
             col0, col1, col2, col3, col4, col5 = st.columns([1, 1,1,1,2,2])
             col0.text('Layer ' + str(i))
             model_options = ['spherical', 'exponential', 'gaussian', 'matern']
             V.model = col1.selectbox('Model', model_options, index=model_options.index(V.model.__name__), key='model_layer_' + str(i))
             V.n_lags = col2.number_input('n_lags', value=V.n_lags, key='n_lags_layer_' + str(i))
             maxlag = col3.text_input('maxlag', value=V.maxlag, key='max_layer_' + str(i))
             if V.maxlag is None:
                 pass
             else:
                 V.maxlag = float(maxlag)

             print(V)
             fig_variogram = V.plot()
             col4.write(fig_variogram)
             col5.write(V)       # suggested variogram parameters
             print(V.parameters) # [range, sill, nugget]
             my_kriging_i['V'] = V                       # experimental variogram model and parameters
             my_kriging_i['data'] = data_i               # known data will be later used for plotting
             my_krigings[i] = my_kriging_i               # store variogram for the layer

         # set variogram functions
         st.header('User settings for Variogram for each of the layers')
         use_experimental_V = st.checkbox('Use experimental variogram function and parameters?', value=False)
         for i in range(number_layers):
             col0, col1, col2, col3, col4 = st.columns(5)
             model_options = ['spherical', 'exponential', 'gaussian', 'matern']
             col0.text('Layer ' + str(i))
             if not use_experimental_V:
                 model = col1.selectbox('Model', model_options, index=0, key='user_model_spatial_kriging_layer_'+ str(i))
                 range_variogram = col2.number_input('Range', value=30.0, step=1.0, key='range_layer_' + str(i))
                 sill_variogram = col3.number_input('Sill', value=1.0, step=1.0, key='sill_layer_' + str(i))
                 nugget_variogram = col4.number_input('Nugget', value=0.0, step=1.0, key='nugget_layer_' + str(i))

             else: # use model and parameters estimated from experimental data
                 model = col1.selectbox('Model', model_options, index=model_options.index(my_krigings[i]['V'].model.__name__), key='user_model_spatial_kriging_layer_'+ str(i))
                 range_variogram = col2.number_input('Range', value=my_krigings[i]['V'].parameters[0], step=1.0, key='range_layer_' + str(i))
                 sill_variogram = col3.number_input('Sill', value=my_krigings[i]['V'].parameters[1], step=1.0, key='sill_layer_' + str(i))
                 nugget_variogram = col4.number_input('Nugget', value=float(my_krigings[i]['V'].parameters[2]), step=1.0, key='nugget_layer_' + str(i))

             my_krigings[i]['model'].set_variogram_model_parameters(range_variogram, sill_variogram, nugget_variogram, model)

         # calculate variance matrices
         show_matrices = st.checkbox('Show variogram and variance matrices?', value=False)
         if show_matrices:
            for i in range(number_layers):
                st.header('Kriging data preparation')
                h_variogram = my_krigings[i]['model'].dist_matrix[:,:].flatten()
                gamma_variogram = my_krigings[i]['model'].variogram_matrix[:-1,:-1].flatten()
                fig, ax = plt.subplots()
                ax.scatter(h_variogram, gamma_variogram)
                ax.set_xlabel('h')
                ax.set_ylabel('$\gamma(h)$')
                col1, col2, col3 = st.columns([3, 4, 3])
                col1.markdown('Variogram')
                col1.write(fig)

                col2.markdown('Semivariance matrix from known data')
                col2.write(my_krigings[i]['model'].variogram_matrix)
                fig, ax = plt.subplots()
                sns.heatmap(my_krigings[i]['model'].variogram_matrix, ax=ax)
                col3.markdown('Heatmap plot of the semivariance matrix')
                col3.write(fig)


         # show results
         st.header('Generate dense data points within range of known data, perform kriging on each of the points and plot results')
         show_results = st.checkbox('Show Kriging results?', value=False)
         if show_results:
             # visualize grid data
             for ii in range(number_layers):
                 x_min, x_max = np.min(my_krigings[ii]['data']['X']), np.max(my_krigings[ii]['data']['X'])
                 y_min, y_max = np.min(my_krigings[ii]['data']['Y']), np.max(my_krigings[ii]['data']['Y'])
                 x_points_grid = np.linspace(x_min, x_max, 30)
                 y_points_grid = np.linspace(y_min, y_max, 30)

                 xx, yy = np.meshgrid(x_points_grid, y_points_grid)
                 zz = np.zeros_like(xx)         # estimated mean
                 zz_var = np.zeros_like(xx)     # estimation variance
                 for i in range(zz.shape[0]):
                   for j in range(zz.shape[1]):
                     zz[i,j], zz_var[i,j], _, _ = my_krigings[ii]['model'].estimate_with_ordinary_kriging((xx[i,j], yy[i,j]))

                 trace_known = go.Scatter3d(
                    x = my_krigings[ii]['data']['X'], y = my_krigings[ii]['data']['Y'], z = my_krigings[ii]['data']['Z'], text='', mode='markers+text', marker = dict(
                       size = 5.0,
                       color = 'grey', # set color to an array/list of desired values
                       #colorscale = 'Viridis'
                       )
                    )
                 data_out = [trace_known, go.Surface(x=xx, y=yy, z=zz)]
                 my_krigings[ii]['data_out'] = data_out


             layout = go.Layout(title = 'Estimated Mean (grey points: known points, surface: estimated elevation surface)',
                                autosize=False,
                                width=1200,
                                height=1200)
             data_to_plot = []
             for ii in range(number_layers):
                 data_to_plot += my_krigings[ii]['data_out']
             fig1 = go.Figure(data=data_to_plot, layout=layout)
             st.plotly_chart(fig1, use_container_width=False, sharing='streamlit')