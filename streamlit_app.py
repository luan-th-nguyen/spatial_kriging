import streamlit as st
from main_spatial_kriging import main_spatial_kriging
from main_spatial_kriging_test import main_spatial_kriging_test
from main_variogram import main_variogram



st.set_page_config(page_title='Spatial Kriging', layout="wide", page_icon="⚙️")


if __name__ == '__main__':
   st.sidebar.markdown('# Form selection')
   select_options = ['Variogram analysis', 'Spatial Kriging', 'Test Spatial Kriging']

   select_event = st.sidebar.selectbox('Select one of the forms', select_options, index=0)
    
   if select_event == 'Variogram analysis':
       main_variogram(st)

   if select_event == 'Spatial Kriging':
       main_spatial_kriging(st)

   elif select_event == 'Test Spatial Kriging':
       main_spatial_kriging_test(st)
