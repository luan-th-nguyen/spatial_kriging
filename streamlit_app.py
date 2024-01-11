import streamlit as st
from main_spatial_kriging import main_spatial_kriging
from main_spatial_kriging_multilayers import main_spatial_kriging_multilayers
from main_spatial_kriging_test import main_spatial_kriging_test
from main_variogram import main_variogram
from main_hangsicherung_vernagelung import main_hangsicherung_vernagelung



st.set_page_config(page_title='Spatial Kriging', layout="wide", page_icon="⚙️")


if __name__ == '__main__':
    st.sidebar.markdown('# Form selection')
    select_options = ['Variogram analysis', 'Spatial Kriging, Elementary test', 'Spatial Kriging, Single layer', 'Spatial Kriging, Multiple layers', 'Test Spatial Kriging',
                      'Hangsicherung durch Vernagelung']

    index_current = select_options.index('Hangsicherung durch Vernagelung') 
    select_event = st.sidebar.selectbox('Select one of the forms', select_options, index=index_current)

    if select_event == 'Variogram analysis':
        main_variogram(st)

    if select_event == 'Spatial Kriging, Elementary test':
        main_spatial_kriging_test(st)

    if select_event == 'Spatial Kriging, Single layer':
        main_spatial_kriging(st)

    if select_event == 'Spatial Kriging, Multiple layers':
        main_spatial_kriging_multilayers(st)

    elif select_event == 'Test Spatial Kriging':
        main_spatial_kriging_test(st)

    elif select_event == 'Hangsicherung durch Vernagelung':
        main_hangsicherung_vernagelung(st)