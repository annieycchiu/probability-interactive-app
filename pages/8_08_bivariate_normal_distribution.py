import streamlit as st
import numpy as np
import plotly.graph_objs as go

from utils.other_utils import add_logo, setup_sticky_header, add_title
from utils.stats_viz import BivariateNormalDistribution


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Bivariate Normal Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Bivariate Normal Distribution'
        add_title(title)

        col1, _ ,col2, _, col3 = st.columns([0.30, 0.03, 0.30, 0.03, 0.30])
        with col1:
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Distribution X (yellow curve)</span>", 
                unsafe_allow_html=True)
            
            mean_x = st.slider("Mean X", -5.0, 5.0, 0.0)
            std_x = st.slider("Standard Deviation X", 0.1, 3.0, 1.0)

        with col2:
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Distribution Y (green curve)</span>", 
                unsafe_allow_html=True)
            
            mean_y = st.slider("Mean Y", -5.0, 5.0, 2.0)
            std_y = st.slider("Standard Deviation Y", 0.1, 3.0, 0.5)

        with col3:
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Correlation Coefficient (ρ)</span>", 
                unsafe_allow_html=True)
            
            rho = st.slider("Correlation Coefficient (ρ)", -1.0, 1.0, 0.0, label_visibility='collapsed')

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    biNormDist = BivariateNormalDistribution(mean_x, mean_y, std_x, std_y, rho)

    # Generate surface plot based on user inputs
    biNormDist.plot_bivariate_normal_3D()


if __name__ == '__main__':
    main()