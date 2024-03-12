import streamlit as st

from utils.stats_viz import NormalDistribution
from utils.other_utils import add_logo, setup_sticky_header, add_title, display_content_page_formulas, add_customized_expander
from utils.formulas import normal_notation, normal_pdf, normal_cdf, normal_exp, normal_var

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Normal Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add logo
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Normal Distribution'
        add_title(title, normal_notation)

        # Get user defined parameters
        col1, col2, _, col3, col4 = st.columns([0.1, 0.40, 0.05, 0.1, 0.20])
        with col1:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Parameters:</span>", 
            unsafe_allow_html=True)
        with col2:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                mean = st.slider('Mean (μ)', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
            with sub_col2:
                std_dev = st.slider('Standard Deviation (σ)', min_value=0.1, max_value=10.0, value=3.0, step=0.01)
        with col3:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Simulation:</span>", 
            unsafe_allow_html=True)
        with col4:
            size = st.slider('Number of simulations', min_value=1, max_value=10000, value=5000)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    setup_sticky_header(header)

    # Add customized expander to display functions and formulas
    add_customized_expander()
    with st.expander("**:pushpin: Normal Distribution - PDF, CDF, Expectation, Variance**"):
        display_content_page_formulas(
            normal_pdf, normal_cdf, normal_exp, normal_var, type='Continuous'
        )

    st.write('')

    # Split main app section into two columns. 
    # One for plotting PDF and the other one for plotting CDF
    normalDist = NormalDistribution(mean, std_dev, size)

    col11, _, col12 = st.columns([0.45, 0.1, 0.45])
    with col11:
        normalDist.plot_pdfs()

    with col12:
        normalDist.plot_cdfs()

    # The following block of code is outdated. Keeping it for reference purposes. (old layout)
    # # Split main app section into two columns. 
    # # One for displaying formulas and the other one for plotting.
    # col11, _, col12 = st.columns([0.35, 0.05, 0.60])
    # with col11:
    #     display_content_page_formulas(
    #         normal_pdf, normal_cdf, normal_exp, normal_var, type='Continuous'
    #     )     

    # with col12:
    #     normalDist = NormalDistribution(mean, std_dev, size)
    #     normalDist.plot_pdfs()
    #     normalDist.plot_cdfs()
        

if __name__ == '__main__':
    main()

