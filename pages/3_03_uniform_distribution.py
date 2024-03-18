import streamlit as st

from utils.stats_viz import UniformDistribution
from utils.other_utils import add_logo, setup_sticky_header, add_title, display_content_page_formulas, add_customized_expander
from utils.formulas import uniform_notation, uniform_pdf, uniform_cdf, uniform_exp, uniform_var

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Uniform Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add logo
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Uniform Distribution'
        add_title(title, uniform_notation)

        # Get user defined parameters
        col1, col2, _, col3, col4 = st.columns([0.1, 0.25, 0.1, 0.1, 0.25])
        with col1:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Parameters:</span>", 
            unsafe_allow_html=True)
        with col2:
            lower, upper = st.slider(
                'Lower Bound (α) and Upper Bound (β)', 
                min_value=-30.0, max_value=30.0, value=(10.0, 15.0), step=0.1)
        with col3:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Simulation:</span>", 
            unsafe_allow_html=True)
        with col4:
            size = st.slider('Number of simulations', min_value=1, max_value=3000, value=500)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    setup_sticky_header(header)

    # Add customized expander to display functions and formulas
    add_customized_expander()
    with st.expander("**:pushpin: Uniform Distribution - PDF, CDF, Expectation, Variance**"):
        display_content_page_formulas(
            uniform_pdf, uniform_cdf, uniform_exp, uniform_var, type='Continuous'
        )

    st.write(' ')

    # Split main app section into two columns. 
    # One for plotting PDF and the other one for plotting CDF
    uniformDist = UniformDistribution(lower, upper, size)

    col11, _, col12 = st.columns([0.45, 0.1, 0.45])
    with col11:
        uniformDist.plot_pdfs()

    with col12:
        uniformDist.plot_cdfs()

    # The following block of code is outdated. Keeping it for reference purposes. (old layout)
    # # Split main app section into two columns. 
    # # One for displaying formulas and the other one for plotting.
    # col11, _, col12 = st.columns([0.35, 0.05, 0.60])
    # with col11:
    #     display_content_page_formulas(
    #         uniform_pdf, uniform_cdf, uniform_exp, uniform_var, type='Continuous'
    #     ) 

    # with col12:
    #     uniformDist = UniformDistribution(lower, upper, size)
    #     uniformDist.plot_pdfs()
    #     uniformDist.plot_cdfs()

if __name__ == '__main__':
    main()
