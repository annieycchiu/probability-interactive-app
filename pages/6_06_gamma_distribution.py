import streamlit as st

from utils.stats_viz import GammaDistribution
from utils.other_utils import add_logo, setup_sticky_header, add_title, display_content_page_formulas, add_customized_expander
from utils.formulas import gamma_notation, gamma_pdf, gamma_cdf, gamma_exp, gamma_var

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Gamma Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add logo
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Gamma Distribution'
        add_title(title, gamma_notation)

        # Get user defined parameters
        col1, col2, _, col3, col4 = st.columns([0.1, 0.40, 0.05, 0.1, 0.20])
        with col1:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Parameters:</span>", 
            unsafe_allow_html=True)
        with col2:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                shape = st.slider('Shape (α)', min_value=0.1, max_value=10.0, value=2.0, step=0.01)
            with sub_col2:
                scale = st.slider('Inverse Scale or Rate (β)', min_value=0.1, max_value=10.0, value=2.0, step=0.01)
        with col3:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Empirical:</span>", 
            unsafe_allow_html=True)
        with col4:
            size = st.slider('Sample size', min_value=1, max_value=3000, value=500)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    # Add customized expander to display functions and formulas
    add_customized_expander()
    with st.expander("**:pushpin: Gamma Distribution - PDF, CDF, Expectation, Variance**"):
        display_content_page_formulas(
            gamma_pdf, gamma_cdf, gamma_exp, gamma_var, type='Continuous'
        )

    st.write('')

    # Split main app section into two columns. 
    # One for plotting PDF and the other one for plotting CDF
    gammaDist = GammaDistribution(shape, scale, size)

    col11, _, col12 = st.columns([0.45, 0.1, 0.45])
    with col11:
        gammaDist.plot_pdfs()

    with col12:
        gammaDist.plot_cdfs()

    # The following block of code is outdated. Keeping it for reference purposes. (old layout)
    # # Split main app section into two columns. 
    # # One for displaying formulas and the other one for plotting.
    # col11, _, col12 = st.columns([0.35, 0.05, 0.60])
    # with col11:
    #     display_content_page_formulas(
    #         gamma_pdf, gamma_cdf, gamma_exp, gamma_var, type='Continuous'
    #     ) 

    # with col12:
    #     gammaDist = GammaDistribution(shape, scale, size)
    #     gammaDist.plot_pdfs()
    #     gammaDist.plot_cdfs()

if __name__ == '__main__':
    main()
