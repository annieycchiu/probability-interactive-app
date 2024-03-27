import streamlit as st

from utils.stats_viz import BinomialDistribution
from utils.other_utils import add_logo, setup_sticky_header, add_title, display_content_page_formulas, add_customized_expander
from utils.formulas import binomial_notation, binomial_pmf, binomial_cdf, binomial_exp, binomial_var


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Binomial Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Binomial Distribution'
        add_title(title, binomial_notation)

        # Get user defined parameters
        col1, col2, _, col3, col4 = st.columns([0.1, 0.40, 0.05, 0.1, 0.20])
        with col1:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Parameters:</span>", 
            unsafe_allow_html=True)
        with col2:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                n = st.slider('Number of trials (n)', min_value=1, max_value=100, value=20)
            with sub_col2:
                p = st.slider('Probability of success (p)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        with col3:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Simulation:</span>", 
            unsafe_allow_html=True)
        with col4:
            size = st.slider('Number of simulations', min_value=1, max_value=3000, value=500)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    # Add customized expander to display functions and formulas
    # add_customized_expander()
    # Custom CSS 
    st.markdown(
        '''
        <style>
        .streamlit-expanderHeader {
            background-color: green;
            color: white; 
        }
        .streamlit-expanderContent {
            background-color: rgba(240, 242, 246);
            color: black; 
        }
        </style>
        ''',
        unsafe_allow_html=True
    )
    with st.expander("**:pushpin: Binomial Distribution - PMF, CDF, Expectation, Variance**"):
        display_content_page_formulas(
            binomial_pmf, binomial_cdf, binomial_exp, binomial_var, type='Discrete'
        )

    st.write('')

    # Split main app section into two columns. 
    # One for plotting PMF and the other one for plotting simulated results
    binomialDist = BinomialDistribution(n, p, size)

    col21, _, col22 = st.columns([0.45, 0.1, 0.45])
    with col21:
        # Plot Theoretical PMF
        binomialDist.plot_theoretical_pmf()

    with col22:
        # Plot the simulation
        binomialDist.plot_empirial_pmf()

    col31, _ = st.columns([0.50, 0.50])
    with col31:
        # Display probability table
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Probability Table</span>", 
            unsafe_allow_html=True)
        
        binomialDist.plot_prob_table()

if __name__ == '__main__':
    main()

