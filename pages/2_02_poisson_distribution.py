import streamlit as st

from utils.stats_viz import PoissonDistribution
from utils.other_utils import add_logo, setup_sticky_header, add_title, display_content_page_formulas, add_customized_expander
from utils.formulas import poisson_notation, poisson_pmf, poisson_cdf, poisson_exp, poisson_var


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Poisson',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Poisson Distribution'
        add_title(title, poisson_notation)

        # Get user defined parameters
        col1, col2, _, col3, col4 = st.columns([0.1, 0.25, 0.1, 0.1, 0.25])
        with col1:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Parameters:</span>", 
            unsafe_allow_html=True)
        with col2:
            lmbda = st.slider('Constant average rate (λ)', min_value=0, max_value=20, value=5, step=1)
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
    with st.expander("**:pushpin: Poisson Distribution - PMF, CDF, Expectation, Variance**"):
        display_content_page_formulas(
            poisson_pmf, poisson_cdf, poisson_exp, poisson_var, type='Discrete'
        )

    st.write('')

    # Split main app section into two columns. 
    # One for plotting PMF and the other one for plotting simulated results
    poissonDist = PoissonDistribution(lmbda, size)

    col11, _, col12 = st.columns([0.45, 0.1, 0.45])
    with col11:
        # Plot PMF
        poissonDist.plot_theoretical_pmf()

    with col12:
        # Plot the simulation
        poissonDist.plot_empirial_pmf()

    col21, _ = st.columns([0.50, 0.50])
    with col21:
        # Display probability table
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Probability Table</span>", 
            unsafe_allow_html=True)
        
        poissonDist.plot_prob_table()

if __name__ == '__main__':
    main()
