import streamlit as st

from utils.stats_viz import ExponentialDistribution
from utils.other_utils import add_logo, setup_sticky_header, add_title


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Exponential Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add logo
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Exponential Distribution'
        notation = r'$X \sim Exp(\lambda)$'

        add_title(title, notation)

        # Get user defined parameters
        col1, col2, _, col3, col4 = st.columns([0.1, 0.25, 0.1, 0.1, 0.25])
        with col1:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Parameters:</span>", 
            unsafe_allow_html=True)
        with col2:
            rate = st.slider('Rate (λ)', min_value=0.1, max_value=10.0, value=1.0, step=0.01)
        with col3:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Simulation:</span>", 
            unsafe_allow_html=True)
        with col4:
            size = st.slider('Number of simulations', min_value=1, max_value=10000, value=5000)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    # Split main app section into two columns. 
    # One for displaying formulas and the other one for plotting.
    col11, _, col12 = st.columns([0.25, 0.1, 0.75])
    with col11:
        # Display PDF function
        pdf_func = r'f(x) = \lambda e^{-\lambda x}'

        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Probability Density Function (PDF)</span>",
            unsafe_allow_html=True)
        st.latex(pdf_func)
        st.write('')

        # Display CDF function
        cdf_func = r'F(x) = 1 - e^{-\lambda x}'

        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Cumulative Distribution Function (CDF)</span>",
            unsafe_allow_html=True)
        st.latex(cdf_func)
        st.write('')
        st.write('')  

        # Display expectation formula
        expectation_formula = r'$E[X] = \frac{1}{\lambda}$'
        
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Expectation:</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left: 10px;'>", expectation_formula, "</span>", 
            unsafe_allow_html=True)
        st.write('')

        # Display variance formula
        variance_formula = r'$Var(X) = \frac{1}{\lambda^2}$'

        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Variance:</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left: 37px;'>", variance_formula, "</span>", 
            unsafe_allow_html=True)
        st.write('')
        st.write('')

    with col12:
        exponentialDist = ExponentialDistribution(rate, size)
        exponentialDist.plot_pdfs()
        exponentialDist.plot_cdfs()

if __name__ == '__main__':
    main()
