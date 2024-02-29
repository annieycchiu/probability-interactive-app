import streamlit as st

from utils.stats_viz import NormalDistribution
from utils.other_utils import add_logo, setup_sticky_header, add_title, add_exp_var

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Normal Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Define sticky header
    header = st.container()
    with header:
        title = 'Normal Distribution'
        notation = r'$X \sim N(\mu, \sigma^2)$'
        expectation_formula = r'$E[X] = \mu$'
        variance_formula = r'$Var(X) = \sigma^2$'

        add_title(title, notation)
        add_exp_var(expectation_formula, variance_formula)

        # Get user defined parameters
        _, col1, _, col2, _ = st.columns([0.1, 0.35, 0.1, 0.35, 0.1])
        with col1:
            mean = st.slider('Mean (μ)', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
        with col2:
            std_dev = st.slider('Standard Deviation (σ)', min_value=0.1, max_value=10.0, value=3.0, step=0.01)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    setup_sticky_header(header)

    # Add logo
    add_logo()

    # Split main app section into two columns. One for plotting PDF and the other one for plotting CDF
    col11, _, col12 = st.columns([0.35, 0.1, 0.55])
    with col11:
        st.write('**Probability Density Function (PDF)**')
        st.latex(r'f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}')
        st.write('**Cumulative Distribution Function (CDF)**')
        st.latex(r'F(x) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x - \mu}{\sigma \sqrt{2}}\right)\right] = \Phi\left(\frac{x - \mu}{\sigma}\right)')
        st.write('')
        st.write('')
        size = st.slider('Number of simulations', min_value=1, max_value=10000, value=5000)

    with col12:
        normalDist = NormalDistribution(mean, std_dev, size)
        normalDist.plot_simulation_pdf_cdf()
        

if __name__ == '__main__':
    main()

