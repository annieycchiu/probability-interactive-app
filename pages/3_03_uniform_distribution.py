import streamlit as st

from utils.stats_viz import UniformDistribution
from utils.other_utils import add_logo, setup_sticky_header, add_title, add_exp_var

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Uniform Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Define sticky header
    header = st.container()
    with header:
        title = 'Uniform Distribution'
        notation = r'$X \sim U(\alpha, \beta)$'
        expectation_formula = r'$E[X] = \frac{1}{2}(\alpha + \beta)$'
        variance_formula = r'$Var(X) = \frac{1}{12}(\beta - \alpha)^2$'

        add_title(title, notation)
        add_exp_var(expectation_formula, variance_formula)

        # Get user defined parameters
        _, col1, _, _, _ = st.columns([0.1, 0.35, 0.1, 0.35, 0.1])
        with col1:
            lower, upper = st.slider(
                'Lower Bound (α) and Upper Bound (β)', 
                min_value=-30.0, max_value=30.0, value=(10.0, 15.0), step=0.1)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    setup_sticky_header(header)

    # Add logo
    add_logo()

    # Split main app section into two columns. One for plotting PDF and the other one for plotting CDF
    col11, _, col12 = st.columns([0.35, 0.1, 0.55])
    with col11:
        st.write('**Probability Density Function (PDF)**')
        st.latex(r'f(x) = \begin{cases}\frac{1}{b-a} & \text{for } a \leq x \leq b,\\0 & \text{for } x < a \text{ or } x > b.\end{cases}')
        st.write('**Cumulative Distribution Function (CDF)**')
        st.latex(r'F(x) = \begin{cases}0 & \text{for } x < a,\\\frac{x-a}{b-a} & \text{for } a \leq x \leq b,\\1 & \text{for } x > b.\end{cases}')

        st.write('')
        st.write('')
        size = st.slider('Number of simulations', min_value=1, max_value=10000, value=5000)

    with col12:
        uniformDist = UniformDistribution(lower, upper, size)
        uniformDist.plot_simulation_pdf_cdf()

if __name__ == '__main__':
    main()
