import streamlit as st

from utils.stats_viz import PoissonPMF, PoissonSimulation
from utils.other_utils import add_logo, setup_sticky_header, add_title, add_exp_var


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Poisson',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Define sticky header
    header = st.container()
    with header:
        title = 'Poisson Distribution'
        notation = r'$X \sim \mathrm{Poi}(\lambda)$'
        expectation_formula = r'$E[X] = \lambda$'
        variance_formula = r'$Var(X) = \lambda$'

        add_title(title, notation)
        add_exp_var(expectation_formula, variance_formula)

        # Get user defined parameter
        _, col1, _ = st.columns([0.1, 0.35, 0.55])
        with col1:
            lmbda = st.slider('Constant average rate (λ)', min_value=0, max_value=20, value=5, step=1)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    setup_sticky_header(header)

    # Add USF logo at sidebar
    add_logo()

    # Split main app section into two columns. One for plotting PMF and the other one for plotting simulated results
    col11, _, col12 = st.columns([0.45, 0.1, 0.45])
    with col11:
        st.write('**Probability Mass Function**')
        st.latex(r'P(X=x) = \frac{{\lambda^x e^{{-\lambda}}}}{{x!}}')

    with col12:
        st.write('**Simulations**')
        _, col, _ = st.columns([0.05, 0.35, 0.05])
        with col:
            size = st.slider('Number of simulations', min_value=1, max_value=10000, value=5000)

    # Calculate PMF
    poissonPmf = PoissonPMF(lmbda)

    # Run the simulation
    poissonSim = PoissonSimulation(lmbda, size)

    col21, _, _ = st.columns([0.45, 0.1, 0.45])
    with col21:
        # Display probability table
        st.write('**Probability Table**')
        poissonPmf.plot_prob_table()

    col31, _, col32 = st.columns([0.45, 0.1, 0.45])
    with col31:
        # Plot PMF
        poissonPmf.plot_pmf()

    with col32:
        # Plot the simulation
        poissonSim.plot_simulation()

if __name__ == '__main__':
    main()
