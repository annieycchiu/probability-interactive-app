colors = {
    'USF_Green': '#00543C',
    'USF_Yellow': '#FDBB30',
    'USF_Gray': '#75787B'
}

import streamlit as st
import numpy as np
import plotly.graph_objs as go
from utils.other_utils import add_logo, setup_sticky_header, add_title

class GammaDistribution:
    def __init__(self, shape, scale, size=1, colors=colors):
        """
        Parameters:
        - shape (float): Shape parameter of the gamma distribution.
        - scale (float): Scale parameter of the gamma distribution.
        - size (int): Number of simulations to run (default is 1).
        """
        self.shape = shape
        self.scale = scale
        self.size = size
        self.colors = colors

        self.simulated_data = None
        self.simulate_gamma_dist_data()

        self.x_axis_min = None
        self.x_axis_max = None
        self.x_vals = None
        self.generate_x_axis()

        self.pdf_vals = None
        self.calculate_pdf()

        self.cdf_vals = None
        self.calculate_cdf()

    def simulate_gamma_dist_data(self):
        """
        Simulate gamma distribution data based on the provided parameters.
        """
        simulated_data = np.random.gamma(shape=self.shape, scale=self.scale, size=self.size)
        self.simulated_data = simulated_data

    def generate_x_axis(self):
        self.x_axis_min = 0
        self.x_axis_max = np.max(self.simulated_data)
        self.x_vals = np.linspace(self.x_axis_min, self.x_axis_max, 1000)

    def calculate_pdf(self):
        """
        Calculate the probability density function (PDF) for the gamma distribution.
        """
        self.pdf_vals = (1/(self.scale**self.shape * np.math.gamma(self.shape))) * (self.x_vals**(self.shape - 1)) * np.exp(-self.x_vals/self.scale)

    def calculate_cdf(self):
        """
        Calculate the cumulative distribution function (CDF) for the gamma distribution.
        """
        self.cdf_vals = np.cumsum(self.pdf_vals) * (self.x_vals[1] - self.x_vals[0])

    def plot_pdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data,
            histnorm='probability density',
            name='Simulation',
            marker_color=self.colors['USF_Green'],
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for PDF
        pdf_trace = go.Scatter(
            x=self.x_vals, y=self.pdf_vals,
            mode='lines', name='PDF',
            line=dict(color=self.colors['USF_Yellow'], width=3),
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        pdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title='Gamma Distribution Simulation (PDF)',
            updatemenus=[{
                'buttons': [{
                    'args': [{'visible': [True, False]}],
                    'label': 'Empirical PDF',
                    'method': 'update',
                }, {
                    'args': [{'visible': [True, True]}],
                    'label': 'Overlay Theoretical PDF',
                    'method': 'update'
                }],
                'type': 'buttons',
                'direction': 'down',
                'showactive': False,
                'x': -0.35,
                'xanchor': 'left', 'yanchor': 'top'}])

        # Create figure
        fig = go.Figure(data=[simulation_trace, pdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig)

    def plot_cdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data, 
            histnorm='probability density',
            cumulative_enabled=True,
            name='Simulation',
            marker_color=self.colors['USF_Green'],
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for CDF
        cdf_trace = go.Scatter(
            x=self.x_vals, y=self.cdf_vals,
            mode='lines', name='CDF',
            line=dict(color=self.colors['USF_Yellow'], width=3),
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        cdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title='Gamma Distribution Simulation (CDF)',
            updatemenus=[{
                'buttons': [{
                    'args': [{'visible': [True, False]}],
                    'label': 'Empirical CDF',
                    'method': 'update'
                }, {
                    'args': [{'visible': [True, True]}],
                    'label': 'Overlay Theoretical CDF',
                    'method': 'update'
                }],
                'type': 'buttons',
                'direction': 'down',
                'showactive': False,
                'x': -0.35,
                'xanchor': 'left', 'yanchor': 'top'}])

        # Create figure
        fig = go.Figure(data=[simulation_trace, cdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig)

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
        notation = r'$X \sim \Gamma (\alpha, \beta) \equiv \operatorname{Gamma}(\alpha, \beta)$'

        add_title(title, notation)

        # Get user defined parameters
        col1, col2, _, col3, col4 = st.columns([0.1, 0.25, 0.1, 0.1, 0.25])
        with col1:
            st.write(
            "<span style='font-size:18px; font-weight:bold;'>Parameters:</span>", 
            unsafe_allow_html=True)
        with col2:
            shape = st.slider('Shape (α)', min_value=0.1, max_value=10.0, value=2.0, step=0.01)
            scale = st.slider('Inverse Scale/ Rate (β)', min_value=0.1, max_value=10.0, value=2.0, step=0.01)
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
        pdf_func = r'f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x} \quad \text{ for } x \ge 0 \quad \alpha, \beta \ge 0'

        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Probability Density Function (PDF)</span>",
            unsafe_allow_html=True)
        st.latex(pdf_func)
        st.write('')

        # Display CDF function
        cdf_func = r'F(x) = \frac{1}{\Gamma(\alpha)} \gamma(\alpha, \beta x)'

        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Cumulative Distribution Function (CDF)</span>",
            unsafe_allow_html=True)
        st.latex(cdf_func)
        st.write('')
        st.write('')

        # Display expectation formula
        expectation_formula = r'$E[X] = \frac{\alpha}{\beta}$'
        
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Expectation:</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left: 10px;'>", expectation_formula, "</span>", 
            unsafe_allow_html=True)
        st.write('')

        # Display variance formula
        variance_formula = r'$Var(X) = \frac{\alpha}{\beta^2}$'

        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Variance:</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left: 37px;'>", variance_formula, "</span>", 
            unsafe_allow_html=True)
        st.write('')
        st.write('')  

    with col12:
        gammaDist = GammaDistribution(shape, scale, size)
        gammaDist.plot_pdfs()
        gammaDist.plot_cdfs()

if __name__ == '__main__':
    main()
