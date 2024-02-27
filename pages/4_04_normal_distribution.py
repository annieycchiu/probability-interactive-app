import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.special import erf

from utils.other_utils import add_logo

def normal_simulation(mu, sigma, size=1):
    """
    Simulate Normal distribution.
    
    Parameters:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
        size (int): Number of simulations.
        
    Returns:
        ndarray: Array of simulated outcomes.
    """
    return np.random.normal(mu, sigma, size)

def plot_normal_simulation(simulated_data):
    """
    Plot the histogram of Normal simulation.
    
    Parameters:
        simulated_data (ndarray): Array of simulated outcomes.
    """
    hist, bins = np.histogram(simulated_data, bins='auto', density=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bins, y=hist, marker_color='#205C40'))
    fig.update_layout(
        title="Normal Distribution Simulation",
        xaxis_title="Value",
        yaxis_title="Density",
        bargap=0.02,
    )
    st.plotly_chart(fig)

def probability_table(mu, sigma):
    # Generate values for the random variable
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    
    # Calculate PDF using Normal distribution formula
    pdf = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Calculate CDF using cumulative sum of PDF
    cdf = np.cumsum(pdf) * (x[1] - x[0])

    prob_df = pd.DataFrame({
        'x': x, 
        'f(x)': pdf,
        'F(x)': cdf})
    return prob_df

def plot_normal_pdf_cdf(mu, sigma):
    """
    Plot the PDF and CDF of Normal distribution.
    
    Parameters:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
    """
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    pdf = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    cdf = 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', marker_color='#205C40'))
    fig.add_trace(go.Scatter(x=x, y=cdf, mode='lines', name='CDF', marker_color='#F7B512'))
    fig.update_layout(
        title="Normal Probability Density and Cumulative Distribution Functions",
        xaxis_title="Value",
        yaxis_title="Density / Probability",
    )
    st.plotly_chart(fig)

def main():

    add_logo()
    
    st.title('Normal Distribution')

    # # Side bar
    # st.sidebar.image('usfca_logo.png')

    st.write('Notation')
    st.latex(r'X \sim \mathcal{N}(\mu, \sigma^2)')
    st.write('Probability Density Function (PDF)')
    st.latex(r'f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}')
    st.write('Cumulative Distribution Function (CDF)')
    st.latex(r'F(x) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]')
    st.write('Expectation')
    st.latex(r'E[X] = \mu')
    st.write('Variance')
    st.latex(r'Var(X) = \sigma^2')

    # Input parameters
    mu = st.sidebar.slider('Mean (μ)', min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    sigma = st.sidebar.slider('Standard Deviation (σ)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    size = st.sidebar.slider('Number of simulations', min_value=1, max_value=10000, value=1000)

    st.write('\n\n')
    # col1, col2 = st.columns([0.2, 0.8])

    # with col1:
    #     # Display probability table
    #     st.markdown('Probability Table')
    #     prob_df = probability_table(mu, sigma)
    #     st.markdown(prob_df.to_html(index=False), unsafe_allow_html=True)

    # with col2:
    # Plot PDF and CDF
    plot_normal_pdf_cdf(mu, sigma)

    # Simulate Normal distribution
    simulated_data = normal_simulation(mu, sigma, size)

    # Plot the simulation
    plot_normal_simulation(simulated_data)


if __name__ == '__main__':
    main()
