import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

import plotly.graph_objs as go
import streamlit as st

colors = {
    'USF_Green': '#00543C',
    'USF_Green_rbga_fill': 'rgba(0, 84, 60, 0.4)',
    'USF_Yellow': '#FDBB30',
    'USF_Yellow_rbga_line': 'rgba(253, 187, 48, 0.8)',
    'USF_Yellow_rbga_fill': 'rgba(253, 187, 48, 0.4)',
    'USF_Gray': '#75787B',
    'Red': '#FF0000'
}


## Binomial Distribution

class BinomialDistribution:
    def __init__(self, n, p, size=1, colors=colors):
        self.n = n
        self.p = p
        self.size = size
        self.colors = colors

        self.simulated_data = None
        self.simulate_binomial_dist_data()

        self.x_axis_min = 0
        self.x_axis_max = self.n + 1
        self.x_vals = np.arange(0, self.n + 1)

        self.pmf_probs = None
        self.calculate_pmf()

    def simulate_binomial_dist_data(self):
        """
        Simulate binomial distribution data based on the provided parameters.
        """
        outcomes = np.random.rand(self.size, self.n) < self.p
        self.simulated_data = np.sum(outcomes, axis=1)

    def calculate_pmf(self):
        # Calculate PMF using binomial distribution formula
        self.pmf_probs = np.array([round(math.comb(self.n, k) * (self.p**k) * ((1 - self.p)**(self.n - k)), 3) for k in self.x_vals])

    def plot_theoretical_pmf(self):
        """
        Plot the probability mass function (PMF) for the binomial distribution using Plotly.
        """
        # Define hover text
        hover_temp = '<b>Number of Successes</b>: %{x}<br><b>Probability</b>: %{customdata}'
        customize_data = [f'{round(y * 100, 2)}%' for y in self.pmf_probs]

        # Create stem plot
        fig = go.Figure()

        # Add scatter dots
        fig.add_trace(
            go.Scatter(
                x=self.x_vals, y=self.pmf_probs, 
                mode='markers', marker=dict(color=self.colors['USF_Green']),
                hovertemplate=hover_temp, name='', customdata=customize_data,
                hoverlabel=dict(font=dict(color='white'))))

        # Add vertical stems
        for i in range(len(self.x_vals)):
            fig.add_shape(
                type='line',
                x0=self.x_vals[i], y0=0,
                x1=self.x_vals[i], y1=self.pmf_probs[i],
                line=dict(color=self.colors['USF_Green'], width=2))

        # Set layout
        if max(self.x_vals) <= 30:
            tickvals = self.x_vals
            tickmode = 'array'
        else:
            tickvals = None
            tickmode = 'auto'

        fig.update_layout(
            title="<span style='font-size:18px; font-weight:bold;'>Binomial Distribution Theoretical PMF</span>",
            xaxis_title='Number of Successes',
            yaxis_title='Probability',
            hoverlabel=dict(font=dict(size=14), bgcolor=self.colors['USF_Green']),
            xaxis=dict(tickmode=tickmode, tickvals=tickvals, tickangle=0))
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)

    def plot_prob_table(self):
        """
        Plot the probability table for the binomial distribution.
        """
        prob_df = pd.DataFrame({'x': self.x_vals, 'P(X=x)': self.pmf_probs})

        # Transpose DataFrame
        transposed_df = prob_df.T

        # Render DataFrame as HTML table with row names but without column names
        html_table = transposed_df.to_html(header=False, index=True)

        # Wrap HTML table within a <div> element with a fixed width and horizontal scrollbar
        html_with_scrollbar = f'<div style="overflow-x:auto;">{html_table}</div>'

        # Display HTML table with horizontal scrollbar
        st.write(html_with_scrollbar, unsafe_allow_html=True)

    def plot_empirical_pmf(self):
        """
        Plot the simulation results using Plotly.
        """
        # Count the occurrences of each value
        unique, count = np.unique(self.simulated_data, return_counts=True)

        y_vals = [0] * (self.n + 1)

        for u, c in zip(unique, count):
            y_vals[u] = c

        # Define hover text
        hover_temp = '<b>Number of Successes</b>: %{x}<br><b>Count</b>: %{y}<br><b>Percentage</b>: %{customdata}'
        customize_data = [f'{round(y/self.size * 100, 2)}%' for y in y_vals]

        # Create bar plot
        fig = go.Figure()

        # Add trace for bar plots
        fig.add_trace(
            trace=go.Bar(
                x=self.x_vals, y=y_vals,
                # marker=dict(color=self.colors['USF_Yellow'], opacity=0.8),
                marker=dict(
                    color=self.colors['USF_Yellow_rbga_fill'],
                    line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
                hovertemplate=hover_temp, name='', customdata=customize_data,
                hoverlabel=dict(font=dict(color='white'))))

        # Set layout
        if max(self.x_vals) <= 30:
            tickvals = self.x_vals
            tickmode = 'array'
        else:
            tickvals = None
            tickmode = 'auto'

        fig.update_layout(
            go.Layout(
                title="<span style='font-size:18px; font-weight:bold;'>Binomial Distribution Empirical PMF (Simulation)</span>",
                xaxis_title='Number of Successes',
                yaxis_title='Frequency (Count)',
                hoverlabel=dict(font=dict(size=14), bgcolor=self.colors['USF_Yellow']),
                xaxis=dict(tickmode=tickmode, tickvals=tickvals, tickangle=0)))
            
        # Show plot 
        st.plotly_chart(fig, use_container_width=True)

# The following block of code is outdated. Keeping it for reference purposes.
'''
class BinomialPMF:
    def __init__(self, n, p, color=colors['USF_Green']):
        """
        Parameters:
        - n (int): Number of trials.
        - p (float): Probability of success for each trial.
        - color (str): Color code for plots (default is '#00543C').
        """
        self.n = n
        self.p = p
        self.color = color
        self.pmf_prob = None
        self.calculate_pmf()

    def calculate_pmf(self):
        """
        Calculate the probability mass function (PMF) for the binomial distribution.
        """
        # Generate values for the random variable
        x = np.arange(0, self.n + 1)

        # Calculate PMF using binomial distribution formula
        self.pmf_prob = np.array([round(math.comb(self.n, k) * (self.p**k) * ((1 - self.p)**(self.n - k)), 3) for k in x])

    def plot_pmf(self):
        """
        Plot the probability mass function (PMF) for the binomial distribution using Plotly.
        """
        x_vals = np.arange(0, self.n + 1)
        y_vals = self.pmf_prob

        # Define hover text
        hover_temp = '<b>Number of Successes</b>: %{x}<br><b>Probability</b>: %{customdata}'
        customize_data = [f'{round(y * 100, 2)}%' for y in y_vals]

        # Create stem plot
        fig = go.Figure()

        # Add scatter dots
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals, 
                mode='markers', marker=dict(color=self.color),
                hovertemplate=hover_temp, name='', customdata=customize_data,
                hoverlabel=dict(font=dict(color='white'))))

        # Add vertical stems
        for i in range(len(x_vals)):
            fig.add_shape(
                type='line',
                x0=x_vals[i], y0=0,
                x1=x_vals[i], y1=y_vals[i],
                line=dict(color=self.color, width=2))

        # Set layout
        if max(x_vals) <= 30:
            tickvals = x_vals
            tickmode = 'array'
        else:
            tickvals = None
            tickmode = 'auto'

        fig.update_layout(
            title='Binomial PMF',
            xaxis_title='Number of Successes',
            yaxis_title='Probability',
            hoverlabel=dict(font=dict(size=14), bgcolor=self.color),
            xaxis=dict(tickmode=tickmode, tickvals=tickvals, tickangle=0))
        
        # Show plot
        st.plotly_chart(fig)

    def plot_prob_table(self):
        """
        Plot the probability table for the binomial distribution.
        """
        x = np.arange(0, self.n + 1)
        y = self.pmf_prob
        prob_df = pd.DataFrame({'x': x, 'P(X=x)': y})

        # Transpose DataFrame
        transposed_df = prob_df.T

        # Render DataFrame as HTML table with row names but without column names
        html_table = transposed_df.to_html(header=False, index=True)

        # Wrap HTML table within a <div> element with a fixed width and horizontal scrollbar
        html_with_scrollbar = f'<div style="overflow-x:auto;">{html_table}</div>'

        # Display HTML table with horizontal scrollbar
        st.write(html_with_scrollbar, unsafe_allow_html=True)


class BinomialSimulation:
    def __init__(self, n, p, size=1, color=colors['USF_Green']):
        """
        Parameters:
        - n (int): Number of trials.
        - p (float): Probability of success for each trial.
        - size (int): Number of simulations to run (default is 1).
        - color (str): Color for the plot (default is '#00543C').
        """
        self.n = n
        self.p = p
        self.size = size
        self.color = color
        self.simulated_data = None
        self.simulate_binomial_data()  # Automatically simulate data upon initialization

    def simulate_binomial_data(self):
        """
        Simulate binomial data based on the provided parameters.
        """
        outcomes = np.random.rand(self.size, self.n) < self.p
        self.simulated_data = np.sum(outcomes, axis=1)

    def plot_simulation(self):
        """
        Plot the simulation results using Plotly.
        """
        # Count the occurrences of each value
        unique, count = np.unique(self.simulated_data, return_counts=True)

        x_vals = np.arange(0, self.n + 1)
        y_vals = [0] * (self.n + 1)

        for u, c in zip(unique, count):
            y_vals[u] = c

        # Define hover text
        hover_temp = '<b>Number of Successes</b>: %{x}<br><b>Count</b>: %{y}<br><b>Percentage</b>: %{customdata}'
        customize_data = [f'{round(y/self.size * 100, 2)}%' for y in y_vals]

        # Create bar plot
        fig = go.Figure()

        # Add trace for bar plots
        fig.add_trace(
            trace=go.Bar(
                x=x_vals, y=y_vals,
                marker=dict(color=self.color),
                hovertemplate=hover_temp, name='', customdata=customize_data,
                hoverlabel=dict(font=dict(color='white'))))

        # Set layout
        if max(x_vals) <= 30:
            tickvals = x_vals
            tickmode = 'array'
        else:
            tickvals = None
            tickmode = 'auto'

        fig.update_layout(
            go.Layout(
                title='Binomial Distribution Simulation',
                xaxis_title='Number of Successes',
                yaxis_title='Frequency (Count)',
                hoverlabel=dict(font=dict(size=14), bgcolor=self.color),
                xaxis=dict(tickmode=tickmode, tickvals=tickvals, tickangle=0)))
            
        # Show plot 
        st.plotly_chart(fig) 
'''


## Poisson Distribution

class PoissonDistribution:
    def __init__(self, lmbda, size=1, colors=colors):
        self.lmbda = lmbda
        self.size = size
        self.colors = colors

        self.simulated_data = None
        self.simulate_poisson_dist_data()

        self.x_axis_min = 0
        self.x_axis_max = 21 if self.lmbda <= 10 else 31
        self.x_vals = list(range(self.x_axis_max))

        self.pmf_probs = None
        self.calculate_pmf()

    def simulate_poisson_dist_data(self):
        """
        Simulate Poisson data based on the provided parameters.
        """
        self.simulated_data = np.random.poisson(self.lmbda, self.size)

    def calculate_pmf(self):
        # Calculate PMF using Poisson distribution formula
        self.pmf_probs = np.array([round((np.exp(-self.lmbda) * self.lmbda**k) / math.factorial(k), 3) for k in self.x_vals])

    def plot_theoretical_pmf(self):
        """
        Plot the probability mass function (PMF) for the Poisson distribution using Plotly.
        """
        # Define hover text
        hover_temp = '<b>Number of Events</b>: %{x}<br><b>Probability</b>: %{customdata}'
        customize_data = [f'{round(y, 3)}' for y in self.pmf_probs]

        # Create stem plot
        fig = go.Figure()

        # Add scatter dots
        fig.add_trace(
            go.Scatter(
                x=self.x_vals, y=self.pmf_probs, 
                mode='markers', marker=dict(color=self.colors['USF_Green']),
                hovertemplate=hover_temp, name='', customdata=customize_data,
                hoverlabel=dict(font=dict(color='white'))))

        # Add vertical stems
        for i in range(len(self.x_vals)):
            fig.add_shape(
                type='line',
                x0=self.x_vals[i], y0=0,
                x1=self.x_vals[i], y1=self.pmf_probs[i],
                line=dict(color=self.colors['USF_Green'], width=2))

        # Set layout
        fig.update_layout(
            title="<span style='font-size:18px; font-weight:bold;'>Poisson Distribution Theoretical PMF</span>",
            xaxis_title='Number of Events',
            yaxis_title='Probability',
            hoverlabel=dict(font=dict(size=14), bgcolor=self.colors['USF_Green']),
            xaxis=dict(tickmode='array', tickvals=self.x_vals, tickangle=0))

        # Show plot
        st.plotly_chart(fig, use_container_width=True)

    def plot_prob_table(self):
        """
        Plot the probability table for the poisson distribution.
        """
        prob_df = pd.DataFrame({'x': self.x_vals, 'P(X=x)': self.pmf_probs})

        # Transpose DataFrame
        transposed_df = prob_df.T

        # Render DataFrame as HTML table with row names but without column names
        html_table = transposed_df.to_html(header=False, index=True)

        # Wrap HTML table within a <div> element with a fixed width and horizontal scrollbar
        html_with_scrollbar = f'<div style="overflow-x:auto;">{html_table}</div>'

        # Display HTML table with horizontal scrollbar
        st.write(html_with_scrollbar, unsafe_allow_html=True)

    def plot_empirical_pmf(self):
        """
        Plot the simulation results using Plotly.
        """
        # Count the occurrences of each value
        unique, count = np.unique(self.simulated_data, return_counts=True)

        y_vals = [0] * self.x_axis_max

        for u, c in zip(unique, count):
            if u < len(y_vals):
                y_vals[u] = c

        # Define hover text
        hover_temp = '<b>Number of Events</b>: %{x}<br><b>Count</b>: %{y}<br><b>Percentage</b>: %{customdata}'
        customize_data = [f'{round(y/self.size * 100, 2)}%' for y in y_vals]

        # Create bar plot
        fig = go.Figure()

        # Add trace for bar plots
        fig.add_trace(
            go.Bar(
                x=self.x_vals, y=y_vals,
                # marker=dict(color=self.colors['USF_Yellow'], opacity=0.8),
                marker=dict(
                    color=self.colors['USF_Yellow_rbga_fill'],
                    line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
                hovertemplate=hover_temp, name='', customdata=customize_data,
                hoverlabel=dict(font=dict(color='white'))))

        # Set layout
        fig.update_layout(
            title="<span style='font-size:18px; font-weight:bold;'>Poisson Distribution Empirical PMF (Simulation)</span>",
            xaxis_title='Number of Events',
            yaxis_title='Frequency (Count)',
            hoverlabel=dict(font=dict(size=14), bgcolor=self.colors['USF_Yellow']),
            xaxis=dict(tickmode='array', tickvals=self.x_vals, tickangle=0))

        # Show plot 
        st.plotly_chart(fig, use_container_width=True)
        
# The following block of code is outdated. Keeping it for reference purposes.
'''        
class PoissonPMF:
    def __init__(self, lmbda, color=colors['USF_Green']):
        """
        Parameters:
        - lmbda (float): Average rate of the Poisson distribution.
        - color (str): Color code for plots (default is '#00543C').
        """
        self.lmbda = lmbda
        self.color = color
        self.pmf_prob = None
        self.k_limit = 21 if lmbda <= 10 else 31
        self.calculate_pmf()


    def calculate_pmf(self):
        """
        Calculate the probability mass function (PMF) for the Poisson distribution.
        """
        # Generate values for the random variable
        x = range(self.k_limit)

        # Calculate PMF using Poisson distribution formula
        self.pmf_prob = np.array([round((np.exp(-self.lmbda) * self.lmbda**k) / math.factorial(k), 3) for k in x])

    def plot_pmf(self):
        """
        Plot the probability mass function (PMF) for the Poisson distribution using Plotly.
        """
        x_vals = list(range(self.k_limit))
        y_vals = self.pmf_prob

        # Define hover text
        hover_temp = '<b>Number of Events</b>: %{x}<br><b>Probability</b>: %{customdata}'
        customize_data = [f'{round(y, 3)}' for y in y_vals]

        # Create stem plot
        fig = go.Figure()

        # Add scatter dots
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals, 
                mode='markers', marker=dict(color=self.color),
                hovertemplate=hover_temp, name='', customdata=customize_data,
                hoverlabel=dict(font=dict(color='white'))))

        # Add vertical stems
        for i in range(len(x_vals)):
            fig.add_shape(
                type='line',
                x0=x_vals[i], y0=0,
                x1=x_vals[i], y1=y_vals[i],
                line=dict(color=self.color, width=2))

        # Set layout
        fig.update_layout(
            title='Poisson PMF',
            xaxis_title='Number of Events',
            yaxis_title='Probability',
            hoverlabel=dict(font=dict(size=14), bgcolor=self.color),
            xaxis=dict(tickmode='array', tickvals=x_vals, tickangle=0))

        # Show plot
        st.plotly_chart(fig)

    def plot_prob_table(self):
        """
        Plot the probability table for the Poisson distribution.
        """
        x = list(range(self.k_limit))
        y = self.pmf_prob
        prob_df = pd.DataFrame({'x': x, 'P(X=x)': y})

        # Transpose DataFrame
        transposed_df = prob_df.T

        # Render DataFrame as HTML table with row names but without column names
        html_table = transposed_df.to_html(header=False, index=True)

        # Wrap HTML table within a <div> element with a fixed width and horizontal scrollbar
        html_with_scrollbar = f'<div style="overflow-x:auto;">{html_table}</div>'

        # Display HTML table with horizontal scrollbar
        st.write(html_with_scrollbar, unsafe_allow_html=True)


class PoissonSimulation:
    def __init__(self, lmbda, size=1, color=colors['USF_Green']):
        """
        Parameters:
        - lmbda (float): Average rate of the Poisson distribution.
        - size (int): Number of simulations to run.
        - color (str): Color for the plot (default is '#00543C').
        """
        self.lmbda = lmbda
        self.size = size
        self.color = color
        self.simulated_data = None
        self.k_limit = 21 if lmbda <= 10 else 31
        self.simulate_poisson_data()  # Automatically simulate data upon initialization

    def simulate_poisson_data(self):
        """
        Simulate Poisson data based on the provided parameters.
        """
        self.simulated_data = np.random.poisson(self.lmbda, self.size)

    def plot_simulation(self):
        """
        Plot the simulation results using Plotly.
        """
        # Count the occurrences of each value
        unique, count = np.unique(self.simulated_data, return_counts=True)

        x_vals = list(range(self.k_limit))
        y_vals = [0] * self.k_limit

        for u, c in zip(unique, count):
            if u < len(y_vals):
                y_vals[u] = c

        # Define hover text
        hover_temp = '<b>Number of Events</b>: %{x}<br><b>Count</b>: %{y}<br><b>Percentage</b>: %{customdata}'
        customize_data = [f'{round(y/self.size * 100, 2)}%' for y in y_vals]

        # Create bar plot
        fig = go.Figure()

        # Add trace for bar plots
        fig.add_trace(
            go.Bar(
                x=x_vals, y=y_vals,
                marker=dict(color=self.color),
                hovertemplate=hover_temp, name='', customdata=customize_data,
                hoverlabel=dict(font=dict(color='white'))))

        # Set layout
        fig.update_layout(
            title='Poisson Distribution Simulation',
            xaxis_title='Number of Events',
            yaxis_title='Frequency (Count)',
            hoverlabel=dict(font=dict(size=14), bgcolor=self.color),
            xaxis=dict(tickmode='array', tickvals=x_vals, tickangle=0))

        # Show plot 
        st.plotly_chart(fig)
''' 

## Normal Distribution

class NormalDistribution:
    def __init__(self, mean, std_dev, size=1, colors=colors):
        """
        Parameters:
        - mean (float): Mean of the normal distribution.
        - std_dev (float): Standard deviation of the normal distribution.
        - size (int): Number of simulations to run (default is 1).
        """
        self.mean = mean
        self.std_dev = std_dev
        self.size = size
        self.colors = colors

        self.simulated_data = None
        self.simulate_normal_dist_data()

        self.x_axis_min = None
        self.x_axis_max = None
        self.x_vals = None
        self.generate_x_axis()

        self.pdf_vals = None
        self.calculate_pdf()

        self.cdf_vals = None
        self.calculate_cdf()

    def simulate_normal_dist_data(self):
        """
        Simulate normal distribution data based on the provided parameters.
        """
        simulated_data = np.random.normal(self.mean, self.std_dev, self.size)
        self.simulated_data = simulated_data

    def generate_x_axis(self):
        simulation_trace=go.Histogram(
            x=self.simulated_data, 
            histnorm='probability density', 
            marker_color=self.colors['USF_Green'], 
            showlegend=False)
        
        x_axis_limits = [np.min(simulation_trace.x), np.max(simulation_trace.x)]
        self.x_axis_min = x_axis_limits[0]
        self.x_axis_max = x_axis_limits[1]
        self.x_vals = np.linspace(self.x_axis_min, self.x_axis_max, 1000)

    def calculate_pdf(self):
        """
        Calculate the probability density function (PDF) for the normal distribution.
        """
        self.pdf_vals = (1 / (self.std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((self.x_vals - self.mean) / self.std_dev) ** 2)

    def calculate_cdf(self):
        """
        Calculate the cumulative distribution function (CDF) for the normal distribution.
        """
        self.cdf_vals = 0.5 * (1 + np.array([math.erf((val - self.mean) / (self.std_dev * np.sqrt(2))) for val in self.x_vals]))

    # The following block of code is outdated. Keeping it for reference purposes.
    """
    def plot_simulation_pdf_cdf(self):
        # Trace of histogram for simulated data
        simulation_trace=go.Histogram(
            x=self.simulated_data, 
            histnorm='probability density', 
            name='Simulation',
            marker_color=self.colors['USF_Green'], 
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for PDF
        pdf_trace=go.Scatter(
            x=self.x_vals, y=self.pdf_vals, 
            mode='lines', name='PDF', 
            line=dict(color=self.colors['USF_Yellow'], width=3), 
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for CDF
        cdf_trace=go.Scatter(
                x=self.x_vals, y=self.cdf_vals, 
                mode='lines', name='CDF', 
                line=dict(color=self.colors['USF_Gray'], width=3), 
                showlegend=False,
                hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        pdf_trace.visible = True
        cdf_trace.visible = False

        # Create layout with updatemenus
        layout = go.Layout(
            title='Normal Distribution Simulation with PDF',
            updatemenus=[{
                'buttons': [{
                    'args': [{'visible': [True, False, False]}, {'title': 'Normal Distribution Simulation'}],
                    'label': 'Simulated Data Only',
                    'method': 'update'
                    }, {
                    'args': [{'visible': [True, True, False]}, {'title': 'Normal Distribution Simulation with PDF'}],
                    'label': 'Overlay PDF',
                    'method': 'update'
                    }, {
                        'args': [{'visible': [True, False, True]}, {'title': 'Normal Distribution Simulation with CDF'}],
                        'label': 'Overlay CDF',
                        'method': 'update'
                    }, {
                        'args': [{'visible': [True, True, True]}, {'title': 'Normal Distribution Simulation with PDF & CDF'}],
                        'label': 'Overlay Both',
                        'method': 'update'
                    }],
                'type': 'buttons',
                'direction': 'down',
                'showactive': False,
                'x': -0.35,
                'xanchor': 'left', 'yanchor': 'top'}])
        
        # Create figure
        fig = go.Figure(data=[simulation_trace, pdf_trace, cdf_trace], layout=layout)
        
        # Show plot
        st.plotly_chart(fig)
    """

    def plot_pdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data,
            histnorm='probability density',
            name='Empirical PDF',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for PDF
        pdf_trace = go.Scatter(
            x=self.x_vals, y=self.pdf_vals,
            mode='lines', name='Theoretical PDF',
            line=dict(color=self.colors['USF_Green'], width=3),
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        pdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Normal Distribution Simulation (PDF)</span>",
            xaxis_title='Values that x can take on',
            yaxis_title='Probability Density',
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[simulation_trace, pdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)

    def plot_cdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data, 
            histnorm='probability density',
            cumulative_enabled=True,
            name='Empirical CDF',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for CDF
        cdf_trace = go.Scatter(
            x=self.x_vals, y=self.cdf_vals,
            mode='lines', name='Theoretical CDF',
            line=dict(color=self.colors['USF_Green'], width=3),
            hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        cdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Normal Distribution Simulation (CDF)</span>",
            xaxis_title='Values that x can take on',
            yaxis_title='Cumulative Probability',
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[simulation_trace, cdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)


## Uniform Distribution

class UniformDistribution:
    def __init__(self, lower, upper, size=1, colors=colors):
        """
        Parameters:
        - lower (float): Lower bound of the uniform distribution.
        - upper (float): Upper bound of the uniform distribution.
        - size (int): Number of simulations to run (default is 1).
        """
        self.lower = lower
        self.upper = upper
        self.size = size
        self.colors = colors

        self.simulated_data = None
        self.simulate_uniform_dist_data()

        self.x_axis_min = None
        self.x_axis_max = None
        self.x_vals = None
        self.generate_x_axis()

        self.pdf_vals = None 
        self.generate_pdf_vals()

        self.cdf_vals = None
        self.generate_cdf_vals()

    def simulate_uniform_dist_data(self):
        """
        Simulate uniform distribution data based on the provided parameters.
        """
        simulated_data = np.random.uniform(self.lower, self.upper, self.size)
        self.simulated_data = simulated_data

    def generate_x_axis(self):
        self.x_axis_min = self.lower - (self.upper - self.lower)*0.3
        self.x_axis_max = self.upper + (self.upper - self.lower)*0.3
        self.x_vals = np.linspace(self.x_axis_min, self.x_axis_max, 1000)

    def generate_pdf_vals(self):
        # Initialize y_vals with zeros
        y_vals = np.zeros_like(self.x_vals)

        # Set y_vals to 1/(upper-lower) where x_vals is between lower and upper
        mask = np.logical_and(self.x_vals >= self.lower, self.x_vals <= self.upper)
        y_vals[mask] = 1 / (self.upper - self.lower)

        self.pdf_vals = y_vals

    def generate_cdf_vals(self):
        # Initialize y_vals with zeros
        y_vals = np.zeros_like(self.x_vals)

        # Set y_vals to 1 where x_vals is larger than upper
        mask1 = self.x_vals > self.upper
        y_vals[mask1] = 1

        # Set y_vals to be linearly spaced out between 0 and 1 where x_vals is between lower and upper
        mask2 = np.logical_and(self.x_vals >= self.lower, self.x_vals <= self.upper)
        y_vals[mask2] = np.interp(self.x_vals[mask2], [self.lower, self.upper], [0, 1])

        self.cdf_vals = y_vals

    # The following block of code is outdated. Keeping it for reference purposes.
    """
    def plot_simulation_pdf_cdf(self):
        # Trace of histogram for simulated data
        simulation_trace=go.Histogram(
            x=self.simulated_data, 
            histnorm='probability density', 
            name='Simulation',
            marker_color=self.colors['USF_Green'], 
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')
        
        # Trace of line plot for PDF
        pdf_trace=go.Scatter(
            x=self.x_vals, y=self.pdf_vals, 
            mode='lines', name='PDF', 
            line=dict(color=self.colors['USF_Yellow'], width=3), 
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')
        
        # Trace of line plot for CDF
        cdf_trace=go.Scatter(
            x=self.x_vals, y=self.cdf_vals, 
            mode='lines', name='CDF', 
            line=dict(color=self.colors['USF_Gray'], width=3), 
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        pdf_trace.visible = True
        cdf_trace.visible = False

        # Create layout with updatemenus
        layout = go.Layout(
            title='Uniform Distribution Simulation with PDF',
            xaxis=dict(range=[self.x_axis_min, self.x_axis_max]),
            updatemenus=[{
                'buttons': [{
                    'args': [{'visible': [True, False, False]}, {'title': 'Uniform Distribution Simulation'}],
                    'label': 'Simulated Data Only',
                    'method': 'update'
                    }, {
                    'args': [{'visible': [True, True, False]}, {'title': 'Uniform Distribution Simulation with PDF'}],
                    'label': 'Overlay PDF',
                    'method': 'update'
                    }, {
                        'args': [{'visible': [True, False, True]}, {'title': 'Uniform Distribution Simulation with CDF'}],
                        'label': 'Overlay CDF',
                        'method': 'update'
                    }, {
                        'args': [{'visible': [True, True, True]}, {'title': 'Uniform Distribution Simulation with PDF & CDF'}],
                        'label': 'Overlay Both',
                        'method': 'update'
                    }],
                'type': 'buttons',
                'direction': 'down',
                'showactive': False,
                'x': -0.35,
                'xanchor': 'left', 'yanchor': 'top'}])
        
        # Create figure
        fig = go.Figure(data=[simulation_trace, pdf_trace, cdf_trace], layout=layout)
        
        # Show plot
        st.plotly_chart(fig)
        """

    def plot_pdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data,
            histnorm='probability density',
            name='Empirical PDF',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for PDF
        pdf_trace = go.Scatter(
            x=self.x_vals, y=self.pdf_vals,
            mode='lines', name='Theoretical PDF',
            line=dict(color=self.colors['USF_Green'], width=3),
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        pdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Uniform Distribution Simulation (PDF)</span>",
            xaxis_title='Values that x can take on',
            yaxis_title='Probability Density',
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[simulation_trace, pdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)

    def plot_cdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data, 
            histnorm='probability density',
            cumulative_enabled=True,
            name='Empirical CDF',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for CDF
        cdf_trace = go.Scatter(
            x=self.x_vals, y=self.cdf_vals,
            mode='lines', name='Theoretical CDF',
            line=dict(color=self.colors['USF_Green'], width=3),
            hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        cdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Uniform Distribution Simulation (CDF)</span>",
            xaxis_title='Values that x can take on',
            yaxis_title='Cumulative Probability',
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[simulation_trace, cdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)

## Exponential Distribution

class ExponentialDistribution:
    def __init__(self, rate, size=1, colors=colors):
        """
        Parameters:
        - rate (float): Rate parameter of the exponential distribution.
        - size (int): Number of simulations to run (default is 1).
        """
        self.rate = rate
        self.size = size
        self.colors = colors

        self.simulated_data = None
        self.simulate_exponential_dist_data()

        self.x_axis_min = None
        self.x_axis_max = None
        self.x_vals = None
        self.generate_x_axis()

        self.pdf_vals = None
        self.calculate_pdf()

        self.cdf_vals = None
        self.calculate_cdf()

    def simulate_exponential_dist_data(self):
        """
        Simulate exponential distribution data based on the provided parameters.
        """
        simulated_data = np.random.exponential(scale=1/self.rate, size=self.size)
        self.simulated_data = simulated_data

    def generate_x_axis(self):
        self.x_axis_min = 0
        self.x_axis_max = np.max(self.simulated_data)
        self.x_vals = np.linspace(self.x_axis_min, self.x_axis_max, 1000)

    def calculate_pdf(self):
        """
        Calculate the probability density function (PDF) for the exponential distribution.
        """
        self.pdf_vals = self.rate * np.exp(-self.rate * self.x_vals)

    def calculate_cdf(self):
        """
        Calculate the cumulative distribution function (CDF) for the exponential distribution.
        """
        self.cdf_vals = 1 - np.exp(-self.rate * self.x_vals)

    # The following block of code is outdated. Keeping it for reference purposes.
    """
    def plot_simulation_pdf_cdf(self):
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

        # Trace of line plot for CDF
        cdf_trace = go.Scatter(
            x=self.x_vals, y=self.cdf_vals,
            mode='lines', name='CDF',
            line=dict(color=self.colors['USF_Gray'], width=3),
            showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        pdf_trace.visible = True
        cdf_trace.visible = False

        # Create layout with updatemenus
        layout = go.Layout(
            title='Exponential Distribution Simulation with PDF',
            updatemenus=[{
                'buttons': [{
                    'args': [{'visible': [True, False, False]}, {'title': 'Exponential Distribution Simulation'}],
                    'label': 'Simulated Data Only',
                    'method': 'update'
                }, {
                    'args': [{'visible': [True, True, False]}, {'title': 'Exponential Distribution Simulation with PDF'}],
                    'label': 'Overlay PDF',
                    'method': 'update'
                }, {
                    'args': [{'visible': [True, False, True]}, {'title': 'Exponential Distribution Simulation with CDF'}],
                    'label': 'Overlay CDF',
                    'method': 'update'
                }, {
                    'args': [{'visible': [True, True, True]}, {'title': 'Exponential Distribution Simulation with PDF & CDF'}],
                    'label': 'Overlay Both',
                    'method': 'update'
                }],
                'type': 'buttons',
                'direction': 'down',
                'showactive': False,
                'x': -0.35,
                'xanchor': 'left', 'yanchor': 'top'}])

        # Create figure
        fig = go.Figure(data=[simulation_trace, pdf_trace, cdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig)
    """

    def plot_pdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data,
            histnorm='probability density',
            name='Empirical PDF',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            # showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for PDF
        pdf_trace = go.Scatter(
            x=self.x_vals, y=self.pdf_vals,
            mode='lines', name='Theoretical PDF',
            line=dict(color=self.colors['USF_Green'], width=3),
            # showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        pdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Exponential Distribution Simulation (PDF)</span>",
            xaxis_title='Values that x can take on',
            yaxis_title='Probability Density',
            # # disable updatemenus
            # updatemenus=[{
            #     'buttons': [{
            #         'args': [{'visible': [True, False]}],
            #         'label': 'Empirical PDF',
            #         'method': 'update',
            #     }, {
            #         'args': [{'visible': [True, True]}],
            #         'label': 'Overlay Theoretical PDF',
            #         'method': 'update'
            #     }],
            #     'type': 'buttons',
            #     'direction': 'down',
            #     'showactive': False,
            #     'x': -0.35,
            #     'xanchor': 'left', 'yanchor': 'top'}],
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[simulation_trace, pdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)

    def plot_cdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data, 
            histnorm='probability density',
            cumulative_enabled=True,
            name='Empirical CDF',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            # showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for CDF
        cdf_trace = go.Scatter(
            x=self.x_vals, y=self.cdf_vals,
            mode='lines', name='Theoretical CDF',
            line=dict(color=self.colors['USF_Green'], width=3),
            # showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        cdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Exponential Distribution Simulation (CDF)</span>",
            xaxis_title='Values that x can take on',
            yaxis_title='Cumulative Probability',
            # # disable updatemenus
            # updatemenus=[{
            #     'buttons': [{
            #         'args': [{'visible': [True, False]}],
            #         'label': 'Empirical CDF',
            #         'method': 'update'
            #     }, {
            #         'args': [{'visible': [True, True]}],
            #         'label': 'Overlay Theoretical CDF',
            #         'method': 'update'
            #     }],
            #     'type': 'buttons',
            #     'direction': 'down',
            #     'showactive': False,
            #     'x': -0.35,
            #     'xanchor': 'left', 'yanchor': 'top'}],
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[simulation_trace, cdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)


## Gamma Distribution

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
            name='Empirical PDF',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            # showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for PDF
        pdf_trace = go.Scatter(
            x=self.x_vals, y=self.pdf_vals,
            mode='lines', 
            name='Theoretical PDF',
            line=dict(color=self.colors['USF_Green'], width=3),
            # showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        pdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Gamma Distribution Simulation (PDF)</span>",
            xaxis_title='Values that x can take on',
            yaxis_title='Probability Density',
            # # disable updatemenus
            # updatemenus=[{
            #     'buttons': [{
            #         'args': [{'visible': [True, False]}],
            #         'label': 'Empirical PDF',
            #         'method': 'update',
            #     }, {
            #         'args': [{'visible': [True, True]}],
            #         'label': 'Overlay Theoretical PDF',
            #         'method': 'update'
            #     }],
            #     'type': 'buttons',
            #     'direction': 'down',
            #     'showactive': False,
            #     'x': -0.35,
            #     'xanchor': 'left', 'yanchor': 'top'}],
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[simulation_trace, pdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)

    def plot_cdfs(self):
        # Trace of histogram for simulated data
        simulation_trace = go.Histogram(
            x=self.simulated_data, 
            histnorm='probability density',
            cumulative_enabled=True,
            name='Empirical CDF',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            # showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>f(x)</b>: %{y}')

        # Trace of line plot for CDF
        cdf_trace = go.Scatter(
            x=self.x_vals, y=self.cdf_vals,
            mode='lines', name='Theoretical CDF',
            line=dict(color=self.colors['USF_Green'], width=3),
            # showlegend=False,
            hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y}')

        # Set initial visibility
        simulation_trace.visible = True
        cdf_trace.visible = True

        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Gamma Distribution Simulation (CDF)</span>",
            xaxis_title='Values that x can take on',
            yaxis_title='Cumulative Probability',
            # # disable updatemenus
            # updatemenus=[{
            #     'buttons': [{
            #         'args': [{'visible': [True, False]}],
            #         'label': 'Empirical CDF',
            #         'method': 'update'
            #     }, {
            #         'args': [{'visible': [True, True]}],
            #         'label': 'Overlay Theoretical CDF',
            #         'method': 'update'
            #     }],
            #     'type': 'buttons',
            #     'direction': 'down',
            #     'showactive': False,
            #     'x': -0.35,
            #     'xanchor': 'left', 'yanchor': 'top'}],
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[simulation_trace, cdf_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)


## Central Limit Therom
        
class CentralLimitTheorm():
    def __init__(self, population_data, sample_size, n_samples, colors=colors):
        self.population_data = population_data 
        self.sample_size = sample_size 
        self.n_samples = n_samples
        self.colors = colors

        self.sample_means = None
        self.generate_sample_means()
        self.mean_of_sample_means = np.mean(self.sample_means)

    def generate_sample_means(self):
        # Randomly select "sample_size" of observations from the population data for "n_samples" times
        self.sample_means = [
            np.mean(random.sample(self.population_data.tolist(), self.sample_size)) for _ in range(self.n_samples)]

    def plot_clt_sample_mean(self):
        sample_means_trace = go.Histogram(
            x=self.sample_means,
            name='Sample Means',
            # marker_color=self.colors['USF_Yellow'],
            # opacity=0.8,
            marker=dict(
                color=self.colors['USF_Yellow_rbga_fill'],
                line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)),
            showlegend=True)
        
        # Create layout with updatemenus
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Sample Means Distribution</span>",
            xaxis_title='Sample Mean',
            yaxis_title='Frequency (Count)',
            legend=dict(
                orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[sample_means_trace], layout=layout)

        # Add vertical line for the mean of sampling distribution
        fig.add_vline(
            x=self.mean_of_sample_means, 
            line={'color': self.colors['USF_Gray'], 'dash': 'dash', 'width': 2.5},
            annotation_text=f"Mean: {self.mean_of_sample_means:.3f}", 
            annotation_font_size=16,
            annotation_font_color=self.colors['USF_Gray'],
            annotation_position="top")

        # Show plot
        st.plotly_chart(fig, use_container_width=True)


## Multimodal Distribution
        
class MultimodalDistribution():
    def __init__(self, dist1_data, dist2_data, colors=colors):
        self.dist1_data = dist1_data
        self.dist2_data = dist2_data
        self.colors = colors

    def plot_distribution(self):
        # Generate multiple modes dataset
        data = np.concatenate([self.dist1_data, self.dist2_data])

        # Create a histogram to estimate the density
        hist, bins = np.histogram(data, bins=100, density=True)

        # Create a histogram trace
        hist_trace = go.Bar(
            x=bins, y=hist, 
            name='Histogram',
            marker=dict(
                    color=self.colors['USF_Yellow_rbga_fill'],
                    line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1)))

        # Calculate kernel density estimate
        kde = sns.kdeplot(data, bw_adjust=0.5)

        # Create a kernel density estimate trace
        kde_trace = go.Scatter(
            x=kde.get_lines()[0].get_data()[0],
            y=kde.get_lines()[0].get_data()[1],
            mode='lines',
            name='Kernel Density Estimate',
            marker_color=self.colors['USF_Green'])

        # Create layout
        layout = go.Layout(
            title="<span style='font-size:18px; font-weight:bold;'>Multimodal Distribution</span>",
            xaxis=dict(title='Value'),
            yaxis=dict(title='Density'),
            legend=dict(
                # orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[hist_trace, kde_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)


## Bivariate Normal Distribution

class BivariateNormalDistribution:
    def __init__(self, mean_x, mean_y, std_x, std_y, rho, colors=colors):
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.std_x = std_x
        self.std_y = std_y
        self.rho = rho
        self.colors = colors

        self.x = np.linspace(-5, 5, 100)
        self.y = np.linspace(-5, 5, 100)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = None
        self.calculate_height()

    def calculate_height(self):
        self.Z = np.exp(
            -(
                ((self.X - self.mean_x) ** 2) / (2 * (self.std_x ** 2))
                + ((self.Y - self.mean_y) ** 2) / (2 * (self.std_y ** 2))
                - (2 * self.rho * (self.x - self.mean_x) * (self.y - self.mean_y)) / (self.std_x * self.std_y)
            )
            / (2 * (1 - self.rho ** 2))
        ) / (2 * np.pi * self.std_x * self.std_y * np.sqrt(1 - self.rho ** 2))

    def plot_bivariate_normal_3D(self):
        surface_plot = go.Surface(
            x=self.x,
            y=self.y,
            z=self.Z,
            colorscale='greys',  # blues, greys, portland
            contours=dict(z=dict(show=True, usecolormap=True)))
        
        # Generate traces for marginal distributions
        x_values = np.linspace(-5, 5, 100)
        y_values = np.linspace(-5, 5, 100)
        x_pdf = (1 / (self.std_x * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - self.mean_x) / self.std_x) ** 2)
        y_pdf = (1 / (self.std_y * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_values - self.mean_y) / self.std_y) ** 2)
        
        x_trace = go.Scatter3d(
            x=x_values,
            y=np.zeros_like(x_values) - 5,
            z=x_pdf,
            mode='lines',
            name='X Distribution',
            line=dict(color=self.colors['USF_Green'], width=4))
        
        y_trace = go.Scatter3d(
            x=np.zeros_like(y_values) - 5,
            y=y_values,
            z=y_pdf,
            mode='lines',
            name='Y Distribution',
            line=dict(color=self.colors['USF_Yellow'], width=4))
        
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Probability Density"),
            ),
            margin=dict(l=0, r=0, t=0, b=0))

        fig = go.Figure(data=[surface_plot, x_trace, y_trace], layout=layout)

        # Hide legend for 'X Distribution' and 'Y Distribution'
        fig.data[1].update(showlegend=False)
        fig.data[2].update(showlegend=False)
        
        # Show plot
        st.plotly_chart(fig, use_container_width=True)


## Hypothesis Test
        
def calculate_critical_value(hypothesis, alpha, distribution, df=None):
    if hypothesis == 'two-tailed':
        # For two-tailed test, divide alpha by 2
        alpha /= 2

    if distribution == 'z':
        # Calculate the z-score for the given alpha
        critical_val = round(stats.norm.ppf(1 - alpha), 2)
    elif distribution == 't':
        # Check if degree of freedom is provided
        if df is None:
            raise ValueError("Degrees of freedom 'df' must be provided for t-distribution.")
        # Calculate the t-score for the given alpha and degrees of freedom
        critical_val = round(stats.t.ppf(1 - alpha, df), 2)
    elif distribution == 'chi-square':
        # Check if degree of freedom is provided
        if df is None:
            raise ValueError("Degrees of freedom 'df' must be provided for Chi-square distribution.")
        # Calculate the chi-square critical value for the given alpha and degrees of freedom
        if hypothesis == 'two-tailed':
            critical_val_left = round(stats.chi2.ppf(alpha, df), 2)
            critical_val_right = round(stats.chi2.ppf(1 - alpha, df), 2)
        elif hypothesis == 'left-tailed':
            critical_val_left = round(stats.chi2.ppf(alpha, df), 2)
        elif hypothesis == 'right-tailed':
            critical_val_right = round(stats.chi2.ppf(1 - alpha, df), 2)

    if distribution == 'chi-square':
        if hypothesis == 'two-tailed':
            critical_vals = [critical_val_left, critical_val_right]
        elif hypothesis == 'left-tailed':
            critical_vals = [critical_val_left]
        elif hypothesis == 'right-tailed':
            critical_vals = [critical_val_right]
    else:
        if hypothesis == 'two-tailed':
            critical_vals = [-critical_val, critical_val]
        elif hypothesis == 'left-tailed':
            critical_vals = [-critical_val]
        elif hypothesis == 'right-tailed':
            critical_vals = [critical_val]

    return critical_vals


def calculate_p_value(hypothesis, distribution, test_statistic, df=None):
    if distribution == 'z':
        if hypothesis == 'two-tailed':
            p_value = stats.norm.sf(abs(test_statistic))*2
        else:
            p_value = stats.norm.sf(abs(test_statistic))
    elif distribution == 't':
        # Check if degree of freedom is provided
        if df == None:
            raise ValueError("Degrees of freedom 'df' must be provided for t-distribution.")
        
        if hypothesis == 'two-tailed':
            p_value = stats.t.sf(abs(test_statistic), df=df)*2
        else:
            p_value = stats.t.sf(abs(test_statistic), df=df)
    elif distribution == 'chi-square':
        # Check if degree of freedom is provided
        if df == None:
            raise ValueError("Degrees of freedom 'df' must be provided for 'Chi-square distribution.")
        
        if hypothesis == 'two-tailed':
            p_value = stats.chi2.sf(test_statistic, df=df)*2
        else:
            p_value = stats.chi2.sf(test_statistic, df=df)

    return p_value


class HypothesisTest():
    def __init__(self, hypothesis, distribution, critical_vals, test_statistic, df=None, colors=colors):
        self.hypothesis = hypothesis
        self.critical_vals = critical_vals
        self.test_statistic = test_statistic
        self.colors = colors

        # Check if hypothesis is either 'two-tailed' or 'left-tailed' or 'right-tailed'
        if hypothesis not in ['two-tailed', 'left-tailed',  'right-tailed']:
            raise ValueError("Invalid distribution type. It can only be either 'two-tailed' or 'left-tailed' or 'right-tailed'.")
        self.distribution = distribution

        # Check if distribution is either 'z' or 't' or 'chi-square'
        if distribution not in ['z', 't', 'chi-square']:
            raise ValueError("Invalid distribution type. Use 'z' for Z-distribution, 't' for t-distribution, or 'chi-square' or 'Chi-square distribution'.")
        self.distribution = distribution

        # Check if degree of freedom is provided
        if distribution in ['t', 'chi-square'] and df == None:
            raise ValueError("Degrees of freedom 'df' must be provided for t-distribution or 'Chi-square distribution'.")
        self.df = df

        self.x = None
        self.y = None
        self.generate_distribution_data()
    
    def generate_distribution_data(self):
        if self.distribution == 'z':
            # Generate data for standard normal distribution
            self.x = np.linspace(-4, 4, 1000)
            self.y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.x**2)
        elif self.distribution == 't':
            # Generate data for t-distribution
            self.x = np.linspace(-4, 4, 1000)
            self.y = stats.t.pdf(self.x, self.df)
        elif self.distribution == 'chi-square':
            # Generate data for Chi-square distribution
            self.x = np.linspace(0, 20, 1000)  # Chi-square is defined for x >= 0
            self.y = stats.chi2.pdf(self.x, self.df)

    def plot_hypothesis(self):
        # Create a line plot for the relevant distribution
        fig = go.Figure()

        if self.distribution == 'z':
            name = 'Standard Normal Distribution (Z-Distribution)'
        elif self.distribution == 't':
            name = 't-Distribution'
        elif self.distribution == 'chi-square':
            name = 'Chi-square Distribution'

        fig.add_trace(go.Scatter(
            x=self.x, y=self.y, mode='lines', 
            marker=dict(color=self.colors['USF_Gray']),
            name=name))

        fig.update_layout(
            title=dict(
                text=f"<span style='font-size:18px; font-weight:bold;'>{name}</span>",
                y=0.95),
            yaxis_title='Probability Density',
            legend=dict(
                orientation='h', font=dict(size=14),
                yanchor='top', y=1.2, xanchor='center', x=0.5))
        
        for c in self.critical_vals: 
            # Add vertical line to split rejection and non-rejection region
            fig.add_shape(
                type='line',
                x0=c, y0=0, x1=c, y1=0.3,
                line=dict(color=self.colors['USF_Yellow'], width=3, dash='dash'))
            
            fig.add_annotation(
                x=c, y=0.36, 
                text=f'Critical Value', 
                showarrow=False, 
                font=dict(size=16, color=self.colors['USF_Yellow']))
            
            fig.add_annotation(
                x=c, y=0.33, 
                text=f'{c}', 
                showarrow=False, 
                font=dict(size=16, color=self.colors['USF_Yellow']))
            
            # Highlight the rejection region
            if c < 0:
                x_partial = self.x[self.x <= c]
                y_partial = self.y[:len(x_partial)]

            else:
                x_partial = self.x[self.x >= c]
                y_partial = self.y[-len(x_partial):]

            # Fill out the rejection region
            fig.add_trace(
                go.Scatter(
                    x=x_partial, y=y_partial, 
                    fill='tozeroy', 
                    fillcolor=self.colors['USF_Yellow_rbga_fill'], 
                    line=dict(color=self.colors['USF_Yellow_rbga_fill']), name='Rejection Region'))
            
        # Fill out the non-rejection region
        if len(self.critical_vals) == 2:
            x_partial = self.x[(self.x >= -c) & (self.x <= c)]
            y_partial = self.y[(self.x >= -c) & (self.x <= c)]
        else:
            if self.critical_vals[0] < 0:
                x_partial = self.x[self.x >= c]
                y_partial = self.y[-len(x_partial):]
            else:
                x_partial = self.x[self.x <= c]
                y_partial = self.y[:len(x_partial)]

        fig.add_trace(
            go.Scatter(
                x=x_partial, y=y_partial, 
                fill='tozeroy', 
                fillcolor=self.colors['USF_Green_rbga_fill'], 
                line=dict(color=self.colors['USF_Green_rbga_fill']), name='Non-Rejection Region'))
        
        # Add vertical line of test statistic
        fig.add_shape(
            type='line',
            x0=self.test_statistic, y0=0, x1=self.test_statistic, y1=0.3,
            line=dict(color=self.colors['Red'], width=3))
        
        fig.add_annotation(
            x=self.test_statistic, y=0.36, 
            text=f'Test Statistic', 
            showarrow=False, 
            font=dict(size=16, color=self.colors['Red']))
        
        fig.add_annotation(
            x=self.test_statistic, y=0.33, 
            text=f'{self.test_statistic}', 
            showarrow=False, 
            font=dict(size=16, color=self.colors['Red']))
        
        st.plotly_chart(fig, use_container_width=True)

    def generate_conclusion(self):
        conclusion = 'Reject the Null Hypothesis' if abs(self.test_statistic) > abs(self.critical_vals[0]) \
            else 'Do Not Reject the Null Hypothesis'

        if conclusion == 'Reject the Null Hypothesis':
            detail_1 = '| Test Statistic | > | Critical Value |'
            detail_2 = 'P-Value < Significance Level'
        else:
            detail_1 = '| Test Statistic | <= | Critical Value |'
            detail_2 = 'P-Value >= Significance Level'

        # Add a markdown section 
        st.markdown(
            """
            <style>
            .section {
                background-color: #f0f0f0; 
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            .content-text {
                font-size: 16px; 
            }
            .title-text {
                font-size: 20px; font-weight:bold;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Add conclusion text inside the section
        st.markdown(
            f'<div class="section"><div class="title-text">{conclusion}</div></div>',
            unsafe_allow_html=True)
        
        st.markdown(
            f'<div class="section"><div class="content-text">{detail_1}</div></div>',
            unsafe_allow_html=True)
        
        st.markdown(
            f'<div class="section"><div class="content-text">{detail_2}</div></div>',
            unsafe_allow_html=True)


def section_1_hypothesis(parameter, key):
    # Check if the parameter to compare is either 'mean' or 'proportion' or 'variance'
    if parameter not in ['mean', 'proportion', 'variance']:
        raise ValueError("This app only support the hypothesis test for either 'mean' or 'proportion' or 'variance'.")
    
    if parameter == 'mean':
        param = '\mu'
    if parameter == 'proportion':
        param = 'p'
    if parameter == 'variance':
        param = '\sigma^2'
    
    hypo_options = [
        f"$$H_0: {param} = {param}_0 \quad H_a: {param} \\neq {param}_0 \quad$$ (two-tailed)",
        f"$$H_0: {param} = {param}_0 \quad H_a: {param} > {param}_0 \quad$$ (one-tailed, right-tailed)",
        f"$$H_0: {param} = {param}_0 \quad H_a: {param} < {param}_0 \quad$$ (one-tailed, left-tailed)"
    ]
        
    st.write(
        "<span style='font-size:18px; font-weight:bold;'>1. Hypothesis</span>", 
        unsafe_allow_html=True)

    hypothesis = st.radio(
        'hypothesis', hypo_options, horizontal=True, 
        label_visibility='collapsed',
        key=f'hypo_{key}')
    
    if 'two-tailed' in hypothesis:
        hypothesis = 'two-tailed'
    elif 'right-tailed' in hypothesis:
        hypothesis = 'right-tailed'
    elif 'left-tailed' in hypothesis:
        hypothesis = 'left-tailed'

    st.write('')

    return hypothesis


def section_2_significance_level(key):
    st.write(
        "<span style='font-size:18px; font-weight:bold;'>2. Significance Level</span>", 
        unsafe_allow_html=True)
    
    alpha = st.radio(
        'alpha', ['0.05 (5%)', '0.01 (1%)', '0.10 (10%)'], 
        horizontal=True, 
        label_visibility='collapsed',
        key=f'sig_{key}')
    
    alpha = float(alpha.split(' ')[0])

    st.write('')

    return alpha


def section_4_critical_value(critical_vals):
    highlighted_criticals = \
        ", ".join(f"<span style='background-color: rgba(253, 187, 48, 0.4);'>{c}</span>" for c in critical_vals)
    
    st.write(
        f"<span style='font-size:18px; font-weight:bold;'>4. Critical Value: {highlighted_criticals}</span>", 
        unsafe_allow_html=True)
    
    st.write()


def section_5_p_value(p_value):
    st.write(
        f"<span style='font-size:18px; font-weight:bold;'>5. P-Value: {round(p_value, 5)}</span>", 
        unsafe_allow_html=True)

    st.write()


## Bootstrapping
    
class Bootstrapping():
    def __init__(self, original_sample, n_resamplings, statistic_text, alpha, colors=colors):
        self.original_sample = original_sample
        self.n_resamplings = n_resamplings
        self.statistic_text = statistic_text
        self.statistic = np.mean if statistic_text == 'mean' else np.median
        self.alpha = alpha
        self.colors = colors

        self.sample_size = len(self.original_sample)

        self.bootstrap_samples = None
        self.stat_estimate = None
        self.lower_bound = None
        self.upper_bound = None
        self.bootstrap()


    def bootstrap(self):
        self.bootstrap_samples = []
        self.stats = []
        for _ in range(self.n_resamplings):
            bootstrap_sample = np.random.choice(self.original_sample, size=self.sample_size, replace=True)
            self.bootstrap_samples.append(bootstrap_sample)

            stat = self.statistic(bootstrap_sample)
            self.stats.append(stat)

            self.stat_estimate = np.mean(self.stats)
            self.lower_bound = np.percentile(self.stats, 100 * self.alpha / 2)
            self.upper_bound = np.percentile(self.stats, 100 * (1 - self.alpha / 2))

    def plot_sampling_distribution(self):
        # Create histogram 
        fig = go.Figure()

        # Add trace for the histogram of bootstrapped statistics
        fig.add_trace(
            trace=go.Histogram(
                x=self.stats, nbinsx=30,
                marker=dict(
                    color=self.colors['USF_Yellow_rbga_fill'],
                    line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1))))

        # Add vertical line for the estimated statistic
        fig.add_shape(
            type='line',
            x0=self.stat_estimate, y0=0, x1=self.stat_estimate, y1=1,
            xref='x', yref='y domain',
            line=dict(color=self.colors['Red'], width=3))
        
        fig.add_annotation(
            x=self.stat_estimate, y=1.2, 
            xref='x', yref='y domain',
            text=f'Estimated {self.statistic_text}<br>{round(self.stat_estimate, 2)}', 
            showarrow=False, 
            font=dict(size=14, color=self.colors['Red']))
        
        
        # Add vertical line for the confidence interval
        fig.add_shape(
            type='line',
            x0=self.lower_bound, y0=0, x1=self.lower_bound, y1=1,
            xref='x', yref='y domain',
            line=dict(color=self.colors['USF_Green'], width=3, dash='dash'))
        
        fig.add_annotation(
            x=self.lower_bound, y=1.25, 
            xref='x', yref='y domain',
            text=f'{self.alpha/2*100}th<br>percentile<br>{round(self.lower_bound, 2)}', 
            showarrow=False, 
            font=dict(size=14, color=self.colors['USF_Green']))
        
        fig.add_shape(
            type='line',
            x0=self.upper_bound, y0=0, x1=self.upper_bound, y1=1,
            xref='x', yref='y domain',
            line=dict(color=self.colors['USF_Green'], width=3, dash='dash'))
        
        fig.add_annotation(
            x=self.upper_bound, y=1.25, 
            xref='x', yref='y domain',
            text=f'{(1-self.alpha/2)*100}th<br>percentile<br>{round(self.upper_bound, 2)}', 
            showarrow=False, 
            font=dict(size=14, color=self.colors['USF_Green']))
        
        # Add background rectangle for the confidence interval
        fig.add_shape(
            type='rect',
            x0=self.lower_bound, y0=0, x1=self.upper_bound, y1=1,
            xref='x', yref='paper',
            fillcolor=self.colors['USF_Green'], opacity=0.15, layer='below', line_width=0
        )

        fig.add_annotation(
            x=self.stat_estimate, y=0.8, 
            xref='x', yref='y domain',
            text=f'<b>{int((1-self.alpha)*100)}% confidence interval</b>', 
            showarrow=False, 
            font=dict(size=18, color=self.colors['USF_Green']))
        

        # Update layout
        fig.update_layout(
            title={
                'text': f"<span style='font-size:18px; font-weight:bold;'>Sampling Distribution of Bootstrapped {self.statistic_text}</span>",
                'y': 0.97, 'yanchor': 'top'},
            xaxis_title=f'Sample {self.statistic_text}',
            yaxis_title='Frequency',
        )

        # Show plot
        st.plotly_chart(fig, use_container_width=True)
