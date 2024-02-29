import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import streamlit as st

colors = {
    'USF_Green': '#00543C',
    'USF_Yellow': '#FDBB30',
    'USF_Gray': '#75787B'
}


## Binomial Distribution

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


## Poisson Distribution
        
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