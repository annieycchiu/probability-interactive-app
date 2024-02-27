import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import streamlit as st

USF_Green = '#00543C'
USF_Yellow = '#FDBB30'
USF_Gray = '#75787B'



## Binomial Distribution

class BinomialPMF:
    def __init__(self, n, p, color=USF_Green):
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
        pmf_prob = np.array([round(math.comb(self.n, k) * (self.p**k) * ((1 - self.p)**(self.n - k)), 3) for k in x])
        self.pmf_prob = pmf_prob

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
    def __init__(self, n, p, size=1, color=USF_Green):
        """
        Parameters:
        - n (int): Number of trials.
        - p (float): Probability of success for each trial.
        - size (int): Number of simulations to run (default is '#00543C').
        - color (str): Color for the plot (default is 1).
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
        simulated_data = np.sum(outcomes, axis=1)
        self.simulated_data = simulated_data

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
    def __init__(self, lmbda, color=USF_Green):
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
        pmf_prob = np.array([round((np.exp(-self.lmbda) * self.lmbda**k) / math.factorial(k), 3) for k in x])
        self.pmf_prob = pmf_prob

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
    def __init__(self, lmbda, size=1, color=USF_Green):
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
        simulated_data = np.random.poisson(self.lmbda, self.size)
        self.simulated_data = simulated_data

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