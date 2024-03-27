import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random

from utils.other_utils import add_logo, setup_sticky_header, add_title
from utils.stats_viz import UniformDistribution, NormalDistribution, ExponentialDistribution
from utils.formulas import (
    uniform_notation, uniform_exp, uniform_var,
    normal_notation, normal_exp, normal_var,
    exponential_notation, exponential_exp, exponential_var
    )

colors = {
    'USF_Green': '#00543C',
    'USF_Yellow': '#FDBB30',
    'USF_Yellow_rbga_line': 'rgba(253, 187, 48, 0.8)',
    'USF_Yellow_rbga_fill': 'rgba(253, 187, 48, 0.4)',
    'USF_Gray': '#75787B'
}

class Bootstrapping():
    def __init__(self, population_data, sample_size, n_resamplings, colors=colors):
        self.population_data = population_data
        self.sample_size = sample_size
        self.n_resamplings = n_resamplings
        self.colors = colors

        self.original_sample = None
        self.bootstrap_samples = None
        self.generate_bootstrap_sample()

        self.bootstrap_means = None
        self.compute_bootstrap_means()

    def generate_bootstrap_sample(self):
        self.original_sample = random.sample(self.population_data, self.sample_size)

        self.bootstrap_samples = []
        for _ in range(self.n_resamplings):
            bootstrap_sample = np.random.choice(self.original_sample, size=self.sample_size, replace=True)
            self.bootstrap_samples.append(bootstrap_sample)

    def compute_bootstrap_means(self):
        self.bootstrap_means = []
        for bootstrap_sample in self.bootstrap_samples:
            bootstrap_mean = np.mean(bootstrap_sample)
            self.bootstrap_means.append(bootstrap_mean)

    def plot_sampling_distribution(self):
        # Create histogram 
        fig = go.Figure()

        # Add trace for the histogram of bootstrap means
        fig.add_trace(
            trace=go.Histogram(
                x=self.bootstrap_means, nbinsx=30,
                marker=dict(
                    color=self.colors['USF_Yellow_rbga_fill'],
                    line=dict(color=self.colors['USF_Yellow_rbga_line'], width=1))))

        sampling_distribution_mean = np.mean(self.bootstrap_means)

        # Add vertical line for the mean of sampling distribution
        fig.add_vline(
            x=sampling_distribution_mean, 
            line={'color': self.colors['USF_Green'], 'dash': 'dash', 'width': 2.5},
            annotation_text=f"Mean: {sampling_distribution_mean:.3f}", 
            annotation_position="top")

        # Update layout
        fig.update_layout(
            title="Bootstrap Sampling Distribution",
            xaxis_title="Bootstrap Sample Mean",
            yaxis_title="Frequency",
        )

        # Show plot
        st.plotly_chart(fig)


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Bootstrapping',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Bootstrapping'
        add_title(title)

        col1, _ ,col2 = st.columns([0.45, 0.1, 0.45])
        with col1:
            # User selection: Population Distribution
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Population Distribution</span>", 
                unsafe_allow_html=True)
            
            popu_dist = st.radio(
                'Population Distribution', 
                ['uniform', 'normal', 'exponential'], 
                horizontal=True, 
                label_visibility='collapsed',
                index=1)
            
            st.write('')

            # User selection: Parameters of Population Distribution
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Parameters of Population Distribution</span>", 
                unsafe_allow_html=True)
            
            popu_size = 50

            if popu_dist == 'uniform':
                lower, upper = st.slider(
                    'Lower Bound (α) and Upper Bound (β)', 
                    min_value=-30.0, max_value=30.0, value=(10.0, 15.0), step=0.1)
                
                uniformDist = UniformDistribution(lower, upper, size=popu_size)
                popu_data = uniformDist.simulated_data
                
            elif popu_dist == 'normal':
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    mean = st.slider(
                        'Mean (μ)', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
                with sub_col2:
                    std_dev = st.slider(
                        'Standard Deviation (σ)', min_value=0.1, max_value=10.0, value=3.0, step=0.01)
                    
                normalDist = NormalDistribution(mean, std_dev, size=popu_size)
                popu_data = normalDist.simulated_data

            elif popu_dist == 'exponential':
                rate = st.slider(
                    'Rate (λ)', min_value=0.1, max_value=10.0, value=1.0, step=0.01)
                
                exponentialDist = ExponentialDistribution(rate, size=popu_size)
                popu_data = exponentialDist.simulated_data

            with col2:
                # User selection: Sample Size
                st.write(
                    "<span style='font-size:18px; font-weight:bold;'>Sample Size (n)</span>", 
                    unsafe_allow_html=True)
                
                sample_size = st.slider(
                    'Sample Size', min_value=10, max_value=30, value=10, step=5, label_visibility='collapsed')
                
                # User selection: Number of Resamplings
                st.write(
                    "<span style='font-size:18px; font-weight:bold;'>Number of Resamplings</span>", 
                    unsafe_allow_html=True)
                
                n_resamplings = st.slider(
                    'Number of Resamplings', min_value=100, max_value=3000, value=500, step=100, label_visibility='collapsed')

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    ######################################
    col1, _ ,col2 = st.columns([0.45, 0.1, 0.45])
    with col1:
        if popu_dist == 'uniform':
            notation = uniform_notation
            exp_formula = uniform_exp
            var_formula = uniform_var

            exp = 1/2*(lower + upper)
            var = 1/12*(upper - lower)**2

        elif popu_dist == 'normal':
            notation = normal_notation
            exp_formula = normal_exp
            var_formula = normal_var

            exp = mean
            var = std_dev**2

        elif popu_dist == 'exponential':
            notation = exponential_notation  
            exp_formula = exponential_exp
            var_formula = exponential_var

            exp = 1/rate
            var = 1/(rate**2)

        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Population Distribution</span>", 
            unsafe_allow_html=True)
        st.write('')
        
        st.write(
            "<span style='font-size:18px; font-weight:bold; margin-left:30px;'>Notation</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left: 27px;'>", notation, "</span>", 
            unsafe_allow_html=True)
        st.write('')

        st.write(
            "<span style='font-size:18px; font-weight:bold; margin-left:30px;'>Mean</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left:50px;'>", exp_formula, "</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left:10px;'>$=$</span>",
            f"<span style='font-size:20px; font-weight:bold;'>", round(exp, 3), "</span>",
            unsafe_allow_html=True)
        st.write('')

        st.write(
            "<span style='font-size:18px; font-weight:bold; margin-left:30px;'>Variance</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left:25px;'>", var_formula, "</span>",
            f"<span style='font-size:16px; font-weight:bold; margin-left:10px;'>$=$</span>",
            f"<span style='font-size:20px; font-weight:bold;'>", round(var, 3), "</span>", 
            unsafe_allow_html=True)


    with col2:
         Boot = Bootstrapping(list(popu_data), sample_size, n_resamplings)
         st.write(f'mean of original sample data: {round(np.mean(Boot.original_sample), 3)}')
         Boot.plot_sampling_distribution()

if __name__ == '__main__':
    main()