import streamlit as st
import numpy as np

from utils.stats_viz import UniformDistribution, NormalDistribution, ExponentialDistribution, CentralLimitTheorm
from utils.other_utils import add_logo, setup_sticky_header, add_title
from utils.formulas import (
    uniform_notation, uniform_exp, uniform_var,
    normal_notation, normal_exp, normal_var,
    exponential_notation, exponential_exp, exponential_var
    )

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Central Limit Therom',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Central Limit Therom'
        add_title(title)

        col1, _, col2 = st.columns([0.45, 0.1, 0.45])
        with col1: 
            # User selection: population distribution
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Population Distribution</span>", 
                unsafe_allow_html=True)
            
            population_dist = st.radio(
                'Population Distribution', 
                ['uniform', 'normal', 'exponential'], 
                horizontal=True, 
                label_visibility='collapsed')
            
            # User selection: population parameters
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Population Parameters</span>", 
                unsafe_allow_html=True)
            
            if population_dist == 'uniform':
                # lower, upper = 10.0, 15.0
                lower, upper = st.slider(
                    'Lower Bound (α) and Upper Bound (β)', 
                    min_value=-30.0, max_value=30.0, value=(10.0, 15.0), step=0.1)
                
            elif population_dist == 'normal':
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    mean = st.slider('Mean (μ)', min_value=-10.0, max_value=10.0, value=0.0, step=0.01)
                with sub_col2:
                    std_dev = st.slider('Standard Deviation (σ)', min_value=0.1, max_value=10.0, value=3.0, step=0.01)

            elif population_dist == 'exponential':
                rate = st.slider('Rate (λ)', min_value=0.1, max_value=10.0, value=1.0, step=0.01)
            
        with col2: 
            # User selection: sample size (n) - number of observations per batch
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Sample Size (n)</span>", 
                # "<span style='font-size:15px; color: #a61401'> - refers to the number of data per batch</span>",
                unsafe_allow_html=True)
            
            sample_size = st.slider('Sample Size (n)', min_value=30, max_value=100, value=30, step=10,
                                    label_visibility='collapsed')
            
            # User selection: number of samples - number of batches
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Number of Repetitions</span>", 
                # "<span style='font-size:15px; color: #a61401'> - number of repetitions</span>",
                unsafe_allow_html=True)
            
            n_samples = st.slider('Number of Samples', min_value=1, max_value=3000, value=50, step=10,
                                  label_visibility='collapsed')

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    
    col11, _, col12 = st.columns([0.45, 0.1, 0.45])
    with col11: 
        if population_dist == 'uniform':
            notation = uniform_notation
            exp_formula = uniform_exp
            # var_formula = uniform_var

            exp = 1/2*(lower + upper)
            # var = 1/12*(upper - lower)**2

        elif population_dist == 'normal':
            notation = normal_notation
            exp_formula = normal_exp
            # var_formula = normal_var

            exp = mean
            # var = std_dev**2

        elif population_dist == 'exponential':
            notation = exponential_notation  
            exp_formula = exponential_exp
            # var_formula = exponential_var

            exp = 1/rate
            # var = 1/(rate**2)

        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Population Distribution</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left: 20px;'>", notation, "</span>",
            unsafe_allow_html=True)
        st.write('')

        st.write(
            "<span style='font-size:18px; font-weight:bold; margin-left:30px;'>Mean</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left:30px;'>", exp_formula, "</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left:10px;'>$=$</span>",
            f"<span style='font-size:20px; font-weight:bold;'>", round(exp, 3), "</span>",
            unsafe_allow_html=True)
        st.write('')

        # st.write(
        #     "<span style='font-size:18px; font-weight:bold; margin-left:30px;'>Variance</span>", 
        #     f"<span style='font-size:16px; font-weight:bold; margin-left:25px;'>", var_formula, "</span>",
        #     f"<span style='font-size:16px; font-weight:bold; margin-left:10px;'>$=$</span>",
        #     f"<span style='font-size:20px; font-weight:bold;'>", round(var, 2), "</span>", 
        #     unsafe_allow_html=True)
        

        # Plot population distribtuion
        # Set up population distribution size
        pop_size = 5000

        # Fix the random seed for the randomly generated population dataset, 
        # in order to avoid the population dataset being re-generated when user updates sample_size and n_samples.
        # It's Streamlit's default setup to reload the webpage when user has new input
        np.random.seed(42)

        if population_dist == 'uniform':           
            uniformDist = UniformDistribution(lower, upper, pop_size)
            uniformDist.plot_pdfs()

            population_data = uniformDist.simulated_data

        elif population_dist == 'normal':
            normalDist = NormalDistribution(mean, std_dev, pop_size)
            normalDist.plot_pdfs()

            population_data = normalDist.simulated_data

        elif population_dist == 'exponential':
            exponentialDist = ExponentialDistribution(rate, pop_size)
            exponentialDist.plot_pdfs()

            population_data = exponentialDist.simulated_data

    with col12: 
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Sampling Distribution of the Sample Mean</span>", 
            unsafe_allow_html=True)
        st.write('')

        Clt = CentralLimitTheorm(population_data, sample_size, n_samples)
        
        mean_of_sample_means = Clt.mean_of_sample_means

        st.write(
            "<span style='font-size:18px; font-weight:bold; margin-left:30px;'>Mean</span>", 
            # f"<span style='font-size:16px; font-weight:bold; margin-left:50px;'>", exp_formula, "</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left:10px;'>$=$</span>",
            f"<span style='font-size:20px; font-weight:bold;'>", round(mean_of_sample_means, 3), "</span>",
            unsafe_allow_html=True)
        st.write('')

        # Plot sample means distribtuion
        Clt.plot_clt_sample_mean()


if __name__ == '__main__':
    main()