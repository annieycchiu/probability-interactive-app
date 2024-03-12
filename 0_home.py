import streamlit as st
from utils.other_utils import add_logo, display_homepage_formulas
from utils.formulas import *


def main():

    st.set_page_config(
        page_title='MSDS504',
        page_icon=':bar_chart:',
        layout='wide'
    )

    # Add USF logo
    add_logo()

    st.title('USF MSDS504 Statistics & Probability')
    st.write('')
    st.write('')

    # Binomial distribution
    display_homepage_formulas(
        '01. Binomial Distribution',
        binomial_notation,
        binomial_pmf, binomial_cdf,
        binomial_exp, binomial_var,
        type='Discrete'
        )
    st.divider()

    # Poisson distribution
    display_homepage_formulas(
        '02. Poisson Distribution',
        poisson_notation,
        poisson_pmf, poisson_cdf,
        poisson_exp, poisson_var,
        type='Discrete'
        )
    st.divider()

    # Uniform distribution
    display_homepage_formulas(
        '03. Uniform Distribution',
        uniform_notation,
        uniform_pdf, uniform_cdf,
        uniform_exp, uniform_var,
        type='Continuous'
        )
    st.divider()

    # Normal (Gaussian) distribution
    display_homepage_formulas(
        '04. Normal (Gaussian) Distribution',
        normal_notation,
        normal_pdf, normal_cdf,
        normal_exp, normal_var,
        type='Continuous'
        )
    st.divider()

    # Exponential distribution
    display_homepage_formulas(
        '05. Exponential Distribution',
        exponential_notation,
        exponential_pdf, exponential_cdf,
        exponential_exp, exponential_var,
        type='Continuous'
        )
    st.divider()

    # Gamma distribution
    display_homepage_formulas(
        '06. Gamma Distribution',
        gamma_notation,
        gamma_pdf, gamma_cdf,
        gamma_exp, gamma_var,
        type='Continuous'
        )



if __name__ == '__main__':
    main()