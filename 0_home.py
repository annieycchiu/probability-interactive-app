import streamlit as st
from utils.other_utils import add_logo, display_formulas
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
    display_formulas(
        '01. Binomial Distribution',
        binomial_pmf, binomial_cdf,
        binomial_exp, binomial_var,
        type='Discrete'
        )
    st.divider()

    # Poisson distribution
    display_formulas(
        '02. Poisson Distribution',
        poisson_pmf, poisson_cdf,
        poisson_exp, poisson_var,
        type='Discrete'
        )
    st.divider()

    # Uniform distribution
    display_formulas(
        '03. Uniform Distribution',
        uniform_pdf, uniform_cdf,
        uniform_exp, uniform_var,
        type='Continuous'
        )
    st.divider()

    # Normal (Gaussian) distribution
    display_formulas(
        '04. Normal (Gaussian) Distribution',
        normal_pdf, normal_cdf,
        normal_exp, normal_var,
        type='Continuous'
        )
    st.divider()

    # Gamma distribution
    display_formulas(
        '05. Gamma Distribution',
        gamma_pdf, gamma_cdf,
        gamma_exp, gamma_var,
        type='Continuous'
        )
    st.divider()

    # Uniform distribution
    display_formulas(
        '06. Uniform Distribution',
        uniform_pdf, uniform_cdf,
        uniform_exp, uniform_var,
        type='Continuous'
        )



if __name__ == '__main__':
    main()