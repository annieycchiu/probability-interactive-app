import streamlit as st
from utils.other_utils import add_logo, display_homepage_formulas, add_title, setup_sticky_header
from utils.formulas import *


def main():

    st.set_page_config(
        page_title='MSDS504',
        page_icon=':bar_chart:',
        layout='wide'
    )

    # Add USF logo
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'USF MSDS504 Statistics & Probability'
        add_title(title)

        welcome_text = 'Welcome to the Probability and Statistics Learning App! This app is designed to help students learn and practice probability and statistics concepts through interactive visualizations and explanations.'
        st.write(
            f"<span style='font-size:18px;'>{welcome_text}</span>", 
            unsafe_allow_html=True)
        
        github_logo_url = 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png'
        github_url = 'https://github.com/annieycchiu/probability-interactive-app'

        # Add author and Github repository link
        author_and_github = f"""
            <div style="display: flex; justify-content: flex-end; align-items: center;">
                <div style="margin-right: 20px;">
                    <span style='font-size:18px;'>Author: Annie Chiu</span>
                </div>
                <div>
                    <a href="{github_url}" target="_blank">
                        <img src="{github_logo_url}" alt="GitHub Logo" style="width:42px;height:42px;">
                    </a>
                </div>
            </div>
            """

        st.markdown(author_and_github, unsafe_allow_html=True)
        
        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.write("""
            #### Probability:
            - [01. Binomial Distribution](/01_binomial_distribution)
            - [02. Poisson_Distribution](/02_poisson_distribution)
            - [03. Uniform Distribution](/03_uniform_distribution)
            - [04. Normal Distribution](/04_normal_distribution)
            - [05. Exponential Distribution](/05_exponential_distribution)
            - [05. Exponential Distribution](/05_exponential_distribution)
            - [06. Gamma Distribution](/06_gamma_distribution)
            - [07. Multimodal Distribution](/07_multimodal_distribution)
            - [08. Bivariate Normal Distribution](/08_bivariate_normal_distribution)
            - [09. Marginal Distribution](/09_marginal_distribution)
            - [10. Conditional Distribution](/10_conditional_distribution)
            - [11. Central Limit Therom](/11_central_limit_therom)
            """)
            

        st.write("""         
            #### Statistical Interence:
            - [12. Hypothesis Testing](/12_hypothesis_testing)
            - [13. Bootstrapping](/13_bootstrapping)
            """)

    with col2:
        with st.expander(
            '**:pushpin: Formulas of Discrete & Continuous Distributions**',
            expanded=True):
            # Binomial distribution
            display_homepage_formulas(
                '01. Binomial Distribution',
                binomial_notation,
                binomial_pmf, binomial_cdf,
                binomial_exp, binomial_var,
                type='Discrete')
            st.divider()

            # Poisson distribution
            display_homepage_formulas(
                '02. Poisson Distribution',
                poisson_notation,
                poisson_pmf, poisson_cdf,
                poisson_exp, poisson_var,
                type='Discrete')
            st.divider()

            # Uniform distribution
            display_homepage_formulas(
                '03. Uniform Distribution',
                uniform_notation,
                uniform_pdf, uniform_cdf,
                uniform_exp, uniform_var,
                type='Continuous')
            st.divider()

            # Normal (Gaussian) distribution
            display_homepage_formulas(
                '04. Normal (Gaussian) Distribution',
                normal_notation,
                normal_pdf, normal_cdf,
                normal_exp, normal_var,
                type='Continuous')
            st.divider()

            # Exponential distribution
            display_homepage_formulas(
                '05. Exponential Distribution',
                exponential_notation,
                exponential_pdf, exponential_cdf,
                exponential_exp, exponential_var,
                type='Continuous')
            st.divider()

            # Gamma distribution
            display_homepage_formulas(
                '06. Gamma Distribution',
                gamma_notation,
                gamma_pdf, gamma_cdf,
                gamma_exp, gamma_var,
                type='Continuous')

        # st.write('#### Probability:')

    # st.write('- [01. Binomial Distribution](/01_binomial_distribution)')
    # with st.expander("**:pushpin: Binomial Distribution Formulas**"):
    #     display_homepage_formulas(
    #         binomial_notation,
    #         binomial_pmf, binomial_cdf,
    #         binomial_exp, binomial_var,
    #         type='Discrete')
        
    # st.write('- [02. Poisson_Distribution](/02_poisson_distribution)')
    # with st.expander("**:pushpin: Poisson Distribution Formulas**"):
    #     display_homepage_formulas(
    #         poisson_notation,
    #         poisson_pmf, poisson_cdf,
    #         poisson_exp, poisson_var,
    #         type='Discrete')

    # st.write('- [03. Uniform Distribution](/03_uniform_distribution)')
    # with st.expander("**:pushpin: Uniform Distribution Formulas**"):
    #     display_homepage_formulas(
    #         uniform_notation,
    #         uniform_pdf, uniform_cdf,
    #         uniform_exp, uniform_var,
    #         type='Continuous')

    # st.write('- [04. Normal Distribution](/04_normal_distribution)')
    # with st.expander("**:pushpin: Normal Distribution Formulas**"):
    #     display_homepage_formulas(
    #         normal_notation,
    #         normal_pdf, normal_cdf,
    #         normal_exp, normal_var,
    #         type='Continuous')

    # st.write('- [05. Exponential Distribution](/05_exponential_distribution)')
    # with st.expander("**:pushpin: Exponential Distribution Formulas**"):
    #     display_homepage_formulas(
    #         exponential_notation,
    #         exponential_pdf, exponential_cdf,
    #         exponential_exp, exponential_var,
    #         type='Continuous')

    # st.write('- [06. Gamma Distribution](/06_gamma_distribution)')
    # with st.expander("**:pushpin: Gamma Distribution Formulas**"):
    #     display_homepage_formulas(
    #         gamma_notation,
    #         gamma_pdf, gamma_cdf,
    #         gamma_exp, gamma_var,
    #         type='Continuous')


if __name__ == '__main__':
    main()