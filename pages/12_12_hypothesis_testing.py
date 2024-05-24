import numpy as np
import streamlit as st

from utils.stats_viz import (
    calculate_critical_value, calculate_p_value, HypothesisTest,
    section_1_hypothesis, section_2_significance_level,
    section_4_critical_value, section_5_p_value
    )
from utils.other_utils import add_logo, setup_sticky_header, add_title


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Hypothesis Testing',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Hypothesis Testing'
        add_title(title)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    tab1, tab2 = st.tabs(['**Mean**', '**Proportion**'])

    with tab1:
        subtab1, subtab2 = st.tabs(['**When Population Variance is Known**', '**When Population Variance is Unknown**'])
        with subtab1:
            distribution = 'z'
            key = 'mean1'
            
            col1 ,col2 = st.columns([0.4, 0.6])
            with col1:
                # User selection: 1. Hypothesis
                parameter = 'mean'
                hypothesis = section_1_hypothesis(parameter, key)
                
                # User selection: 2. Significance Level
                alpha = section_2_significance_level(key)

                # User selection: 3. Test Statistic
                st.write(
                    "<span style='font-size:18px; font-weight:bold;'>3. Test Statistic</span>", 
                    unsafe_allow_html=True)
                
                subcol1, subcol2 = st.columns([0.4, 0.6])
                with subcol1:
                    test_stat_formula_part1 = r"""
                        $$z = \cfrac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$
                        """
                    st.write(test_stat_formula_part1)
                with subcol2:
                    test_stat_formula_part2 = r"""
                        where:
                        - $$\bar{x}$$ is the sample mean,
                        - $$\mu_0$$ is the population mean,
                        - $$\sigma$$ is the population standard deviation,
                        - $$n$$ is the sample size.
                        """
                    st.write(test_stat_formula_part2)

                st.write('\n')

                st.write(
                    "<span style='font-size:16px;'>Assume the population mean $$\mu_0$$ = 50,</span>", 
                    unsafe_allow_html=True)

                # sample mean
                subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
                with subcol1:
                    st.write(
                        "<span style='font-size:16px;'>sample mean</span>", 
                        unsafe_allow_html=True)
                    
                with subcol2:
                    sample_mean = st.slider(
                        'sample mean ', 0, 100, value=75, step=1, 
                        label_visibility='collapsed',
                        key=f'test_stat_{key}_1')
                
                # population standard deviation
                subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
                with subcol1:
                    st.write(
                        "<span style='font-size:16px;'>population SD ($$\sigma$$)</span>", 
                        unsafe_allow_html=True)
                    
                with subcol2:
                    population_sd = st.slider(
                        'population SD ', 0, 100, value=75, step=1, 
                        label_visibility='collapsed',
                        key=f'test_stat_{key}_2')

                # sample size
                subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
                with subcol1:
                    st.write(
                        "<span style='font-size:16px;'>sample size ($$n$$)</span>", 
                        unsafe_allow_html=True)
                    
                with subcol2:
                    n = st.slider(
                        'sample size', 10, 50, value=20, step=1, 
                        label_visibility='collapsed',
                        key=f'test_stat_{key}_3')

                population_mean = 50
                test_statistic = round((sample_mean-population_mean)/(population_sd/np.sqrt(n)), 2)
                highlighted_test_stat = \
                    f"<span style='background-color: rgba(255, 0, 0, 0.4); font-weight:bold;'>{test_statistic}</span>"

                st.write(
                    f"<span style='font-size:18px;'>The test statistic z = {highlighted_test_stat}</span>", 
                    unsafe_allow_html=True)

                st.write()

                # 4. Critical Value
                df = None if distribution == 'z' else n-1

                critical_vals = calculate_critical_value(hypothesis, alpha, distribution, df)
                section_4_critical_value(critical_vals)

                # 5. P-Value
                p_value = calculate_p_value(hypothesis, distribution, test_statistic, df)
                section_5_p_value(p_value)

            with col2:
                hypoTest = HypothesisTest(
                    hypothesis, distribution, critical_vals, test_statistic, df)
                
                # Plot hypothesis test interactive line plot
                hypoTest.plot_hypothesis()

                # Conclusion Section
                hypoTest.generate_conclusion()

        with subtab2:
            distribution = 't'
            key = 'mean2'
            
            col1 ,col2 = st.columns([0.4, 0.6])
            with col1:
                # User selection: 1. Hypothesis
                parameter = 'mean'
                hypothesis = section_1_hypothesis(parameter, key)
                
                # User selection: 2. Significance Level
                alpha = section_2_significance_level(key)

                # User selection: 3. Test Statistic
                st.write(
                    "<span style='font-size:18px; font-weight:bold;'>3. Test Statistic</span>", 
                    unsafe_allow_html=True)
                
                subcol1, subcol2 = st.columns([0.4, 0.6])
                with subcol1:
                    test_stat_formula_part1 = r"""
                        $$t = \cfrac{\bar{x} - \mu_0}{s / \sqrt{n}}$$
                        """
                    st.write(test_stat_formula_part1)
                with subcol2:
                    test_stat_formula_part2 = r"""
                        where:
                        - $$\bar{x}$$ is the sample mean,
                        - $$\mu_0$$ is the population mean,
                        - $$s$$ is the sample standard deviation,
                        - $$n$$ is the sample size.
                        """
                    st.write(test_stat_formula_part2)

                st.write('\n')

                st.write(
                    "<span style='font-size:16px;'>Assume the population mean $$\mu_0$$ = 50,</span>", 
                    unsafe_allow_html=True)

                # sample mean
                subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
                with subcol1:
                    st.write(
                        "<span style='font-size:16px;'>sample mean ($$\bar{x}$$)</span>", 
                        unsafe_allow_html=True)
                    
                with subcol2:
                    sample_mean = st.slider(
                        'sample mean ', 0, 100, value=75, step=1, 
                        label_visibility='collapsed',
                        key=f'test_stat_{key}_1')
                
                # sample standard deviation
                subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
                with subcol1:
                    st.write(
                        "<span style='font-size:16px;'>sample SD ($$\sigma$$)</span>", 
                        unsafe_allow_html=True)
                    
                with subcol2:
                    sample_sd = st.slider(
                        'sample SD ', 0, 100, value=75, step=1, 
                        label_visibility='collapsed',
                        key=f'test_stat_{key}_2')

                # sample size
                subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
                with subcol1:
                    st.write(
                        "<span style='font-size:16px;'>sample size ($$n$$)</span>", 
                        unsafe_allow_html=True)
                    
                with subcol2:
                    n = st.slider(
                        'sample size', 10, 50, value=20, step=1, 
                        label_visibility='collapsed',
                        key=f'test_stat_{key}_3')

                population_mean = 50
                test_statistic = round((sample_mean-population_mean)/(sample_sd/np.sqrt(n)), 2)
                highlighted_test_stat = \
                    f"<span style='background-color: rgba(255, 0, 0, 0.4); font-weight:bold;'>{test_statistic}</span>"

                st.write(
                    f"<span style='font-size:18px;'>The test statistic t = {highlighted_test_stat}</span>", 
                    unsafe_allow_html=True)

                st.write()

                # 4. Critical Value
                df = None if distribution == 'z' else n-1

                critical_vals = calculate_critical_value(hypothesis, alpha, distribution, df)
                section_4_critical_value(critical_vals)

                # 5. P-Value
                p_value = calculate_p_value(hypothesis, distribution, test_statistic, df)
                section_5_p_value(p_value)

            with col2:
                hypoTest = HypothesisTest(
                    hypothesis, distribution, critical_vals, test_statistic, df)
                
                # Plot hypothesis test interactive line plot
                hypoTest.plot_hypothesis()

                # Conclusion Section
                hypoTest.generate_conclusion()


    with tab2:
        distribution = 'z'
        key = 'proportion'
        
        col1 ,col2 = st.columns([0.4, 0.6])
        with col1:
            # User selection: 1. Hypothesis
            parameter = 'proportion'
            hypothesis = section_1_hypothesis(parameter, key)
            
            # User selection: 2. Significance Level
            alpha = section_2_significance_level(key)

            # User selection: 3. Test Statistic
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>3. Test Statistic</span>", 
                unsafe_allow_html=True)
            
            subcol1, subcol2 = st.columns([0.4, 0.6])
            with subcol1:
                test_stat_formula_part1 = r"""
                    $$z = \cfrac{\hat{p} - p_0}{\sqrt{\cfrac{p_0(1 - p_0)}{n}}}$$
                    """
                st.write(test_stat_formula_part1)
            with subcol2:
                test_stat_formula_part2 = r"""
                    where:
                    - $$\hat{p}$$ is the sample proportion,
                    - $$p_0$$ is the population proportion,
                    - $$n$$ is the sample size.
                    """
                st.write(test_stat_formula_part2)

            st.write('\n')

            st.write(
                "<span style='font-size:16px;'>Assume the population proportion $$p_0$$ = 0.5,</span>", 
                unsafe_allow_html=True)

            # sample proportion
            subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
            with subcol1:
                st.write(
                    "<span style='font-size:16px;'>sample proportion ($$\hat{p}$$)</span>", 
                    unsafe_allow_html=True)
                
            with subcol2:
                p = st.slider(
                    'sample proportion ', 0.0, 1.0, value=0.75, step=0.01, 
                    label_visibility='collapsed',
                    key=f'test_stat_{key}_1')
            
            # sample size
            subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
            with subcol1:
                st.write(
                    "<span style='font-size:16px;'>sample size ($$n$$)</span>", 
                    unsafe_allow_html=True)
                
            with subcol2:
                n = st.slider(
                    'sample size', 10, 50, value=20, step=1, 
                    label_visibility='collapsed',
                    key=f'test_stat_{key}_2')

            population_p = 0.5
            test_statistic = round((p-population_p)/((population_p*(1-population_p)/n)**(1/2)), 2)
            highlighted_test_stat = f"<span style='background-color: rgba(255, 0, 0, 0.4); font-weight:bold;'>{test_statistic}</span>"

            st.write(
                f"<span style='font-size:18px;'>The test statistic z = {highlighted_test_stat}</span>", 
                unsafe_allow_html=True)

            st.write()

            # 4. Critical Value
            df = None if distribution == 'z' else n-1

            critical_vals = calculate_critical_value(hypothesis, alpha, distribution, df)
            section_4_critical_value(critical_vals)

            # 5. P-Value
            p_value = calculate_p_value(hypothesis, distribution, test_statistic, df)
            section_5_p_value(p_value)

        with col2:
            hypoTest = HypothesisTest(
                hypothesis, distribution, critical_vals, test_statistic, df)
            
            # Plot hypothesis test interactive line plot
            hypoTest.plot_hypothesis()

            # Conclusion Section
            hypoTest.generate_conclusion()


if __name__ == '__main__':
    main()