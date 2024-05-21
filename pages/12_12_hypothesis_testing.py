import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import streamlit as st

from utils.stats_viz import calculate_critical_value
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

    # tab1, tab2, tab3 = st.tabs(['**Mean**', '**Proportion**', '**Variance**'])

    # with tab2:
    col1 ,col2 = st.columns([0.4, 0.6])
    with col1:
        # User selection: 1. Hypothesis
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>1. Hypothesis</span>", 
            unsafe_allow_html=True)
        
        hypo_options = [
            "$$H_0: p = p_0 \quad H_a: p \\neq p_0 \quad$$ (two-tailed)",
            "$$H_0: p = p_0 \quad H_a: p > p_0 \quad$$ (one-tailed, right-tailed)",
            "$$H_0: p = p_0 \quad H_a: p < p_0 \quad$$ (one-tailed, left-tailed)"
        ]

        hypo = st.radio('hypothesis', hypo_options, horizontal=True, label_visibility='collapsed')
        st.write('')

        # User selection: 2. Significance Level
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>2. Significance Level</span>", 
            unsafe_allow_html=True)
        
        alpha = st.radio(
            'alpha', ['0.05 (5%)', '0.01 (1%)', '0.10 (10%)'], horizontal=True, label_visibility='collapsed')
        st.write('')

        # 3. Critical Value
        alpha = float(alpha.split(' ')[0])

        if 'two' in hypo:
            z_critical_two_tailed = round(calculate_critical_value(alpha, two_tailed=True), 2)
            criticals = [-z_critical_two_tailed, z_critical_two_tailed]

        if 'left' in hypo:
            z_critical_one_tailed = round(calculate_critical_value(alpha, two_tailed=False), 2)
            criticals = [-z_critical_one_tailed]
            
        if 'right' in hypo:
            z_critical_one_tailed = round(calculate_critical_value(alpha, two_tailed=False), 2)
            criticals = [z_critical_one_tailed]

        highlighted_criticals = ", ".join(f"<span style='background-color: rgba(253, 187, 48, 0.4);'>{c}</span>" for c in criticals)
        st.write(
            f"<span style='font-size:18px; font-weight:bold;'>3. Critical Value: {highlighted_criticals}</span>", 
            unsafe_allow_html=True)

        # User selection: 4. Test Statistic
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>4. Test Statistic</span>", 
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
            "<span style='font-size:16px;'>Let's assume the population proportion $$p_0$$ = 0.5,</span>", 
            unsafe_allow_html=True)

        subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
        with subcol1:
            st.write(
                "<span style='font-size:16px;'>sample proportion ($$\hat{p}$$)</span>", 
                unsafe_allow_html=True)
            
        with subcol2:
            p = st.slider('sample proportion ', 0.0, 1.0, value=0.75, step=0.01, label_visibility='collapsed')

        subcol1, subcol2, _ = st.columns([0.35, 0.5, 0.15])
        with subcol1:
            st.write(
                "<span style='font-size:16px;'>sample size ($$n$$)</span>", 
                unsafe_allow_html=True)
            
        with subcol2:
            n = st.slider('sample size', 10, 50, value=30, step=1, label_visibility='collapsed')

        population_p = 0.5
        test_statistic = round((p-population_p)/((population_p*(1-population_p)/n)**(1/2)), 2)
        highlighted_test_stat = f"<span style='background-color: rgba(255, 0, 0, 0.4); font-weight:bold;'>{test_statistic}</span>"

        st.write(
            f"<span style='font-size:18px;'>The test statistic z = {highlighted_test_stat}</span>", 
            unsafe_allow_html=True)

        st.write()

        # 5. P-Value
        p_value = stats.norm.sf(abs(test_statistic))

        st.write(
            f"<span style='font-size:18px; font-weight:bold;'>5. P-Value: {round(p_value, 5)}</span>", 
            unsafe_allow_html=True)
        
        # 6. Conclusion

        st.write()

    with col2:
        # Generate data for standard normal distribution
        x = np.linspace(-4, 4, 1000)
        y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

        # Create a standard normal distribution line plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', 
            marker=dict(color='#75787B'),
            name='Standard Normal Distribution'))

        fig.update_layout(
            title=dict(
                text="<span style='font-size:18px; font-weight:bold;'>Standard Normal Distribution</span>",
                y=0.95),
            yaxis_title='Probability Density',
            legend=dict(
                orientation='h',
                yanchor='top', y=1.2, 
                xanchor='center',x=0.5,
                font=dict(size=14)))
        
        for c in criticals: 
            # Add vertical line to split rejection and non-rejection region
            fig.add_shape(
                type='line',
                x0=c, y0=0, x1=c, y1=0.3,
                line=dict(color='#FDBB30', width=3, dash='dash'))
            
            fig.add_annotation(
                x=c, y=0.36, text=f'Critical Value', showarrow=False, font=dict(size=16, color='#FDBB30'))
            fig.add_annotation(
                x=c, y=0.33, text=f'{c}', showarrow=False, font=dict(size=16, color='#FDBB30'))
            
            # Highlight the rejection region
            if c < 0:
                x_partial = x[x <= c]
                y_partial = y[:len(x_partial)]

            else:
                x_partial = x[x >= c]
                y_partial = y[-len(x_partial):]

            # Fill out the rejection region
            fig.add_trace(
                go.Scatter(
                    x=x_partial, y=y_partial, 
                    fill='tozeroy', 
                    fillcolor='rgba(253, 187, 48, 0.4)', 
                    line=dict(color='rgba(253, 187, 48, 0.4)'), name='Rejection Region'))
            
        # Fill out the non-rejection region
        if len(criticals) == 2:
            x_partial = x[(x >= -c) & (x <= c)]
            y_partial = y[(x >= -c) & (x <= c)]
        else:
            if criticals[0] < 0:
                x_partial = x[x >= c]
                y_partial = y[-len(x_partial):]
            else:
                x_partial = x[x <= c]
                y_partial = y[:len(x_partial)]

        fig.add_trace(
            go.Scatter(
                x=x_partial, y=y_partial, 
                fill='tozeroy', 
                fillcolor='rgba(0, 84, 60, 0.4)', 
                line=dict(color='rgba(0, 84, 60, 0.4)'), name='Non-Rejection Region'))
        
        # Add vertical line of test statistic
        fig.add_shape(
            type='line',
            x0=test_statistic, y0=0, x1=test_statistic, y1=0.3,
            line=dict(color='#FF0000', width=3))
        
        fig.add_annotation(
            x=test_statistic, y=0.36, text=f'Test Statistic', showarrow=False, font=dict(size=16, color='#FF0000'))
        fig.add_annotation(
            x=test_statistic, y=0.33, text=f'{test_statistic}', showarrow=False, font=dict(size=16, color='#FF0000'))
        
        st.plotly_chart(fig, use_container_width=True)


        # Conclusion Section
        conclusion = 'Reject the Null Hypothesis' if p_value < 0.05 else 'Do Not Reject the Null Hypothesis'

        if conclusion == 'Reject the Null Hypothesis':
            detail_1 = '| Test Statistic | > | Critical Value |'
            detail_2 = 'P-Value < 0.05'
        else:
            detail_1 = '| Test Statistic | <= | Critical Value |'
            detail_2 = 'P-Value >= 0.05'

        # Add a markdown section with a background color and custom font sizes
        st.markdown(
            """
            <style>
            .section {
                background-color: #f0f0f0; /* Set your desired background color */
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

if __name__ == '__main__':
    main()