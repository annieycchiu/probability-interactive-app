import streamlit as st

from utils.other_utils import add_logo, add_title, setup_sticky_header
from utils.stats_viz import UniformDistribution, NormalDistribution, ExponentialDistribution, MultimodalDistribution


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Multimodal Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add logo
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Multimodal Distribution'
        add_title(title)

        size = 5000

        col1, _ ,col2 = st.columns([0.45, 0.1, 0.45])
        with col1:
            # User selection: Distribution 1
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Distribution 1</span>", 
                unsafe_allow_html=True)
            
            dist1 = st.radio(
                'Distribution 1', 
                ['uniform', 'normal', 'exponential'], 
                horizontal=True, 
                label_visibility='collapsed',
                index=1)
            
            st.write('')

            # User selection: Parameters of Distribution 1
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Parameters of Distribution 1</span>", 
                unsafe_allow_html=True)
            
            if dist1 == 'uniform':
                # lower, upper = 10.0, 15.0
                dist1_lower, dist1_upper = st.slider(
                    'Lower Bound (α) and Upper Bound (β)', 
                    min_value=-30.0, max_value=30.0, value=(10.0, 15.0), step=0.1,
                    key='dist1')
                
                uniformDist = UniformDistribution(dist1_lower, dist1_upper, size)
                dist1_data = uniformDist.simulated_data
                
            elif dist1 == 'normal':
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    dist1_mean = st.slider(
                        'Mean (μ)', min_value=-10.0, max_value=10.0, value=0.0, step=0.01, key='dist11')
                with sub_col2:
                    dist1_std_dev = st.slider(
                        'Standard Deviation (σ)', min_value=0.1, max_value=10.0, value=3.0, step=0.01, key='dist12')
                    
                normalDist = NormalDistribution(dist1_mean, dist1_std_dev, size)
                dist1_data = normalDist.simulated_data

            elif dist1 == 'exponential':
                dist1_rate = st.slider(
                    'Rate (λ)', min_value=0.1, max_value=10.0, value=1.0, step=0.01, key='dist1')
                
                exponentialDist = ExponentialDistribution(dist1_rate, size)
                dist1_data = exponentialDist.simulated_data

        with col2:
            # User selection: Distribution 2
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Distribution 2</span>", 
                unsafe_allow_html=True)
            
            dist2 = st.radio(
                'Distribution 2', 
                ['uniform', 'normal', 'exponential'], 
                horizontal=True, 
                label_visibility='collapsed',
                index=1)
            
            st.write('')

            # User selection: Parameters of Distribution 2
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Parameters of Distribution 2</span>", 
                unsafe_allow_html=True)
            
            if dist2 == 'uniform':
                # lower, upper = 10.0, 15.0
                dist2_lower, dist2_upper = st.slider(
                    'Lower Bound (α) and Upper Bound (β)', 
                    min_value=-30.0, max_value=30.0, value=(10.0, 15.0), step=0.1,
                    key='dist2')
                
                uniformDist = UniformDistribution(dist2_lower, dist2_upper, size)
                dist2_data = uniformDist.simulated_data
                
            elif dist2 == 'normal':
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    dist2_mean = st.slider(
                        'Mean (μ)', min_value=-10.0, max_value=10.0, value=-9.0, step=0.01, key='dist21')
                with sub_col2:
                    dist2_std_dev = st.slider(
                        'Standard Deviation (σ)', min_value=0.1, max_value=10.0, value=1.5, step=0.01, key='dist22')
                    
                normalDist = NormalDistribution(dist2_mean, dist2_std_dev, size)
                dist2_data = normalDist.simulated_data

            elif dist2 == 'exponential':
                dist2_rate = st.slider(
                    'Rate (λ)', min_value=0.1, max_value=10.0, value=1.0, step=0.01, key='dist2')
                
                exponentialDist = ExponentialDistribution(dist2_rate, size)
                dist2_data = exponentialDist.simulated_data

    st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    multimodalDist = MultimodalDistribution(dist1_data, dist2_data)
    multimodalDist.plot_distribution()


if __name__ == "__main__":
    main()