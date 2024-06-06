import streamlit as st

from utils.other_utils import add_logo, setup_sticky_header, add_title
from utils.stats_viz import BivariateNormalDistribution


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Bivariate Normal Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='auto' # hides the sidebar on small devices and shows it otherwise
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Bivariate Normal Distribution'
        add_title(title)

        col1, _, col2, _, col3 = st.columns([0.30, 0.03, 0.30, 0.03, 0.30])
        with col1:
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Distribution X1</span>", 
                unsafe_allow_html=True)
            
            mean_x1 = st.slider("Mean of X1", 5.0, 15.0, 7.0)
            std_x1 = st.slider("Standard Deviation of X1", 1.0, 5.0, 1.0)

        with col2:
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Distribution X2</span>", 
                unsafe_allow_html=True)
            
            mean_x2 = st.slider("Mean of X2", 5.0, 15.0, 10.0)
            std_x2 = st.slider("Standard Deviation of X2", 1.0, 5.0, 3.0)

        with col3:
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Correlation Coefficient (ρ)</span>", 
                unsafe_allow_html=True)
            
            r = st.slider("Correlation Coefficient (ρ)", -1.0, 1.0, 0.0, label_visibility='collapsed')

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    bivariateNormalDist = BivariateNormalDistribution(mean_x1, mean_x2, std_x1, std_x2, r)

    col1, _, col2 = st.columns([0.45, 0.05, 0.45])
    with col1:
        bivariateNormalDist.plot_2D_contour()

    with col2:
        bivariateNormalDist.plot_3D_surface()


if __name__ == '__main__':
    main()