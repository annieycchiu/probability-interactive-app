import streamlit as st
from sklearn import datasets
import seaborn as sns
import pandas as pd

from utils.other_utils import add_logo, setup_sticky_header, add_title
from utils.stats_viz import Bootstrapping

# Load datasets
wine = datasets.load_wine()
wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])[['color_intensity']]
wine_df = wine_df.rename(columns={'color_intensity': 'color intensity'})
wine_desc = wine_df.describe().round(3).T.rename(columns={'50%': '50% (median)'})

tips_df = sns.load_dataset('tips')[['total_bill']]
tips_df = tips_df.rename(columns={'total_bill': 'total bill'})
tips_desc = tips_df.describe().round(3).T.rename(columns={'50%': '50% (median)'})


def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Bootstrapping',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='auto' # hides the sidebar on small devices and shows it otherwise
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Bootstrapping'
        add_title(title)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        # User selection: Dataset
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Dataset</span>", 
            unsafe_allow_html=True)
        
        dataset = st.radio(
            'dataset', 
            ['Wine dataset - color intensity (sklearn)', 'Tips dataset - total bill (seaborn)'],
            # ['Wine - color intensity', 'Tips - total bill', 'Penguins - body mass'], 
            label_visibility='collapsed')
        
        if dataset == 'Wine dataset - color intensity (sklearn)':
            st.dataframe(wine_desc)
            original_sample = wine_df['color intensity']
            dataset_link = 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html'
        elif dataset == 'Tips dataset - total bill (seaborn)':
            st.dataframe(tips_desc)
            original_sample = tips_df['total bill']
            dataset_link = 'https://rdrr.io/cran/reshape2/man/tips.html'

        st.write(f'To get more details of the selected dataset: [click here]({dataset_link})')


        # User selection: Statistic to Estimate
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Statistic to Estimate</span>", 
            unsafe_allow_html=True)
        
        stat_to_estimate = st.radio(
            'Statistic to Estimate', ['Mean', 'Median'], 
            horizontal=True, 
            label_visibility='collapsed')
        
        # User selection: Confidence Level
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Confidence Level</span>", 
            unsafe_allow_html=True)
        
        confidence_level = st.radio(
            'confidence level', ['90%', '95%', '99%'], 
            horizontal=True, 
            label_visibility='collapsed')
        
        alpha = round(1-float(confidence_level.strip('%'))/100, 2)

        # User selection: Number of Resamplings
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Number of Resamplings</span>", 
            unsafe_allow_html=True)
        
        n_resamplings = st.slider(
            'Number of Resamplings', min_value=100, max_value=5000, value=2000, step=100, 
            label_visibility='collapsed')
            
    with col2:
        bootstrapping = Bootstrapping(
            original_sample=original_sample, 
            n_resamplings=n_resamplings, 
            statistic_text=stat_to_estimate, 
            alpha=alpha)
        
        bootstrapping.plot_sampling_distribution()


if __name__ == '__main__':
    main()