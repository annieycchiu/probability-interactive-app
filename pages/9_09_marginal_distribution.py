import streamlit as st
import plotly.express as px
from utils.other_utils import add_logo, setup_sticky_header, add_title

colors = {
    'USF_Green': '#00543C',
    'USF_Yellow': '#FDBB30',
    'USF_Gray': '#75787B'
}

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Marginal Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='auto' # hides the sidebar on small devices and shows it otherwise
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Marginal Distribution'
        add_title(title)

        col1, _ ,col2, _, col3 = st.columns([0.30, 0.03, 0.30, 0.03, 0.30])
        with col1:
            # User selection: plot type of marginal x
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Marginal x Plot Type</span>", 
                unsafe_allow_html=True)
            marginal_x = st.radio(
                'Marginal x Plot Type', 
                ['histogram', 'box', 'violin', 'rug'], 
                horizontal=True, 
                label_visibility='collapsed')
            st.write('')

            # User selection: x variable
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>x variable</span>", 
                unsafe_allow_html=True)
            x = st.selectbox(
                'x variable', 
                ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 
                label_visibility='collapsed')
            st.write('')
        
        with col2:
            # User selection: plot type of marginal y
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Marginal y Plot Type</span>", 
                unsafe_allow_html=True)
            marginal_y = st.radio(
                'Marginal y Plot Type', 
                ['histogram', 'box', 'violin', 'rug'], 
                horizontal=True, 
                index=1,
                label_visibility='collapsed')
            st.write('')

            # User selection: y variable
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>y variable</span>", 
                unsafe_allow_html=True)
            y = st.selectbox(
                'y variable', 
                ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 
                index=1, 
                label_visibility='collapsed')
            
        with col3:
            # User selection: main plot type
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Main Plot Type</span>", 
                unsafe_allow_html=True)
            main_plot = st.radio(
                'Main Plot Type', 
                ['scatter', 'density heatmap'], 
                horizontal=True, 
                label_visibility='collapsed')
            st.write('')

            # User selection: sub-group
            st.write(
                "<span style='font-size:18px; font-weight:bold;'>Sub-group</span>", 
                unsafe_allow_html=True)
            if main_plot == 'scatter':
                sub_group = st.selectbox(
                    'Sub-group', 
                    [None, 'species'], 
                    index=1,
                    label_visibility='collapsed')
            elif main_plot == 'density heatmap':
                sub_group = st.selectbox(
                    'Sub-group', 
                    [None], 
                    label_visibility='collapsed')

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    # Load the iris dataset
    df = px.data.iris()

    if main_plot == 'scatter':
        if sub_group == None:
            fig = px.scatter(
                df, x=x, y=y, marginal_x=marginal_x, marginal_y=marginal_y)
            fig.update_traces(marker=dict(color=colors['USF_Green']))
        else:
            # Define a dictionary to map colors to species
            color_map = {
                'setosa': colors['USF_Green'], 
                'versicolor': colors['USF_Yellow'], 
                'virginica': colors['USF_Gray']}
            
            fig = px.scatter(
                df, x=x, y=y, marginal_x=marginal_x, marginal_y=marginal_y, color=sub_group, 
                color_discrete_map=color_map)
    
    elif main_plot == 'density heatmap':
        fig = px.density_heatmap(
            df, x=x, y=y, marginal_x=marginal_x, marginal_y=marginal_y,
            color_continuous_scale=[colors['USF_Green'], colors['USF_Yellow'], colors['USF_Gray']])
    

    col11, col12 = st.columns([0.45, 0.55])
    with col11:
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Iris Dataset</span>", 
            unsafe_allow_html=True)
        
        st.write(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']])

    with col12:
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Marginal Distribution Plot</span>", 
            unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
