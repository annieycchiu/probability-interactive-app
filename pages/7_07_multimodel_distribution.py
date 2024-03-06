import streamlit as st
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

from utils.other_utils import add_logo

colors = {
    'USF_Green': '#00543C',
    'USF_Yellow': '#FDBB30',
    'USF_Gray': '#75787B'
}

def get_input_params(n_modes):
    default_params = [
        [-10.0, 1.0, 2000], 
        [0.0, 3.0, 1000], 
        [20.0, 5.0, 4000], 
        [35.0, 2.0, 2000], 
        [50.0, 3.0, 5000]]
    
    input_params = []
    for i in range(n_modes):
        col1, col2, col3 = st.columns(3)
        with col1:
            mean = st.slider(
                'Mean (μ)', min_value=-50.0, max_value=50.0, 
                value=default_params[i][0], step=0.01, key=f'mean_{i}')
        with col2:
            std_dev = st.slider(
                'Standard Deviation (σ)', min_value=0.1, max_value=10.0, 
                value=default_params[i][1], step=0.01, key=f'std_{i}')
        with col3:
            size = st.slider(
                'Number of Data Points', min_value=1000, max_value=5000, 
                value=default_params[i][2], step=100, key=f'size_{i}')

        params = {'mean': mean, 'std_dev': std_dev, 'size': size}
        input_params.append(params)

    return input_params

class MultimodelDistribution():
    def __init__(self, input_params, colors=colors):
        self.input_params = input_params
        self.colors = colors

        self.data = None
        self.generate_data()
   
    def generate_data(self):
        data = []
        for i in range(len(self.input_params)):
            # print(self.input_params[i])
            data.append(np.random.normal(
                loc=self.input_params[i]['mean'], 
                scale=self.input_params[i]['std_dev'], 
                size=self.input_params[i]['size']))
            
        self.data = data

    def plot_distribution(self):
        # Generate multiple modes dataset
        data = np.concatenate(self.data)

        # Create a histogram to estimate the density
        hist, bins = np.histogram(data, bins=100, density=True)

        # # Create a histogram trace
        hist_trace = go.Bar(
            x=bins, y=hist, 
            name='Histogram',
            marker_color=self.colors['USF_Green'])

        # hist_trace = go.Histogram(
        #     x=data,
        #     histnorm='probability density',
        #     marker_color=self.colors['USF_Green'],
        #     name='Histogram'
        # )

        # Calculate kernel density estimate
        kde = sns.kdeplot(data, bw_adjust=0.5)  # adjust bandwidth for smoother KDE

        # Create a kernel density estimate trace
        kde_trace = go.Scatter(
            x=kde.get_lines()[0].get_data()[0],
            y=kde.get_lines()[0].get_data()[1],
            mode='lines',
            name='Kernel Density Estimate',
            marker_color=self.colors['USF_Yellow'])

        # Create layout
        if len(self.data) == 2:
            title = "Bimodal Distribution"
        else:
            title = "Multimodal Distribution"

        layout = go.Layout(
            title=title,
            xaxis=dict(title='Value'),
            yaxis=dict(title='Density'),
            legend=dict(
                # orientation="h", # horizontal legend
                yanchor="bottom", y=1.02,
                xanchor="right", x=1))

        # Create figure
        fig = go.Figure(data=[hist_trace, kde_trace], layout=layout)

        # Show plot
        st.plotly_chart(fig)

        
                

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Multimodel Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add logo
    add_logo()

    # Set up title
    st.write(
        f"<span style='font-size:35px; font-weight:bold;'>Multimodel Distribution</span>", 
        unsafe_allow_html=True)
    
    st.write('')

    n_modes = st.radio('Number of Modes', [2, 3, 4, 5], horizontal=True)

    col1, _,col2 = st.columns([0.45, 0.05, 0.5])
    with col1:
        with st.expander('Adjust the plot :point_down:'):
            input_params = get_input_params(n_modes)

    with col2:
        multiModel = MultimodelDistribution(input_params)
        multiModel.plot_distribution()

if __name__ == "__main__":
    main()