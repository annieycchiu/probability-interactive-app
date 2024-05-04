import streamlit as st
import pandas as pd

from utils.other_utils import add_logo, setup_sticky_header, add_title

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Conditional Distribution',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Conditional Distribution'
        add_title(title)

        st.write(
            f"<span style='font-size:20px;'>The conditional distribution of "
            f"<span style='color:#00543C; font-weight:bold;'>Y</span> given "
            f"<span style='color:#FDBB30; font-weight:bold;'>X </span> is "
            f"<span style='color:#FDBB30; font-weight:bold;'>X value</span></span>", 
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns([0.1, 0.9])

        with col1:
            st.write("<span style='color:#00543C; font-size:18px; font-weight:bold;'>Y</span>", unsafe_allow_html=True)
            st.write("<span style='color:#FDBB30; font-size:18px; font-weight:bold;'>X</span>", unsafe_allow_html=True)
            st.write("<span style='color:#FDBB30; font-size:18px; font-weight:bold;'>X value</span>", unsafe_allow_html=True)

        with col2:
            Y = st.radio('Y', ['Course', 'Gender'],horizontal=True, label_visibility='collapsed')

            X = 'Course' if Y == 'Gender' else 'Gender'
            st.write(X)

            X_choices = ['Stats & Prob', 'Python', 'EDA', 'Machine Learning', 'Database'] if Y == 'Gender' else ['Male', 'Female']
            X_val = st.radio('X Value', X_choices,horizontal=True, label_visibility='collapsed')

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)

    # Create sample dataset
    df = pd.DataFrame({
        'Stats & Prob': [22, 8, 30],
        'Python': [8, 4, 12],
        'EDA': [13, 8, 21],
        'Machine Learning': [15, 10, 25],
        'Database': [7, 5, 12],
        'Total': [65, 35, 100],
    })

    row_names = ['Male', 'Female', 'Total']
    df.index = row_names

    st.dataframe(df, use_container_width=True)
    
    st.write(
        f"<span style='font-size:20px;'>The conditional distribution of "
        f"<span style='color:#00543C; font-weight:bold;'>{Y}</span> given "
        f"<span style='color:#FDBB30; font-weight:bold;'>{X} </span> is "
        f"<span style='color:#FDBB30; font-weight:bold;'>{X_val}</span></span>", 
        unsafe_allow_html=True
    )

    if Y == 'Course':
        Ys = df.columns[:-1]
        counts = df.loc[X_val][:-1]
        total = df.loc[X_val][-1]
        percentages = df.loc[X_val][:-1]/total

        col1, col2 = st.columns([0.2, 0.8])

    elif Y == 'Gender':
        Ys = df.index[:-1]
        counts = df[X_val][:-1]
        total = df[X_val][-1]
        percentages = df[X_val][:-1]/total

        col1, col2 = st.columns([0.1, 0.9])
        
    
    with col1:
        for y in Ys:
            st.write(f"<span style='font-size:18px; font-weight:bold;'>{y}", unsafe_allow_html=True)

    with col2:
        for cnt, p in zip(counts, percentages):
            st.write(
                f"<span style='font-size:18px;'>"
                f"<span style='color:#00543C; font-weight:bold;'>{cnt}</span> / "
                f"<span style='color:#FDBB30; font-weight:bold;'>{total} </span> = "
                f"<span style='font-weight:bold;'>{p:.2f}</span></span>", 
                unsafe_allow_html=True
            )

if __name__ == '__main__':
    main()