import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Define the profiling categories to be analyzed
profiling_categories_to_analyze = [
    'Occupazione',
    'Tipologia di organizzazione presso cui lavori',
    'Organizzazione presso cui lavori o studi',
    'Seniority',
    'Area aziendale',
    'Settore produttivo'
]

# Set up the Streamlit page configuration
st.set_page_config(page_title="Participant Profiling Dashboard", layout="wide")

st.title("Participant Profiling Dashboard")
st.write("Upload an Excel file to analyze participant profiling data.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read the Excel file into a pandas DataFrame
        df_uploaded = pd.read_excel(uploaded_file)
        st.success("File successfully loaded!")
        st.subheader("First 5 rows of the uploaded data:")
        st.dataframe(df_uploaded.head())

        st.subheader("\nProfiling Category Analysis:")

        # For each profiling category, generate and display visualizations
        for category in profiling_categories_to_analyze:
            if category in df_uploaded.columns:
                st.markdown(f"### {category}")
                # Calculate value counts, dropping NaN values for cleaner output
                value_counts = df_uploaded[category].value_counts(dropna=True)

                if not value_counts.empty:
                    # Create a bar chart using matplotlib and display with st.pyplot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts.plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title(f'Distribution of {category}')
                    ax.set_xlabel(category)
                    ax.set_ylabel('Number of Participants')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    # Close the figure to free memory
                    plt.close(fig)
                else:
                    st.info(
                        f"No data available for '{category}' after dropping NaN values."
                    )
            else:
                st.warning(
                    f"The column '{category}' is not present in the uploaded file."
                )

    except Exception as e:
        st.error(
            f"Error reading file: {e}. Please ensure it is a valid Excel file."
        )
else:
    st.info("Please upload an Excel file to proceed with the analysis.")

# Note:
# The code above implements a Streamlit dashboard for analyzing participant profiling data from Excel files.
# To run this application locally, open a terminal and execute:
#   streamlit run streamlit_app.py
# This will start a local web server and provide a URL (e.g., http://localhost:8501) where you can interact with the dashboard.