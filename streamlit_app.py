import altair as alt
import pandas as pd
import streamlit as st

# Show the page title and description.
st.set_page_config(page_title="LLM Topic Trends on ArXiv", page_icon="🤖")
st.title("LLM-based Research Trends")
st.write(
    """
    Large Language Models (LLMs) have become one of the fastest-growing fields in 
    artificial intelligence, reshaping industries and research landscapes. 

    This app serves as a tool for tracking the topic trends within LLM-related research domain 
    on a **weekly basis**. By examining trends, we aim to uncover insights into 
    the shifting focus areas of research and applications. 

   On November 30, 2022, approximately two years ago, ChatGPT, the closed-source Large Language Model (LLM), 
   was introduced as an interactive web-based application. Over the past two years, millions of users have embraced
  this tool to assist with a wide range of tasks, including generating creative content, summarizing articles or essays, 
  coding, translating languages, or conducting data analysis. Its impact has also sparked growing interest across different
    research domains, leading to studies that explore both the theoretical foundations and practical applications of this technology.

On the ArXiv platform, since the date of the release of the ChatGPT, 537,482 papers have been published. For those unfamiliar with ArXiv, 
it is an open-access archive hosting nearly 2.4 million scholarly articles across diverse fields, including physics, mathematics, 
computer science, quantitative biology, quantitative finance, statistics, electrical engineering, systems science, and economics.

    """
)

# The Plot of the 2 years-weekly papers
# Load the LLM-related dataset
@st.cache_data
def load_llm_data():
    df_llm = pd.read_csv("data/filtered_dataset.csv", parse_dates=["update_date"])
    return df_llm

df_llm = load_llm_data()

# Preprocess the data: Group by week and count articles
df_llm['week'] = df_llm['update_date'].dt.to_period('W').apply(lambda r: r.start_time)
articles_per_week = df_llm.groupby('week').size().reset_index(name='num_articles')

# Altair plot to display the number of published articles per week
chart = (
    alt.Chart(articles_per_week)
    .mark_line(point=alt.OverlayMarkDef(filled=True, size=50))  # Adds points to the line
    .encode(
        x=alt.X(
            "week:T",
            title="Week",
            axis=alt.Axis(format="%b %Y", tickCount="month"),  # Format as 'Month Year'
        ),
        y=alt.Y("num_articles:Q", title="Number of Articles"),
        tooltip=[
            alt.Tooltip("week:T", title="Week"),
            alt.Tooltip("num_articles:Q", title="Number of Articles"),
        ]
    )
    .properties(
        title="Number of Published LLM-related Articles per Week",
        height=400,
        width=800
    )
    .interactive()  # Enables zooming and panning
)


# Display the chart in the Streamlit app
st.altair_chart(chart, use_container_width=True)



# Code 3
# Load the data from a CSV file
import pandas as pd
import streamlit as st
import altair as alt

# Load the data from a CSV file
@st.cache_data
def load_data():
    df = pd.read_csv("data/top_500_entities_.csv")
    return df


# Load the dataset
df = load_data()

# Parse the 'Date' column into datetime objects
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Drop rows with invalid dates
df = df.dropna(subset=["Date"])

# Check if the dataset is empty after cleaning
if df.empty:
    st.error("No valid dates found in the dataset. Please check your data.")
else:
    # Extract valid min and max dates and convert them to Python datetime
    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()

    # Debugging: Ensure min_date and max_date are valid
    if min_date is None or max_date is None:
        st.error("Date range could not be determined due to invalid data.")
    else:
        # Configure the slider with valid datetime objects
        try:
            date_range = st.slider(
                "Date Range",
                min_date,
                max_date,
                (min_date, min_date + pd.Timedelta(days=365 * 2).to_pytimedelta()),  # Default 2-year span
                format="YYYY-MM-DD",
            )
        except Exception as e:
            st.error(f"Error setting up the slider: {e}")
            st.stop()

        # Show a multiselect widget with the entities
        entities = st.multiselect(
            "Entities",
            df["Entity"].unique(),
            df["Entity"].value_counts().head(10).index.tolist(),
        )

        # Filter the dataframe based on user input
        df_filtered = df[
            (df["Entity"].isin(entities)) &
            (df["Date"].between(date_range[0], date_range[1]))
        ]

        # Debugging: Print the filtered dataframe

        # Convert dates to weekly periods
        df_filtered["Week"] = df_filtered["Date"].dt.to_period("W").apply(lambda x: x.start_time)

        # Aggregate data: Count occurrences of each entity per week
        df_reshaped = df_filtered.groupby(["Week", "Entity"]).size().unstack(fill_value=0)
        df_reshaped = df_reshaped.sort_index(ascending=False)

        # Display the data as a table
        st.dataframe(
            df_reshaped,
            use_container_width=True,
        )

        # Prepare the data for charting
        df_chart = pd.melt(
            df_reshaped.reset_index(), id_vars="Week", var_name="Entity", value_name="Count"
        )
        chart = (
            alt.Chart(df_chart)
            .mark_line()
            .encode(
                x=alt.X("Week:T", title="Week"),
                y=alt.Y("Count:Q", title="Count"),
                color="Entity:N",
            )
            .properties(height=320)
        )

        # Display the data as a chart
        st.altair_chart(chart, use_container_width=True)
