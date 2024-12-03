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

    Imagine tracking the pulse of LLM-related topics dynamically. Could we identify 
    emerging trends early enough to inform decision-making or research directions? 

    Use the widgets below to interact with the data and explore the topic 
    world of LLM-based reserach domains.
    """
)
# Load the LLM-related dataset
@st.cache_data
def load_llm_data():
    df_llm = pd.read_csv("data/LLM_related.csv", parse_dates=["update_date"])
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



# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    df = pd.read_csv("data/movies_genres_summary.csv")
    return df


df = load_data()

# Show a multiselect widget with the genres using `st.multiselect`.
genres = st.multiselect(
    "Genres",
    df.genre.unique(),
    ["Action", "Adventure", "Biography", "Comedy", "Drama", "Horror"],
)

# Show a slider widget with the years using `st.slider`.
years = st.slider("Years", 1986, 2006, (2000, 2016))

# Filter the dataframe based on the widget input and reshape it.
df_filtered = df[(df["genre"].isin(genres)) & (df["year"].between(years[0], years[1]))]
df_reshaped = df_filtered.pivot_table(
    index="year", columns="genre", values="gross", aggfunc="sum", fill_value=0
)
df_reshaped = df_reshaped.sort_values(by="year", ascending=False)


# Display the data as a table using `st.dataframe`.
st.dataframe(
    df_reshaped,
    use_container_width=True,
    column_config={"year": st.column_config.TextColumn("Year")},
)

# Display the data as an Altair chart using `st.altair_chart`.
df_chart = pd.melt(
    df_reshaped.reset_index(), id_vars="year", var_name="genre", value_name="gross"
)
chart = (
    alt.Chart(df_chart)
    .mark_line()
    .encode(
        x=alt.X("year:N", title="Year"),
        y=alt.Y("gross:Q", title="Gross earnings ($)"),
        color="genre:N",
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)


