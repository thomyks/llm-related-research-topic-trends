import altair as alt
import pandas as pd
import streamlit as st

# Show the page title and description.
st.set_page_config(page_title="LLM Topic Trends on ArXiv", page_icon="🤖")

# Title
st.title("Tracking LLM-based Research Trends on ArXiv corpus")

# Introduction
st.write(
    """
    Large Language Models (LLMs) have emerged as one of the most dynamic and transformative fields in 
    artificial intelligence, reshaping industries and academic landscapes alike.
    """
)

# Context and Significance
st.subheader("The Rise of LLM-based Research Domain")
st.write(
    """
    On November 30, 2022, OpenAI launched ChatGPT, a closed-source LLM as an 
    interactive web-based application. In just two years, millions of users have adopted this tool for a 
    wide array of tasks, including generating creative content, summarizing articles or essays, coding, language translations,
    or conducting data analysis.
    The immense popularity and impact of ChatGPT have spurred a wave of interest in LLMs across 
    academic and professional domains, prompting studies on both theoretical underpinnings 
    and practical applications of this technology.
    """
)

# ArXiv Overview
st.subheader("ArXiv Corpus")
st.write(
    """
    Since ChatGPT's release, **537,482 papers** have been published on ArXiv domain, reflecting the 
    rapid growth of scholarly contributions across mutitple disciplines. For those unfamiliar, ArXiv 
    is an open-access archive hosting nearly **2.4 million scholarly articles** spanning fields such as: 
    Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, Systems Science, Economics.

    With this app, we explore only LLM-related research trends.
    """
)

st.write(
    """
    ### LLM-Related Research Domain

    The "LLM-related research domain" encompasses a diverse range of concepts, frameworks, methodologies, and technologies centered on developing and applying LLMs. Below are the key categories that define this domain:

    #### 1. Core Models and Architectures
    Focus on designing the foundational structures of LLMs, including Transformer-based architectures, attention mechanisms, and scaling laws. Emphasize the theoretical and practical underpinnings that define a model’s capability.

    #### 2. Learning Paradigms
    Define methodologies that enable LLMs to adapt and generalize across tasks, such as few-shot, zero-shot, fine-tuning, and RLHF. Highlight approaches for improving task-specific performance without major architectural changes.

    #### 3. Optimization Techniques
    Explore strategies to enhance computational efficiency, scalability, and resource-friendliness, including quantization, pruning, and lightweight architectures (e.g., DistilBERT). Focus on making existing models more practical for deployment.

    #### 4. Applications and Use Cases
    Showcase real-world deployments of LLMs in tasks like conversational AI, Retrieval-Augmented Generation, and domain-specific automation. Emphasize the impact of LLMs across industries like healthcare, finance, and education.

    #### 5. Societal Impacts and Ethics
    Examine the societal implications of LLM deployment, including addressing issues of bias, fairness, transparency, and equitable access. Advocate for responsible and inclusive AI development practices.

    #### 6. Infrastructure and Tools
    Delve into the technical backbone that supports LLMs, such as APIs, libraries, deployment frameworks, and interoperability solutions. Ensure this category focuses on enabling efficient development and seamless integration of LLM systems.

    #### 7. Evaluation and Benchmarks
    Center on the frameworks, datasets, and metrics (e.g., GLUE, BLEU, BERTScore) that quantitatively assess model performance, robustness, fairness, and usability. Highlight the role of evaluation in driving iterative improvements.
    """
)


# Purpose
st.subheader("The App")
st.write(
    """
    This application provides an interactive tool to track and analyze **weekly trends** within 
    LLM-related research on the ArXiv platform between 2022-11-30 and 2024-11-29. By identifying shifting focus areas, 
    we aim to uncover valuable insights into the evolution of research themes and applications.
    """
)

# Per-week research articles
st.markdown("### Number of Published LLM-related Articles per Week")
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
        height=400,
        width=800
    )
    .interactive()  # Enables zooming and panning
)


# Display the chart in the Streamlit app
st.altair_chart(chart, use_container_width=True)





st.markdown("### Entity Trends in LLM-Related Research Papers")

st.write(
    """
    The following dashboard highlights trends in paper abstracts the top 500 entities related to LLM-domain over a defined timespan (2022 November–2024 November). You can explore its trends and interactions with related entities like **fine-tuning**, **embeddings**, and other model-related terminologies.
    """
)

# Load the data from a CSV file
@st.cache_data
def load_data():
    df = pd.read_csv("data/top_500_entity_data.csv")
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

