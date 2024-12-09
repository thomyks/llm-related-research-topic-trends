import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
from PyPDF2 import PdfReader
import together
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import requests
import numpy as np
import io
import matplotlib
matplotlib.rcParams["figure.dpi"] = 72
import datamapplot as dmp


# Correct order for set_page_config
st.set_page_config(
    page_title="LLM Research Trends",
    page_icon="🤖",
    layout="wide",
    menu_items={
        "About": "🔍 Explore and Extract LLM Trends!"
    }
)

# Add custom meta tags for SEO and previews
components.html("""
    <meta name="description" content="🔍 Explore LLM Trends! Extract insights to uncover LLM evolution.">
    <meta property="og:title" content="Explore and Extract LLM Trends!">
    <meta property="og:description" content="🔍 Explore LLM Trends! Extract insights to uncover LLM evolution.">
    <meta property="og:image" content="https://example.com/path-to-your-preview-image.png"> <!-- Replace with your image URL -->
    <meta property="og:url" content="https://llmtrends.streamlit.app"> <!-- Replace with your app URL -->
""", height=0)


# Title
st.title("Discover, Analyze, and Export Insights on Your Favorite LLM Research Topics")

section = st.sidebar.radio(
    "Go to",
    [ "Topic Tracking","Topic Overview","LLM-related Research Overview", "Entity Tracking", "Topic Discovery"],
    index=0
)

# Sidebar Information
st.sidebar.markdown("### About this App")
st.sidebar.write(
    """
Keeping pace with the rapid surge of LLM research can feel like chasing a moving target.
This interactive app analyzes over 40,000 ArXiv LLM-related papers from November 30, 2022, to November 29, 2024, uncovering weekly/monthly trends, spotlighting key topics, and mapping the evolution of this groundbreaking field.    """
)



# Topic Trends
# Topic Trends
if section == "Topic Tracking":
    # Load data from the CSV file
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/Datamap/Concatenated_LLM_Subdomains_embeddings.csv")
        return df

    # Load the dataset
    df = load_data()

    # Ensure the necessary columns exist in the dataset
    if "Categories" in df.columns and "Subdomain" in df.columns and "Human_Readable_Topic" in df.columns:
        # List of available categories (domains)
        available_categories = df["Categories"].unique().tolist()

        # Create columns layout for domain (Categories) and subdomain (Subdomain) selection
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Select LLM-related Domain/s")
            selected_categories = []
            for category in available_categories:
                if st.checkbox(category, value=False, key=category):  # Default is not selected
                    selected_categories.append(category)

        with col2:
            st.markdown("### Select Subdomain/s")
            available_subdomains = df[df["Categories"].isin(selected_categories)]["Subdomain"].unique().tolist()
            selected_subdomains = []
            for subdomain in available_subdomains:
                if st.checkbox(subdomain, value=False, key=subdomain):  # Default is not selected
                    selected_subdomains.append(subdomain)

        # Filter data based on the selected categories and subdomains
        df_filtered = df[df["Categories"].isin(selected_categories)]
        df_subdomain_filtered = df_filtered[df_filtered["Subdomain"].isin(selected_subdomains)]

        # Check if the filtered dataset is empty
        if df_subdomain_filtered.empty:
            st.warning(f"No data available for the selected domain(s): {', '.join(selected_categories)} and subdomain(s): {', '.join(selected_subdomains)}.")
        else:
                       # Datamapplot visualization integration
            if "2d_coords" in df_subdomain_filtered.columns and "title" in df_subdomain_filtered.columns:
                # Ensure the '2d_coords' column is in the correct format
                df_subdomain_filtered["2d_coords"] = df_subdomain_filtered["2d_coords"].apply(eval)  # Convert string to list if needed
                coords_array = np.array(df_subdomain_filtered["2d_coords"].tolist(), dtype=np.float32)
                labels_array = df_subdomain_filtered["Human_Readable_Topic"].to_numpy()
                hover_data = df_subdomain_filtered["title"].tolist()

                import datamapplot as dmp

                plot = dmp.create_interactive_plot(
                    coords_array,
                    labels_array,
                    hover_text=hover_data,
                    font_family="Playfair Display SC",
                    title=f"Datamap for Subdomain(s): {', '.join(selected_subdomains)}",
                    sub_title="An interactive visualization of selected data",
                    logo=None,  # Add logo if desired
                    on_click="window.open(`http://google.com/search?q=\"{hover_text}\"`)",
                    enable_search=True,
                    darkmode=True,
                )

                # Render the datamapplot visualization in Streamlit
                try:
                    if hasattr(plot, "to_html"):
                        st.components.v1.html(plot.to_html(), height=800, scrolling=True)
                    else:
                        # Save the plot to an HTML file and render
                        plot.save("subdomain_plot.html")
                        with open("subdomain_plot.html", "r") as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Unable to render the plot. Error: {e}")

            # Step 3: Topic Trends Section
            st.markdown(f"### Topic Trends for Subdomain(s): {', '.join(selected_subdomains)}")

            # Parse the 'update_date' column into datetime objects
            df_subdomain_filtered["update_date"] = pd.to_datetime(df_subdomain_filtered["update_date"], errors="coerce")

            # Add a column for monthly periods
            df_subdomain_filtered["Month_Start"] = df_subdomain_filtered["update_date"].dt.to_period("M").apply(lambda x: x.start_time)

            # Count articles per topic per month
            df_grouped = df_subdomain_filtered.groupby(["Month_Start", "Human_Readable_Topic"]).size().reset_index(name="Monthly_Count")

            # Calculate cumulative sum for each topic
            df_grouped["Cumulative_Count"] = df_grouped.groupby("Human_Readable_Topic")["Monthly_Count"].cumsum()

            # Get the total count for each topic to determine the top 10 topics
            topic_totals = df_grouped.groupby("Human_Readable_Topic")["Cumulative_Count"].max().reset_index()
            top_topics = topic_totals.sort_values(by="Cumulative_Count", ascending=False)["Human_Readable_Topic"].head(1).tolist()

            # Allow the user to toggle topics
            topics = df_grouped["Human_Readable_Topic"].unique()
            selected_topics = st.multiselect(
                "Choose Topic/s",
                options=topics,
                default=top_topics,  # Default to the top topic
            )

            # Filter the grouped data based on selected topics
            df_grouped_filtered = df_grouped[df_grouped["Human_Readable_Topic"].isin(selected_topics)]

              # Add a select box for choosing the plot type
            plot_type = st.selectbox(
                "Select Plot Type",
                options=["Monthly Trend", "Cumulative Trend", "Normalized Cumulative Trend", "Heatmap Trend"],
                index=0,  # Default to the first option
            )

            

            # Generate the appropriate chart based on user selection
            if plot_type == "Monthly Trend":
                base_chart = (
                    alt.Chart(df_grouped_filtered)
                    .mark_line()
                    .encode(
                        x=alt.X("Month_Start:T", title="Month Start"),
                        y=alt.Y("Monthly_Count:Q", title="Monthly Count"),
                        color=alt.Color(
                            "Human_Readable_Topic:N",
                            title="Topics",
                            legend=alt.Legend(
                                orient="right",  # Position the legend to the right of the chart
                                titleFontSize=12,  # Increase title font size
                                labelFontSize=10,  # Adjust label font size
                                labelLimit=500,  # Prevent truncation of long labels
                                direction="vertical",  # Ensure the legend is vertical
                            ),
                        ),
                    )
                )
                st.altair_chart(base_chart, use_container_width=True)

            elif plot_type == "Cumulative Trend":
                base_chart = (
                    alt.Chart(df_grouped_filtered)
                    .mark_line()
                    .encode(
                        x=alt.X("Month_Start:T", title="Month Start"),
                        y=alt.Y("Cumulative_Count:Q", title="Cumulative Paper Count"),
                        color=alt.Color(
                            "Human_Readable_Topic:N",
                            title="Topics",
                            legend=alt.Legend(
                                orient="right",  # Position the legend to the right of the chart
                                titleFontSize=12,  # Increase title font size
                                labelFontSize=10,  # Adjust label font size
                                labelLimit=500,  # Prevent truncation of long labels
                                direction="vertical",  # Ensure the legend is vertical
                            ),
                        ),
                    )
                )
                st.altair_chart(base_chart, use_container_width=True)

            elif plot_type == "Normalized Cumulative Trend":
                if df_grouped_filtered.empty:
                    st.warning("No data available for the selected topics. Please select at least one topic.")
                else:
                    # Calculate the total cumulative count for each topic
                    topic_totals = df_grouped.groupby("Human_Readable_Topic")["Cumulative_Count"].transform("max")

                    # Normalize the cumulative count by dividing by the total count for each topic
                    df_grouped_filtered["Normalized_Cumulative_Count"] = (
                        df_grouped_filtered["Cumulative_Count"] / topic_totals
                    )

                    # Plot the normalized cumulative trend
                    normalized_chart = (
                        alt.Chart(df_grouped_filtered)
                        .mark_line()
                        .encode(
                            x=alt.X("Month_Start:T", title="Month Start"),
                            y=alt.Y("Normalized_Cumulative_Count:Q", title="Normalized Cumulative Paper Count"),
                            color=alt.Color(
                                "Human_Readable_Topic:N",
                                title="Topics",
                                legend=alt.Legend(
                                    orient="right",
                                    titleFontSize=12,
                                    labelFontSize=10,
                                    labelLimit=500,
                                    direction="vertical",
                                ),
                            ),
                        )
                    )
                    st.altair_chart(normalized_chart, use_container_width=True)


            elif plot_type == "Heatmap Trend":
                if not df_grouped_filtered.empty:

                    # Ensure 'Month_Start' is in datetime format
                    df_grouped_filtered["Month_Start"] = pd.to_datetime(df_grouped_filtered["Month_Start"], errors="coerce")

                    # Create a complete range of months and topics
                    all_months = pd.date_range(
                        start=df_grouped_filtered["Month_Start"].min(),
                        end=df_grouped_filtered["Month_Start"].max(),
                        freq="MS"  # Monthly intervals
                    )
                    all_topics = df_grouped_filtered["Human_Readable_Topic"].unique()

                    # Create a DataFrame with all combinations of months and topics
                    complete_index = pd.MultiIndex.from_product(
                        [all_months, all_topics],
                        names=["Month_Start", "Human_Readable_Topic"]
                    )
                    complete_data = pd.DataFrame(index=complete_index).reset_index()

                    # Merge the existing data with the complete data
                    df_complete = pd.merge(
                        complete_data,
                        df_grouped_filtered,
                        on=["Month_Start", "Human_Readable_Topic"],
                        how="left"
                    ).fillna({"Monthly_Count": 0})  # Fill missing counts with 0

                    # Pivot the data to create a 2D matrix for the heatmap
                    heatmap_data = df_complete.pivot(
                        index="Human_Readable_Topic", columns="Month_Start", values="Monthly_Count"
                    ).fillna(0)  # Fill any remaining NaNs with 0

                    # Create the Plotly heatmap
                    import plotly.express as px

                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x="Month Start", y="Topics", color="Monthly Count"),
                        x=heatmap_data.columns.strftime('%b %Y'),  # Format months for display
                        y=heatmap_data.index,
                        color_continuous_scale="Blues"                    )

                    # Update layout for better readability
                    fig.update_layout(
                        xaxis=dict(tickangle=-45),  # Rotate x-axis labels
                        height=600,
                        width=1000,
                    )

                    # Display the Plotly heatmap
                    st.plotly_chart(fig, use_container_width=True)

                
                else:
                    st.warning("No data available for the selected topics.")
            # Display detailed insights
            st.markdown("### Monthly Trends")
            st.write(f"Showing monthly trends for topics under subdomain/s: {', '.join(selected_subdomains)}.")

            # Add a data table for granular details
            st.dataframe(df_grouped_filtered, use_container_width=True)

            # Filter the original data to include rows with selected topics
            df_additional_info = df_filtered[df_filtered["Human_Readable_Topic"].isin(selected_topics)]
            
            # Dynamically update the markdown with selected topics
            if selected_topics:
                selected_topics_text = ", ".join(selected_topics)
                st.markdown(f"### Export the Paper Details in the CSV file!")
                st.write(f"Selected Topics: {selected_topics_text}")
            else:
                st.markdown("### Export the Paper Details in the CSV file!")
                st.write("No topics have been selected.")
            st.write(f"Export detailed information about research paper, including links, dates, titles, abstracts, topic label, categories, submitter, and monthly trends for topics under the subdomain/s: **{', '.join(selected_subdomains)}**.")

            # Display the filtered dataset for download
            st.dataframe(df_additional_info, use_container_width=True, column_config={"id":st.column_config.LinkColumn()})

             # Generate a filename based on selected topics
            if selected_topics:
                # Create a shortened version of the topics for the filename
                topics_for_filename = "_".join(selected_topics[:3])  # Use only the first 3 topics for brevity
                if len(selected_topics) > 3:
                    topics_for_filename += "_and_more"
                file_name = f"research_papers_topics_{topics_for_filename}.csv"
            else:
                file_name = "topics_no_selection.csv"

            # Provide download option for the CSV
            csv = df_additional_info.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
            )    
 


# Main Section Rendering
elif section == "LLM-related Research Overview":
    # Introduction
    st.write(
        """
        Large Language Models (LLMs) have emerged as one of the most dynamic and transformative fields in 
        artificial intelligence, reshaping industries and academic landscapes alike.
        """
    )

    # Context and Significance
    st.subheader("The Rise of LLM-based Research")
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
    st.subheader("LLM-based Research on ArXiv Corpus")
    st.write(
        """
        Since ChatGPT's release, **537,482 papers** have been published on ArXiv domain, reflecting the 
        rapid growth of scholarly contributions across mutitple disciplines. For those unfamiliar, ArXiv 
        is an open-access archive hosting nearly **2.4 million scholarly articles** spanning fields such as: 
        Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, Systems Science, Economics.

        """
    )

    # Per-week research articles
    st.markdown("### Number of Published LLM-related Articles on ArXiv per Week")
    # The Plot of the 2 years-weekly papers
    # Load the LLM-related dataset
    @st.cache_data
    def load_llm_data():
        df_llm = pd.read_csv("data/LLM_domain.csv", parse_dates=["update_date"])
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


    # Datamapplot
    # Step 1: Load Local Data with Caching
    @st.cache_data
    def load_coordinates_data():
        df = pd.read_csv("data/Datamap/Concatenated_LLM_Subdomains_embeddings.csv")
        if isinstance(df['2d_coords'].iloc[0], str):
            df['2d_coords'] = df['2d_coords'].apply(eval)
        return df

    df = load_coordinates_data()

    # Extract coordinates and labels
    coords_array = np.array(df['2d_coords'].tolist())
    labels_array = df['Human_Readable_Topic'].to_numpy()

    # Prepare hover data
    hover_data = df['title'].tolist() if 'title' in df.columns else [f"Topic {i}" for i in range(len(df))]

    # Step 2: Generate the Interactive Plot
    plot = dmp.create_interactive_plot(
        coords_array,
        labels_array,
        hover_text=hover_data,
        font_family="Playfair Display SC",
        title="Discover Data Landscape",
        sub_title="A data map based on the LLM-related research.",
        logo="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/ArXiv_logo_2022.svg/512px-ArXiv_logo_2022.svg.png",
        logo_width=180,
        on_click="window.open(`http://google.com/search?q=\"{hover_text}\"`)",
        enable_search=True,
        darkmode=True,
    )

    # Step 3: Streamlit App
    st.title(" Data Landscape")
    st.subheader("A data map based on LLM-related research papers.")

    # Render the plot in Streamlit
    try:
        # Check if the plot object has `to_html` method for rendering
        if hasattr(plot, 'to_html'):
            st.components.v1.html(plot.to_html(), height=800, scrolling=True)
        else:
            # Save the plot to an HTML file and render
            plot.save("custom_plot.html")
            with open("custom_plot.html", "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Unable to render the plot. Error: {e}")


    # # Step 1: Load Data
    # base_url = "https://github.com/TutteInstitute/datamapplot/raw/main/examples"

    # # Load the data map file
    # data_map_file = requests.get(f"{base_url}/arxiv_ml_data_map.npy")
    # arxivml_data_map = np.load(io.BytesIO(data_map_file.content))

    # # Load label layers
    # arxivml_label_layers = []
    # for layer_num in range(5):
    #     label_file = requests.get(f"{base_url}/arxiv_ml_layer{layer_num}_cluster_labels.npy")
    #     arxivml_label_layers.append(np.load(io.BytesIO(label_file.content), allow_pickle=True))

    # # Step 2: Define Hover Data
    # # Example hover data - replace with your own or dynamically generate based on your data
    # arxiv_hover_data = ["Paper {}".format(i) for i in range(len(arxivml_data_map))]

    # # Step 3: Generate the Interactive Plot
    # plot = dmp.create_interactive_plot(
    #     arxivml_data_map,
    #     arxivml_label_layers[0],
    #     arxivml_label_layers[2],
    #     arxivml_label_layers[4],
    #     hover_text=arxiv_hover_data,
    #     font_family="Playfair Display SC",
    #     title="ArXiv Machine Learning Landscape",
    #     sub_title="A data map of papers from the Machine Learning section of ArXiv",
    #     logo="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/ArXiv_logo_2022.svg/512px-ArXiv_logo_2022.svg.png",
    #     logo_width=180,
    #     on_click="window.open(`http://google.com/search?q=\"{hover_text}\"`)",
    #     enable_search=True,
    #     darkmode=True,
    # )

    # # Step 4: Streamlit App
    # st.title("ArXiv Machine Learning Landscape")
    # st.subheader("A data map of papers from the Machine Learning section of ArXiv")

    # # Render the plot in Streamlit
    # try:
    #     # Check if the plot object has `to_html` method for rendering
    #     if hasattr(plot, 'to_html'):
    #         st.components.v1.html(plot.to_html(), height=800, scrolling=True)
    #     else:
    #         # Save the plot to an HTML file and render
    #         plot.save("arxiv_plot.html")
    #         with open("arxiv_plot.html", "r") as f:
    #             html_content = f.read()
    #         st.components.v1.html(html_content, height=800, scrolling=True)
    # except Exception as e:
    #     st.error(f"Unable to render the plot. Error: {e}")

# Entity Tracking
elif section == "Entity Tracking":
    st.title("Entity Trends in LLM-Related Research Papers")
    st.write(
        """
        The following dashboard highlights trends in paper abstracts the top 500 entities related to LLM-domain over a defined timespan (2022 November–2024 November). 
        You can explore its trends and interactions with related entities like **fine-tuning**, **embeddings**, and other model-related terminologies.

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
            # Show a multiselect widget with the entities
            entities = st.multiselect(
                "Entities",
                df["Entity"].unique(),
                df["Entity"].value_counts().head(5).index.tolist(),
            )

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

            # Filter the dataframe based on user inputs
            df_filtered = df[
                (df["Entity"].isin(entities)) &
                (df["Date"].between(date_range[0], date_range[1]))
            ]

            # Convert dates to weekly periods
            df_filtered["Week"] = df_filtered["Date"].dt.to_period("W").apply(lambda x: x.start_time)

            # Aggregate data: Count occurrences of each entity per week
            df_reshaped = df_filtered.groupby(["Week", "Entity"]).size().unstack(fill_value=0)
            df_reshaped = df_reshaped.sort_index(ascending=False)

            # Prepare the filtered data for charting
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

            # Display the filtered data as a chart
            st.altair_chart(chart, use_container_width=True)

            # Display the filtered data as a table below the chart
            st.dataframe(
                df_reshaped,
                use_container_width=True,
            )


# 
elif section == "Topic Overview":
        # Load data from CSV
    @st.cache_data
    def load_data():
        return pd.read_csv("data/LLM_related_domainss.csv")

    df = load_data()

    # Ensure necessary columns exist
    if {"Categories", "Subdomain", "Human_Readable_Topic"}.issubset(df.columns):
        st.write("### Hierarchical Topic Knowledge of LLM-Related Research Domain")

        # Available categories
        available_categories = df["Categories"].unique().tolist()

        # Create category selector
        selected_category = st.radio(
            "Select Domain", options=available_categories, horizontal=True
        )

        # Filter data by selected category
        df_filtered = df[df["Categories"] == selected_category]

        if df_filtered.empty:
            st.warning(f"No data available for the selected category: {selected_category}.")
        else:
            # Calculate the value column for Sunburst
            value_df = (
                df_filtered.groupby(["Categories", "Subdomain", "Human_Readable_Topic"])
                .size()
                .reset_index(name="Value")
            )

            # Create Sunburst chart
            fig = px.sunburst(
                value_df,
                path=["Categories", "Subdomain", "Human_Readable_Topic"],
                values="Value",
                width=800,
                height=800,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Add dropdowns for manual filtering
            st.write("### Select Subdomain and Topic for more Details")
            subdomains = df_filtered["Subdomain"].unique().tolist()
            selected_subdomain = st.selectbox("Select Subdomain", options=["All"] + subdomains)

            if selected_subdomain != "All":
                df_filtered = df_filtered[df_filtered["Subdomain"] == selected_subdomain]

                topics = df_filtered["Human_Readable_Topic"].unique().tolist()
                selected_topic = st.selectbox("Select Topic", options=["All"] + topics)

                if selected_topic != "All":
                    df_filtered = df_filtered[df_filtered["Human_Readable_Topic"] == selected_topic]

            # Monthly trends section
            st.markdown("### Monthly Trends for Selected Topic Details")
            if selected_subdomain != "All":
                st.write(f"Showing trends for Subdomain: {selected_subdomain}")
            if "selected_topic" in locals() and selected_topic != "All":
                st.write(f"Showing trends for Topic: {selected_topic}")
            else:
                st.write("Showing overall trends for Category.")

            # Group by month and display details
            df_grouped = (
                df_filtered.groupby(["Categories", "Subdomain", "Human_Readable_Topic"])
                .size()
                .reset_index(name="Count")
            )

            st.dataframe(df_grouped, use_container_width=True)

                    # Prepare export data
            st.markdown("### Export the Paper Details in CSV")
            export_data = df_filtered[
                [
                    "title",
                    "id",
                    "abstract",
                    "Human_Readable_Topic",
                    "Categories",
                    "Subdomain",
                    "submitter",
                    "update_date",
                ]
            ]
            st.dataframe(export_data, use_container_width=True, column_config={"id":st.column_config.LinkColumn()})

            # Dynamically construct the file name
            domain_part = selected_category.replace(" ", "_")  # Replace spaces with underscores
            subdomain_part = selected_subdomain.replace(" ", "_") if selected_subdomain != "All" else "All_Subdomains"
            dynamic_file_name = f"research_paper_details_{domain_part}_{subdomain_part}.csv"

            # Allow download of CSV
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Download Paper Details CSV",
                data=csv,
                file_name=dynamic_file_name,
                mime="text/csv",
            )
    else:
        st.error("The required columns ('Categories', 'Subdomain', 'Human_Readable_Topic') are missing in the dataset.")
        # Descriptions Section
    st.write(
        """
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

elif section == "Topic Discovery":
    # Initialize Together AI client
    client = together.Client(api_key='e52dd14cb34eee0f4eab33d6f8ea5202276732546dc66be6394f80629e6c061f')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load datasets with caching
    @st.cache_data
    def load_topic_with_embeddings():
        dataset = pd.read_csv("data/Topic_with_Embeddings.csv")
        dataset['embedding_array'] = dataset['Embedding'].apply(json.loads)
        return dataset

    @st.cache_data
    def load_llm_related_domains():
        return pd.read_csv("data/LLM_related_domainss.csv")

    # PDF text extraction
    def extract_text_from_pdf(file):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    # Title and Abstract extraction
    def extract_title_and_abstract(text):
        title = text.split('.')[0]  # First sentence
        start_idx = text.lower().find("abstract")
        end_idx = text.lower().find("introduction")
        abstract = text[start_idx + len("abstract"):end_idx].strip() if start_idx != -1 and end_idx != -1 else "Abstract not found."
        return title.strip(), abstract.strip()

    # Topic extraction using Together AI
    def extract_topics(title, abstract):
        prompt = f"""
        Based on the provided Title and Abstract of a research paper, extract a list of all the topics that might be necessary to conduct such research. 
        The output should be limited to a maximum of 10 topics.

        Title: {title}

        Abstract: {abstract}

        Provide the output in the format:
        Topics: [List of topics]
        """
        response = ""
        try:
            stream = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.1,
            )
            for chunk in stream:
                response += chunk.choices[0].delta.content or ""
        except Exception as e:
            st.error(f"Error during topic extraction: {e}")
            return None
        return response.strip()

    # Embedding computation
    def compute_embeddings(topics):
        return model.encode(topics, convert_to_numpy=True)

    # Find most similar topics
    def find_most_similar_topics(llm_topics, dataset, top_n=5):
        dataset_embeddings = np.vstack(dataset['embedding_array'].values)
        llm_embeddings = compute_embeddings(llm_topics)
        top_matches = []
        for i, llm_topic in enumerate(llm_topics):
            similarities = cosine_similarity([llm_embeddings[i]], dataset_embeddings)[0]
            top_indices = similarities.argsort()[-top_n:][::-1]
            top_matches.extend(dataset.iloc[top_indices].to_dict(orient='records'))
        unique_topics = {match['Human_Readable_Topic']: match for match in top_matches}
        return list(unique_topics.keys())[:5]

    # Filter and display table logic
    def filter_and_display_domains(filtered_domains, selected_topic=None):
        # Remove duplicates based on the 'id' column
        filtered_domains = filtered_domains.drop_duplicates(subset='id')
        if selected_topic:
            filtered_domains = filtered_domains[filtered_domains['Human_Readable_Topic'] == selected_topic]
        return filtered_domains

    # Streamlit UI with Emojis for Better User Experience
    st.title("🔍 Provide Input, Extract Topics, and Discover Related Research Papers")
    st.write("Easily extract the topic of your research, find similar topics, and explore research papers related to those topics!")

    # Input selection with Emojis
    option = st.radio(
        "📄 **Choose Input Method**:",
        ("Upload PDF", "Fill out Form")
    )

    if option == "Upload PDF":
        uploaded_file = st.file_uploader("📄 Upload your PDF file:", type="pdf")
        if uploaded_file is not None:
            with st.spinner("🔄 Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
            st.success("✅ Text extracted successfully!")
            title, abstract = extract_title_and_abstract(text)
            user_title = st.text_area("📝 **Title** (Edit if needed):", value=title, height=100)
            user_abstract = st.text_area("📋 **Abstract** (Edit if needed):", value=abstract, height=300)

    elif option == "Fill out Form":
        user_title = st.text_input("📝 **Enter Title**:")
        user_abstract = st.text_area("📋 **Enter Abstract**:")

    # Topic extraction button
    if st.button("📊 **Extract Potential Topics**"):
        if user_title and user_abstract:
            with st.spinner("🔄 Analyzing content..."):
                analysis = extract_topics(user_title, user_abstract)
                llm_topics = [topic.strip() for topic in analysis.split('- Topics: ')[-1].split(',')]
                st.session_state.llm_topics = llm_topics

            st.success("✅ Topics extracted successfully!")

    
            # Load datasets
            topic_embeddings_dataset = load_topic_with_embeddings()
            llm_related_domains = load_llm_related_domains()

            # Find top 5 topics
            with st.spinner("🔍 Finding similar topics..."):
                top_5_topics = find_most_similar_topics(llm_topics, topic_embeddings_dataset)
                st.session_state.top_5_topics = top_5_topics

    

            # Filter dataset based on the top 5 topics
            filtered_domains = llm_related_domains[
                llm_related_domains['Human_Readable_Topic'].isin(st.session_state.top_5_topics)
            ]
            st.session_state.filtered_domains = filtered_domains
    # # Ensure E
    # Ensure Extracted Topics and Top 5 Closest Topics persist during the session
    if "llm_topics" in st.session_state:
        st.write("### 🗂️ **Extracted Topics**")
        st.text("\n".join(st.session_state.llm_topics))  # Display original output as plain text

        # st.table(pd.DataFrame({"Extracted Topics": st.session_state.llm_topics}))
        # Ensure Extracted Topics and Top 5 Closest Topics persist during the session


    if "top_5_topics" in st.session_state:
        st.write("### 📌 **Top 5 Closest Human-Readable Topics**")
        st.table(pd.DataFrame({"Human_Readable_Topic": st.session_state.top_5_topics}))

    # Table filtering and visualization
    if "filtered_domains" in st.session_state:
        st.write("### 🔍 **Discover All Research Papers for the Topic**")
           # Remove duplicates based on the 'id' column
        st.session_state.filtered_domains = st.session_state.filtered_domains.drop_duplicates(subset='id')
        unique_topics = st.session_state.filtered_domains['Human_Readable_Topic'].unique()
        selected_topic = st.selectbox("📂 **Choose the Topic!**:", options=["All"] + list(unique_topics))

        # Persist filtered domains in session state to avoid disappearing data
        if selected_topic == "All":
            displayed_domains = st.session_state.filtered_domains
            csv_file_name = "filtered_llm_related_domains.csv"
        else:
            displayed_domains = st.session_state.filtered_domains[
                st.session_state.filtered_domains['Human_Readable_Topic'] == selected_topic
            ]
            csv_file_name = f"filtered_domains_{selected_topic.replace(' ', '_').replace('/', '_')}.csv"

        # Save the filtered results in session state
        st.session_state.displayed_domains = displayed_domains

        # Display the table
        st.dataframe(st.session_state.displayed_domains)

        # Download button
        csv_data = st.session_state.displayed_domains.to_csv(index=False)
        st.download_button(
            label="📥 **Download Filtered Table as CSV**",
            data=csv_data,
            file_name=csv_file_name,
            mime="text/csv",
        )
