import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
import wizmap
import os


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
    [ "Topic Tracking","Topic Overview","LLM-related Research Overview", "Entity Tracking"],
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
if section == "Topic Tracking":
        # Load data from the CSV file
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/LLM_related_domainss.csv")
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
            st.markdown("### Select Subdomain(s)")
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
                "Select Topics to Display",
                options=topics,
                default=top_topics,  # Default to the top 5 topics
            )

            # Filter the grouped data based on selected topics
            df_grouped_filtered = df_grouped[df_grouped["Human_Readable_Topic"].isin(selected_topics)]

            # Add a select box for choosing the plot type
            plot_type = st.selectbox(
                "Select Plot Type",
                options=["Monthly Trend", "Cumulative Trend"],
                index=0,  # Default to the first option
            )
            # Add a hover selection for the helper line
            hover = alt.selection_point(
                on="mousemove",  # Trigger the selection on mouse movement
                nearest=True,  # Select the nearest point
                fields=["Month_Start"],  # Use the Month_Start field for alignment
                empty="none"  # Do not display the line if no point is hovered
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
           
                # Add points for interaction
                points = base_chart.mark_point(size=50).encode(opacity=alt.value(0))

                # Add a vertical line for hover interaction
                hover_line = (
                    alt.Chart(df_grouped_filtered)
                    .mark_rule(color="gray", strokeWidth=2)
                    .encode(
                        x="Month_Start:T",
                        tooltip=[
                            alt.Tooltip("Month_Start:T", title="Month Start"),
                            alt.Tooltip("Monthly_Count:Q", title="Monthly Count"),
                            alt.Tooltip("Human_Readable_Topic:N", title="Topic"),
                        ],
                    )
                    .transform_filter(hover)
                )

                # Combine base chart, points, and hover line
                chart = (
                    alt.layer(base_chart, points, hover_line)
                    .add_selection(hover)
                    .properties(
                        height=400,
                        width=800,
                        title="Monthly Topic Trends"
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)

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

                # Add points for interaction
                points = base_chart.mark_point(size=50).encode(opacity=alt.value(0))

                # Add a vertical line for hover interaction
                hover_line = (
                    alt.Chart(df_grouped_filtered)
                    .mark_rule(color="gray", strokeWidth=2)
                    .encode(
                        x="Month_Start:T",
                        tooltip=[
                            alt.Tooltip("Month_Start:T", title="Month Start"),
                            alt.Tooltip("Cumulative_Count:Q", title="Cumulative Count"),
                            alt.Tooltip("Human_Readable_Topic:N", title="Topic"),
                        ],
                    )
                    .transform_filter(hover)
                )

                # Combine base chart, points, and hover line
                chart = (
                    alt.layer(base_chart, points, hover_line)
                    .add_selection(hover)
                    .properties(
                        height=400,
                        width=800,
                        title="Cumulative Total Papers Per Topic"
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)


            # # Step 3: Topic Trends Section
            # st.markdown(f"### Topic Trends for Subdomain(s): {', '.join(selected_subdomains)}")

            # # Parse the 'update_date' column into datetime objects
            # df_subdomain_filtered["update_date"] = pd.to_datetime(df_subdomain_filtered["update_date"], errors="coerce")

            # # Add a column for monthly periods
            # df_subdomain_filtered["Month_Start"] = df_subdomain_filtered["update_date"].dt.to_period("M").apply(lambda x: x.start_time)

            # # Count articles per topic per month
            # df_grouped = df_subdomain_filtered.groupby(["Month_Start", "Human_Readable_Topic"]).size().reset_index(name="Monthly_Count")

            # # Get the total count for each topic to determine the top 10 topics
            # topic_totals = df_grouped.groupby("Human_Readable_Topic")["Monthly_Count"].sum().reset_index()
            # top_topics = topic_totals.sort_values(by="Monthly_Count", ascending=False)["Human_Readable_Topic"].head(5).tolist()

            # # Allow the user to toggle topics with a unique key
            # topics = df_grouped["Human_Readable_Topic"].unique()

            # selected_topics = st.multiselect(
            #     "Select Topics to Display",
            #     options=topics,
            #     default=top_topics,  # Default to the top 5 topics
            #     key=f"topic_multiselect_{'_'.join(selected_categories)}_{'_'.join(selected_subdomains)}"  # Unique key for each selection
            # )

            # # Filter the grouped data based on selected topics
            # df_grouped_filtered = df_grouped[df_grouped["Human_Readable_Topic"].isin(selected_topics)]

            # # Create an interactive line chart
            # chart = (
            #     alt.Chart(df_grouped_filtered)
            #     .mark_line()
            #     .encode(
            #         x=alt.X("Month_Start:T", title="Month Start"),
            #         y=alt.Y("Monthly_Count:Q", title="Monthly Count"),
            #         color=alt.Color(
            #             "Human_Readable_Topic:N",
            #             title="Topics",  # Add a title for the legend
            #             legend=alt.Legend(
            #                 orient="right",  # Position the legend to the right
            #                 titleFontSize=12,  # Adjust the font size of the legend title
            #                 labelFontSize=10,  # Adjust the font size of the legend labels
            #                 labelLimit=200,  # Increase the label limit to avoid truncation
            #                 symbolLimit=50,  # Adjust the number of symbols displayed
            #                 labelOverlap="greedy"  # Avoid overlapping of legend labels
            #             )
            #         )
            #     )
            #     .properties(
            #         height=400,  # Set the height of the chart
            #         width=800,  # Set the width of the chart
            #     )
            #     .configure_legend(
            #         padding=10,  # Add padding around the legend
            #         cornerRadius=5  # Round the corners of the legend box
            #     )
            #     .interactive()  # Enable zoom and pan interactions
            # )

            # # Display the chart
            # st.altair_chart(chart, use_container_width=True)

            # Display detailed insights
            st.markdown("### Monthly Trends for selected Topic Details")
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

            # Provide download option for the CSV
            csv = df_additional_info.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="selected_topics_data.csv",
                mime="text/csv",
            )
    else:
        st.error("The required columns ('Categories', 'Subdomain', 'Human_Readable_Topic') are missing in the dataset.")


    # # Title
    # st.markdown("### Topic Trends Across LLM-Related Subdomains")

    # # Load the data from the real dataset
    # @st.cache_data
    # def load_data():
    #     # Replace with the uploaded file path
    #     df = pd.read_csv("data/LLM_data.csv")
    #     return df

    # # Load the dataset
    # df = load_data()

    # # Parse the 'update_date' column into datetime objects
    # df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")

    # # Ensure the dataset is not empty after parsing
    # if df.empty:
    #     st.error("No valid data found in the dataset. Please check your data.")
    # else:
    #     # Create a navigation bar with tabs for each subdomain
    #     subdomains = df["Categories"].unique()

    #     # Add a radio button to select a subdomain with a unique key
    #     tab = st.radio(
    #         "Select Subdomain",
    #         options=subdomains,
    #         horizontal=True,  # This creates a navigation-like bar
    #         key="unique_subdomain_radio"  # Ensure the key is unique
    #     )

    #     # Filter data based on the selected subdomain
    #     df_filtered = df[df["Categories"] == tab]

    #     # Check if the filtered dataset is empty
    #     if df_filtered.empty:
    #         st.warning(f"No data available for the selected subdomain: {tab}.")
    #     else:
    #         # Add a column for monthly periods
    #         df_filtered["Month_Start"] = df_filtered["update_date"].dt.to_period("M").apply(lambda x: x.start_time)

    #         # Count articles per topic per month
    #         df_grouped = df_filtered.groupby(["Month_Start", "Human_Readable_Topic"]).size().reset_index(name="Monthly_Count")

    #         # Get the total count for each topic to determine the top 10 topics
    #         topic_totals = df_grouped.groupby("Human_Readable_Topic")["Monthly_Count"].sum().reset_index()
    #         top_topics = topic_totals.sort_values(by="Monthly_Count", ascending=False)["Human_Readable_Topic"].head(5).tolist()

    #         # Allow the user to toggle topics with a unique key
    #         topics = df_grouped["Human_Readable_Topic"].unique()

    #         selected_topics = st.multiselect(
    #             "Select Topics to Display",
    #             options=topics,
    #             default=top_topics,  # Default to the top 5 topics
    #             key=f"topic_multiselect_{tab}"  # Ensure the key is unique by appending the tab name
    #         )

    #         # Filter the grouped data based on selected topics
    #         df_grouped_filtered = df_grouped[df_grouped["Human_Readable_Topic"].isin(selected_topics)]

    #         # Display trends for the selected subdomain
    #         st.markdown(f"### Topic Trends for Subdomain: **{tab}**")

    #         # # Create an interactive line chart with maximum width
    #         # chart = (
    #         #     alt.Chart(df_grouped_filtered)
    #         #     .mark_line()
    #         #     .encode(
    #         #         x=alt.X("Month_Start:T", title="Month Start"),
    #         #         y=alt.Y("Monthly_Count:Q", title="Monthly Count"),
    #         #         color=alt.Color("Human_Readable_Topic:N", title="Topics"),  # Different colors for each topic
    #         #     )
    #         #     .properties(
    #         #         height=600, 
    #         #         width=1400  # Use the full width of the container
    #         #     )
    #         # )
    #         # Adjust the Topic Trends chart to ensure the legend is fully visible
    #         chart = (
    #             alt.Chart(df_grouped_filtered)
    #             .mark_line()
    #             .encode(
    #                 x=alt.X("Month_Start:T", title="Month Start"),
    #                 y=alt.Y("Monthly_Count:Q", title="Monthly Count"),
    #                 color=alt.Color(
    #                     "Human_Readable_Topic:N",
    #                     title="Topics",  # Add a title for the legend
    #                     legend=alt.Legend(
    #                         orient="right",  # Position the legend to the right
    #                         titleFontSize=12,  # Adjust the font size of the legend title
    #                         labelFontSize=10,  # Adjust the font size of the legend labels
    #                         labelLimit=200,  # Increase the label limit to avoid truncation
    #                         symbolLimit=50,  # Adjust the number of symbols displayed
    #                         labelOverlap="greedy"  # Avoid overlapping of legend labels
    #                     )
    #                 )
    #             )
    #             .properties(
    #                 height=400,  # Set the height of the chart
    #                 width=800,  # Set the width of the chart
    #             )
    #             .configure_legend(
    #                 padding=10,  # Add padding around the legend
    #                 cornerRadius=5  # Round the corners of the legend box
    #             )
    #             .interactive()  # Enable zoom and pan interactions
    #         )
            

    #         # Display the chart as wide as possible
    #         st.altair_chart(chart, use_container_width=True)

    #         # Display detailed insights
    #         st.markdown("### Monthly Trends for selected Topic Details")
    #         st.write(f"Showing monthly trends for topics under subdomain: **{tab}**.")

    #         # Add a data table for granular details
    #         st.dataframe(df_grouped_filtered, use_container_width=True)


    #         # Filter the original data to include rows with selected topics
    #         df_additional_info = df_filtered[df_filtered["Human_Readable_Topic"].isin(selected_topics)]
    #                     # Dynamically update the markdown with selected topics
    #         if selected_topics:
    #             selected_topics_text = ", ".join(selected_topics)
    #             st.markdown(f"### Export the Paper Details in the CSV file!")
    #             st.write(f"Selected Topics: {selected_topics_text}")
    #         else:
    #             st.markdown("### Export the Paper Details in the CSV file!")
    #             st.write("No topics have been selected.")
    #         st.write(f"Export detailed information about research paper, including links, dates, titles, abstracts, topic label, categories, submitter, and monthly trends for topics under the subdomain: **{tab}**.")

    #         st.dataframe(df_additional_info, use_container_width=True, column_config={"link":st.column_config.LinkColumn()})

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

# # Topic Trends
# # Title
# st.markdown("### Topic Trends per LLM-related Subdomains")

# # Load the data from the real dataset
# @st.cache_data
# def load_data():
#     # Replace with the uploaded file path
#     df = pd.read_csv("data/First_step_clustering_results_categorized_with_llm.csv")
#     return df

# # Load the dataset
# df = load_data()

# # Parse the 'update_date' column into datetime objects
# df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")

# # Ensure the dataset is not empty after parsing
# if df.empty:
#     st.error("No valid data found in the dataset. Please check your data.")
# else:
#     # Create a navigation bar with tabs for each subdomain
#     subdomains = df["Categories"].unique()

#     tab = st.radio(
#         "Select Subdomain",
#         options=subdomains,
#         horizontal=True,  # This creates a navigation-like bar
#         key="subdomain_radio"  # Add a unique key
#     )

#     # Filter data based on the selected subdomain
#     df_filtered = df[df["Categories"] == tab]

#     # Check if the filtered dataset is empty
#     if df_filtered.empty:
#         st.warning(f"No data available for the selected subdomain: {tab}.")
#     else:
#         # Add a column for weekly periods
#         df_filtered["Week_Start"] = df_filtered["update_date"].dt.to_period("W").apply(lambda x: x.start_time)

#         # Count articles per topic per week
#         df_grouped = df_filtered.groupby(["Week_Start", "Human_Readable_Topic"]).size().reset_index(name="Weekly_Count")

#         # Allow the user to toggle topics
#         topics = df_grouped["Human_Readable_Topic"].unique()

#         # Select first 10 topics as default
#         default_topics = topics[:10] if len(topics) > 10 else topics

#         selected_topics = st.multiselect(
#             "Select Topics to Display",
#             options=topics,
#             default=default_topics,  # Default to the first 10 topics or all if fewer than 10
#             key="topic_multiselect"  # Add a unique key for the multiselect widget
#         )

#         # Filter the grouped data based on selected topics
#         df_grouped_filtered = df_grouped[df_grouped["Human_Readable_Topic"].isin(selected_topics)]

#         # Display trends for the selected subdomain
#         st.markdown(f"### Topic Trends for Subdomain: **{tab}**")

#         # Create an interactive line chart with maximum width
#         chart = (
#             alt.Chart(df_grouped_filtered)
#             .mark_line()
#             .encode(
#                 x=alt.X("Week_Start:T", title="Week Start"),
#                 y=alt.Y("Weekly_Count:Q", title="Weekly Count"),
#                 color=alt.Color("Human_Readable_Topic:N", title="Topics"),  # Different colors for each topic
#             )
#             .properties(
#                 height=600, 
#                 width="container"  # Use the full width of the container
#             )
#         )

#         # Display the chart as wide as possible
#         st.altair_chart(chart, use_container_width=True)

#         # Display detailed insights
#         st.markdown("### Subdomain and Topic Details")
#         st.write(f"Showing detailed trends for topics under **{tab}**.")

#         # Add a data table for granular details
#         st.dataframe(df_grouped_filtered, use_container_width=True)


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
    st.write(
        """
        ### Hierchical Topic Knowledge of LLM-Related Research Domain
        """
    )
    # Load data from the CSV file
    @st.cache_data
    def load_data():
        df = pd.read_csv("data/LLM_related_domainss.csv")
        return df

    # Load the dataset
    df = load_data()

    # Ensure the necessary columns exist in the dataset
    if "Categories" in df.columns and "Subdomain" in df.columns and "Human_Readable_Topic" in df.columns:
        
        # List of available categories (update these to match the actual categories in your dataset)
        available_categories = df["Categories"].unique().tolist()
        
        # Create a radio button to select the category
        selected_category = st.radio(
            "Select Category",
            options=available_categories,
            horizontal=True  # Makes the options appear horizontally
        )

        # Filter data based on the selected category
        df_filtered = df[df["Categories"] == selected_category]
        
        # Check if the filtered dataset is empty
        if df_filtered.empty:
            st.warning(f"No data available for the selected category: {selected_category}.")
        else:
            # Calculate the value column as the count of each 'Human_Readable_Topic' within the selected category
            value_df = (
                df_filtered.groupby(["Categories", "Subdomain", "Human_Readable_Topic"])
                .size()
                .reset_index(name="Value")
            )

            # Create the sunburst chart
            fig = px.sunburst(
                value_df,
                path=["Categories", "Subdomain", "Human_Readable_Topic"],
                values="Value",
                width=800,  # Increase width
                height=800  # Increase height
            )


                    # Display the chart
            st.plotly_chart(fig, use_container_width=True)

            # Get the selected topics from the sunburst chart
            selected_topics = value_df["Human_Readable_Topic"].unique().tolist()

            # Filter the data based on the selected topics
            df_grouped_filtered = df_filtered[df_filtered["Human_Readable_Topic"].isin(selected_topics)]

            # Display monthly trends for the selected topics
            st.markdown("### Monthly Trends for Selected Topic Details")
            st.write(f"Showing monthly trends for topics under subdomain/s: {', '.join(df_filtered['Subdomain'].unique())}.")

            # Add a data table for granular details
            st.dataframe(df_grouped_filtered, use_container_width=True)

            # Filter the original data to include rows with selected topics
            df_additional_info = df_filtered[df_filtered["Human_Readable_Topic"].isin(selected_topics)]
            
            # Dynamically update the markdown with selected topics
            if selected_topics:
                selected_topics_text = ", ".join(selected_topics)
                st.markdown(f"### Export the Paper Details in the CSV file!")
            else:
                st.markdown("### Export the Paper Details in the CSV file!")
            
            st.write(f"Export detailed information about research papers, including links, dates, titles, abstracts, topic label, categories, submitter, and monthly trends for topics under the subdomain(s): **{', '.join(df_filtered['Subdomain'].unique())}**.")

            # Display the filtered dataset for download
            st.dataframe(df_additional_info, use_container_width=True,column_config={"id":st.column_config.LinkColumn()})

            # Provide download option for the CSV of the filtered paper details
            csv_paper_details = df_additional_info[['title', 'abstract', 'Human_Readable_Topic', 'Categories', 'Subdomain', 'submitter', 'update_date']].to_csv(index=False)
            st.download_button(
                label="Download Paper Details CSV",
                data=csv_paper_details,
                file_name="research_paper_details.csv",
                mime="text/csv",
            )

    else:
        st.error("The required columns ('Categories', 'Subdomain', 'Human_Readable_Topic') are missing in the dataset.")

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
