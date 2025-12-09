#!/usr/bin/env python3
"""
Streamlit app for interactive visualization of clustered embeddings.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def parse_formatted_sections(formatted_text: str) -> dict[str, str]:
    """Parse formatted output into sections: questions, sparql, and rest."""
    if not formatted_text:
        return {"questions": "", "sparql": "", "rest": ""}

    sections = {"questions": "", "sparql": "", "rest": ""}

    # Extract questions section (from start to "SPARQL query")
    questions_match = re.search(
        r"^(.*?)(?=SPARQL query)", formatted_text, re.DOTALL | re.MULTILINE
    )
    if questions_match:
        sections["questions"] = questions_match.group(1).strip()

    # Extract SPARQL query
    sparql_match = re.search(
        r"SPARQL query[^\n]*:\n(.*?)(?=\n\nUsing entities:|\n\nExecution result:|\Z)",
        formatted_text,
        re.DOTALL,
    )
    if sparql_match:
        sections["sparql"] = sparql_match.group(1).strip()

    # Extract rest (everything after SPARQL)
    rest_match = re.search(
        r"SPARQL query[^\n]*:.*?\n\n((?:Using entities:|Execution result:).*)",
        formatted_text,
        re.DOTALL,
    )
    if rest_match:
        sections["rest"] = rest_match.group(1).strip()

    return sections


@st.cache_data
def load_data(
    embeddings_dir: str,
) -> tuple[list[dict], list[int], list[list[float]], dict]:
    """Load metadata, cluster labels, UMAP coordinates, and statistics."""
    embeddings_path = Path(embeddings_dir)

    # Load metadata
    with open(embeddings_path / "metadata.json") as f:
        metadata = json.load(f)

    # Load cluster labels
    with open(embeddings_path / "cluster_labels.json") as f:
        labels = json.load(f)

    # Load UMAP coordinates
    with open(embeddings_path / "umap_coords.json") as f:
        coords = json.load(f)

    # Load cluster stats
    with open(embeddings_path / "cluster_stats.json") as f:
        stats = json.load(f)

    return metadata, labels, coords, stats


def compute_validity_stats(metadata: list[dict]) -> dict:
    """Compute validity statistics from metadata."""
    stats = {
        "total": len(metadata),
        "valid": 0,
        "invalid_reasons": {},
    }

    for sample in metadata:
        if sample.get("valid", False):
            stats["valid"] += 1
        else:
            reason = sample.get("validity_reason", "unknown")
            stats["invalid_reasons"][reason] = (
                stats["invalid_reasons"].get(reason, 0) + 1
            )

    return stats


def create_dataframe(
    metadata: list[dict], labels: list[int], coords: list[list[float]]
) -> pd.DataFrame:
    """Create DataFrame for visualization."""
    data = []

    for i, (sample, label, coord) in enumerate(
        zip(metadata, labels, coords, strict=True)
    ):
        first_question = sample["questions"][0] if sample["questions"] else ""

        data.append(
            {
                "index": i,
                "cluster": label,
                "x": coord[0],
                "y": coord[1],
                "first_question": first_question,
                "num_questions": sample["num_questions"],
                "sparql": sample.get("sparql", "")[:100] + "..."
                if sample.get("sparql", "")
                else "",
                "valid": sample.get("valid", False),
                "validity_reason": sample.get("validity_reason", "unknown"),
                "file": sample["file"],
            }
        )

    return pd.DataFrame(data)


def main() -> None:
    st.set_page_config(page_title="Wikidata Query Embeddings", layout="wide")

    st.title("🔍 Wikidata Query-SPARQL Embeddings Visualization")

    # Sidebar
    st.sidebar.header("Settings")

    embeddings_dir = st.sidebar.text_input(
        "Embeddings Directory",
        value="data/organic-qwen3-next-80b-a3b/embeddings",
    )

    # Load data
    try:
        metadata, labels, coords, cluster_stats = load_data(embeddings_dir)
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.info(
            "Make sure you have run:\n"
            "1. `python generate_embeddings.py`\n"
            "2. `python build_clusters.py`"
        )
        return

    # Create DataFrame
    df = create_dataframe(metadata, labels, coords)

    # Compute validity statistics
    validity_stats = compute_validity_stats(metadata)

    # Sidebar filters
    st.sidebar.header("Filters")

    # Cluster filter with counts
    cluster_counts = df["cluster"].value_counts().to_dict()
    # Sort by count (descending), then by cluster ID
    cluster_options_sorted = sorted(
        cluster_counts.keys(), key=lambda x: (-cluster_counts[x], x)
    )
    cluster_options_with_counts = [
        f"{cluster} ({cluster_counts[cluster]} samples)"
        for cluster in cluster_options_sorted
    ]
    st.sidebar.info("ℹ️ Cluster -1 indicates invalid samples (not used in clustering)")
    selected_clusters_with_counts = st.sidebar.multiselect(
        "Select Clusters",
        options=cluster_options_with_counts,
        default=[],
    )
    # Extract cluster IDs from selected options
    selected_clusters = [
        int(opt.split(" (")[0]) for opt in selected_clusters_with_counts
    ]

    # Validity filter
    validity_filter = st.sidebar.radio(
        "Sample Validity", options=["All", "Valid Only", "Invalid Only"], index=0
    )

    # Apply filters
    filtered_df = df.copy()

    if selected_clusters:
        filtered_df = filtered_df[filtered_df["cluster"].isin(selected_clusters)]

    if validity_filter == "Valid Only":
        filtered_df = filtered_df[filtered_df["valid"]]
    elif validity_filter == "Invalid Only":
        filtered_df = filtered_df[~filtered_df["valid"]]

    # Display statistics
    st.sidebar.header("Statistics")
    st.sidebar.metric("Total Samples", len(df))
    st.sidebar.metric("Filtered Samples", len(filtered_df))
    st.sidebar.metric("Number of Clusters", df["cluster"].nunique())
    valid_count = validity_stats["valid"]
    total_count = validity_stats["total"]
    valid_pct = valid_count / total_count * 100
    st.sidebar.metric(
        "Valid Samples",
        f"{valid_count} ({valid_pct:.1f}%)",
    )

    # Main content - Visualization in full row
    st.header("Embedding Space Visualization")

    # Convert cluster to string for categorical coloring
    plot_df = filtered_df.copy()
    plot_df["cluster_str"] = plot_df["cluster"].astype(str)

    # Create scatter plot with distinct colors
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster_str",
        hover_data={
            "index": True,
            "first_question": True,
            "num_questions": True,
            "valid": True,
            "cluster": True,
            "cluster_str": False,
            "x": False,
            "y": False,
        },
        title="UMAP Projection of Embeddings (colored by cluster)",
        height=600,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        labels={"cluster_str": "Cluster"},
    )

    fig.update_traces(marker={"size": 5, "opacity": 0.6})
    fig.update_layout(showlegend=True, legend_title_text="Cluster")

    # Capture click events
    selected_points = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun", key="scatter"
    )

    # Store clicked point index in session state
    if (
        selected_points
        and "selection" in selected_points
        and "points" in selected_points["selection"]
        and len(selected_points["selection"]["points"]) > 0
    ):
        clicked_idx = selected_points["selection"]["points"][0]["customdata"][0]
        st.session_state["selected_index"] = int(clicked_idx)

    # Cluster Information in columns below
    st.header("Cluster Information")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster Statistics")

        # Display algorithm and cluster count side by side
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Algorithm", cluster_stats["algorithm"].upper())
        with metric_col2:
            st.metric("Total Clusters", cluster_stats["n_clusters"])

        # Get cluster sizes (excluding -1 for invalid)
        if "cluster_sizes" in cluster_stats:
            sizes = {k: v for k, v in cluster_stats["cluster_sizes"].items() if k != -1}
            if sizes:
                sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

                # Create tables for largest and smallest clusters
                largest_data = []
                smallest_data = []

                for cluster_id, count in sorted_sizes[:5]:
                    largest_data.append({"Cluster": cluster_id, "Samples": count})

                if len(sorted_sizes) > 5:
                    for cluster_id, count in sorted_sizes[-5:]:
                        smallest_data.append({"Cluster": cluster_id, "Samples": count})

                # Display tables side by side
                table_col1, table_col2 = st.columns(2)
                with table_col1:
                    st.write("**Largest Clusters:**")
                    st.dataframe(
                        pd.DataFrame(largest_data),
                        hide_index=True,
                        use_container_width=True,
                    )
                with table_col2:
                    if smallest_data:
                        st.write("**Smallest Clusters:**")
                        st.dataframe(
                            pd.DataFrame(smallest_data),
                            hide_index=True,
                            use_container_width=True,
                        )

    with col2:
        st.subheader("Validity Breakdown")

        # Validity pie chart
        validity_df = pd.DataFrame(
            {
                "Status": ["Valid", "Invalid"],
                "Count": [
                    validity_stats["valid"],
                    validity_stats["total"] - validity_stats["valid"],
                ],
            }
        )

        fig_pie = px.pie(
            validity_df, values="Count", names="Status", title="Sample Validity"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    if validity_stats["invalid_reasons"]:
        st.subheader("Invalid Reasons")
        invalid_df = pd.DataFrame(
            [
                {"Reason": k, "Count": v}
                for k, v in validity_stats["invalid_reasons"].items()
            ]
        ).sort_values("Count", ascending=False)
        st.dataframe(invalid_df, use_container_width=True)

    # Sample details
    st.header("Sample Details")

    # Create dropdown options with "idx: question" format (filtered by validity)
    dropdown_options = {}
    for idx, row in filtered_df.iterrows():
        question = row["first_question"]
        # Truncate long questions
        if len(question) > 80:
            question = question[:77] + "..."
        dropdown_options[f"{idx}: {question}"] = idx

    if not dropdown_options:
        st.warning("No samples match the current filters.")
        return

    # Random sample button
    col_select, col_random = st.columns([4, 1])
    with col_random:
        if st.button("🎲 Random", use_container_width=True):
            random_idx = random.choice(list(dropdown_options.values()))
            st.session_state["selected_index"] = random_idx

    # Get default value from session state or use 0
    default_idx = st.session_state.get("selected_index", 0)
    # If default is not in filtered options, use first available
    if default_idx not in dropdown_options.values():
        default_idx = list(dropdown_options.values())[0]

    default_label = None
    for label, idx in dropdown_options.items():
        if idx == default_idx:
            default_label = label
            break
    if default_label is None:
        default_label = list(dropdown_options.keys())[0]

    # Sample selector dropdown
    with col_select:
        selected_label = st.selectbox(
            "Select Sample",
            options=list(dropdown_options.keys()),
            index=list(dropdown_options.keys()).index(default_label),
            label_visibility="collapsed",
        )
    sample_index = dropdown_options[selected_label]

    # Display selected sample
    sample = metadata[sample_index]
    sample_row = df.iloc[sample_index]

    col1, col2, col3 = st.columns(3)
    col1.metric("Cluster", sample_row["cluster"])
    col2.metric("Valid", "✅" if sample_row["valid"] else "❌")
    col3.metric("Num Questions", sample_row["num_questions"])

    # Show invalid reason if sample is invalid
    if not sample_row["valid"]:
        st.warning(f"Invalid Reason: {sample_row['validity_reason']}")

    # Parse and display formatted output in sections
    formatted = sample.get("formatted", "")
    if formatted:
        sections = parse_formatted_sections(formatted)

        # Questions section
        if sections["questions"]:
            st.markdown(sections["questions"])

        # SPARQL query
        if sections["sparql"]:
            st.subheader("SPARQL Query")
            st.code(sections["sparql"], language="sparql")

        # Rest (entities, execution results, etc.)
        if sections["rest"]:
            st.markdown(sections["rest"])
    else:
        st.info("No formatted output available")

    if sample.get("error"):
        st.error(f"Error: {sample['error']}")


if __name__ == "__main__":
    main()
