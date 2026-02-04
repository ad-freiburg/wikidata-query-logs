import random
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from universal_ml_utils.io import load_json


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
        r"SPARQL query[^\n]*:\n(.*?)(?=\n\nUsing entities:|\n\nUsing properties:|\n\nExecution result:|\Z)",
        formatted_text,
        re.DOTALL,
    )
    if sparql_match:
        sections["sparql"] = sparql_match.group(1).strip()

    # Extract rest (everything after SPARQL)
    rest_match = re.search(
        r"SPARQL query[^\n]*:.*?\n\n((?:Using entities:|Using properties:|Execution result:).*)",
        formatted_text,
        re.DOTALL,
    )
    if rest_match:
        sections["rest"] = rest_match.group(1).strip()

    return sections


@st.cache_data
def load_data(
    dataset_dir: str,
) -> tuple[list[dict], list[int], list[list[float]], dict]:
    """Load samples, cluster labels, UMAP coordinates, and statistics."""
    dataset_path = Path(dataset_dir)

    # Load samples
    samples = load_json(dataset_path / "samples.json")

    # Load cluster labels
    labels = load_json(dataset_path / "clusters" / "cluster_labels.json")

    # Load UMAP coordinates
    coords = load_json(dataset_path / "clusters" / "umap_coords.json")

    # Load cluster stats
    stats = load_json(dataset_path / "clusters" / "cluster_stats.json")

    return samples, labels, coords, stats  # type: ignore


def compute_validity_stats(samples: list[dict]) -> dict:
    """Compute validity statistics from samples."""
    stats = {
        "total": len(samples),
        "valid": 0,
        "invalid_reasons": {},
    }

    for sample in samples:
        if sample.get("valid", False):
            stats["valid"] += 1
        else:
            reason = sample.get("validity_reason", "unknown")
            stats["invalid_reasons"][reason] = (
                stats["invalid_reasons"].get(reason, 0) + 1
            )

    return stats


def create_dataframe(
    samples: list[dict],
    labels: list[int],
    coords: list[list[float]],
) -> pd.DataFrame:
    """Create DataFrame for visualization."""
    data = []

    for i, (sample, label, coord) in enumerate(
        zip(samples, labels, coords, strict=True)
    ):
        first_question = sample["questions"][0] if sample["questions"] else ""

        data.append(
            {
                "index": i,
                "cluster": label,
                "x": coord[0],
                "y": coord[1],
                "first_question": first_question,
                "num_questions": len(sample["questions"]),
                "sparql": sample.get("sparql", "")[:100] + "..."
                if sample.get("sparql", "")
                else "",
                "valid": sample.get("valid", False),
                "validity_reason": sample.get("validity_reason", "unknown"),
                "file": sample["origin"]["file"],
            }
        )

    return pd.DataFrame(data)


def main() -> None:
    st.set_page_config(page_title="Wikidata Query Embeddings", layout="wide")

    st.title("🔍 Wikidata Query-SPARQL Embeddings Visualization")

    # Sidebar
    st.sidebar.header("Settings")

    dataset_dir = st.sidebar.text_input(
        "Data Directory",
        value="data/organic-qwen3-next-80b-a3b-dataset",
    )

    # Load data
    try:
        samples, labels, coords, cluster_stats = load_data(dataset_dir)
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.info(
            "Make sure you have run:\n"
            "1. `python generate_embeddings.py`\n"
            "2. `python build_clusters.py`"
        )
        return

    # Create DataFrame
    df = create_dataframe(samples, labels, coords)

    # Compute validity statistics
    validity_stats = compute_validity_stats(samples)

    # Sidebar filters
    st.sidebar.header("Filters")

    # Cluster filter with counts (top 100 largest clusters only)
    cluster_counts = df["cluster"].value_counts().to_dict()
    # Sort by count (descending), then by cluster ID
    cluster_options_sorted = sorted(
        cluster_counts.keys(), key=lambda x: (-cluster_counts[x], x)
    )
    # Limit to top 100 largest clusters
    cluster_options_sorted = cluster_options_sorted[:100]
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
    filtered_df = df

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

    # Subsample for plotting if too many points
    MAX_PLOT_POINTS = 1000
    if len(plot_df) > MAX_PLOT_POINTS:
        plot_df_sampled = plot_df.sample(n=MAX_PLOT_POINTS, random_state=42)
        st.info(
            f"ℹ️ Only showing {MAX_PLOT_POINTS:,} of {len(plot_df):,} points for performance"
        )
    else:
        plot_df_sampled = plot_df

    # Create scatter plot with distinct colors
    fig = px.scatter(
        plot_df_sampled,
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

                # Create table for largest clusters
                largest_data = []

                for cluster_id, count in sorted_sizes[:5]:
                    largest_data.append({"Cluster": cluster_id, "Samples": count})

                # Display largest clusters table
                st.write("**Largest Clusters:**")
                st.dataframe(
                    pd.DataFrame(largest_data),
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

    if len(filtered_df) == 0:
        st.warning("No samples match the current filters.")
        return

    # Random sample button
    if st.button("🎲 Random Sample", use_container_width=False):
        # Use filtered_df for random selection (all filtered data, not just subsampled)
        random_idx = random.choice(filtered_df.index.tolist())
        st.session_state["selected_index"] = random_idx

    # Get sample index from session state or use first available
    sample_index = st.session_state.get("selected_index", filtered_df.index[0])
    # If selected index not in filtered data, use first available
    if sample_index not in filtered_df.index:
        sample_index = filtered_df.index[0]

    # Display selected sample
    sample = samples[sample_index]
    sample_row = df.iloc[sample_index]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample ID", sample_index)
    col2.metric("Cluster", sample_row["cluster"])
    col3.metric("Valid", "✅" if sample_row["valid"] else "❌")
    col4.metric("Num Questions", sample_row["num_questions"])

    # Show invalid reason if sample is invalid
    if not sample_row["valid"]:
        st.warning(f"Invalid Reason: {sample_row['validity_reason']}")

    # Parse and display formatted output in sections
    formatted = sample.get("formatted", "")
    if formatted:
        sections = parse_formatted_sections(formatted)

        # Raw SPARQL from the original query log
        raw_sparql = sample.get("origin", {}).get("input", "")
        if raw_sparql:
            with st.expander("Raw SPARQL (from query log)"):
                st.code(raw_sparql, language="sparql", wrap_lines=True)

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
