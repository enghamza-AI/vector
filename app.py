
from __future__ import annotations

from pathlib import Path

import joblib
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Vector — Lead Intelligence Platform",
    page_icon=":material/target:",
    layout="wide",
)


DEMO_RESULTS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "demo"
    / "vector_results.pkl"
)


@st.cache_resource(show_spinner=False)
def load_demo_results() -> dict:
    return joblib.load(DEMO_RESULTS_PATH)


with st.sidebar:
    st.markdown("### Vector")
    st.caption("SynthSec lead intelligence demo")

    st.divider()

    st.caption(
        "All data on this page is synthetic — generated to demonstrate "
        "the methodology, never real client data."
    )


st.title("Vector — which leads deserve attention?")

st.markdown(
    "A live demo of SynthSec's lead intelligence methodology, built "
    "entirely on **synthetic** engagement and firmographic data."
)


with st.expander("What is this, and what problem does it solve?", expanded=True):
    st.markdown(
        """
Sales teams often work leads in the order they arrived, not the order
they matter. A highly-engaged lead with no real budget wastes a rep's
time; a quiet enterprise lead with real intent gets ignored until it's
too late.

Vector answers two questions per lead:

1. **Tier** (unsupervised) — hot, warm, cold, or unqualified, based on
   engagement behavior AND a firmographic reality check (budget,
   company size) that engagement alone can't see.
2. **Lead score** (supervised, 0-100) — a conversion probability, with
   the single most explanatory signal (a demo request, a budget/
   engagement combo, etc.), so a rep knows not just who to call but why.
        """
    )


try:
    output = load_demo_results()
except FileNotFoundError:
    st.error(
        "Vector demo data is missing. Make sure "
        "`data/demo/vector_results.pkl` exists in the repository."
    )
    st.stop()
except Exception as exc:
    st.error(f"Unable to load Vector demo data: {exc}")
    st.stop()


result = output["result"]


st.divider()
st.subheader("Dashboard")


m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Leads analyzed",
    f"{output['n_leads']:,}",
)

m2.metric(
    "Segments found",
    result["segment_name"].nunique(),
)

m3.metric(
    "Model AUC",
    output["model_auc"],
)

m4.metric(
    "Silhouette score",
    output["silhouette_avg"],
)


segment_counts = result["segment_name"].value_counts().reset_index()
segment_counts.columns = ["segment", "count"]

fig_segments = px.bar(
    segment_counts,
    x="segment",
    y="count",
    color="segment",
    title="Leads per segment",
)

fig_segments.update_layout(
    showlegend=False,
    height=360,
)

st.plotly_chart(
    fig_segments,
    use_container_width=True,
)

del segment_counts, fig_segments


st.subheader("Insights")


fig_scatter = px.scatter(
    result,
    x="total_events",
    y="lead_score",
    color="segment_name",
    hover_data=[
        "lead_id",
        "budget_tier",
        "company_size",
    ],
    title="Total engagement vs. lead score, by segment",
)

fig_scatter.update_layout(height=420)

st.plotly_chart(
    fig_scatter,
    use_container_width=True,
)

del fig_scatter


urgency_counts = (
    result["urgency"]
    .value_counts()
    .reindex(["high", "medium", "low"])
    .fillna(0)
    .reset_index()
)

urgency_counts.columns = ["urgency", "count"]


fig_urgency = px.bar(
    urgency_counts,
    x="urgency",
    y="count",
    color="urgency",
    color_discrete_map={
        "high": "#D85A30",
        "medium": "#EF9F27",
        "low": "#5DCAA5",
    },
    title="Leads by recommended-action urgency",
)

fig_urgency.update_layout(
    showlegend=False,
    height=320,
)

st.plotly_chart(
    fig_urgency,
    use_container_width=True,
)

del urgency_counts, fig_urgency


st.subheader("Recommended actions")
st.caption("Sorted by urgency — start at the top.")


urgency_order = {
    "high": 0,
    "medium": 1,
    "low": 2,
}

display_df = result.copy()

display_df["_sort"] = display_df["urgency"].map(urgency_order)

display_df = display_df.sort_values(
    ["_sort", "lead_score"],
    ascending=[True, False],
)


st.dataframe(
    display_df[
        [
            "lead_id",
            "segment_name",
            "urgency",
            "lead_score",
            "primary_signal",
            "recommended_action",
        ]
    ].head(50),
    use_container_width=True,
    hide_index=True,
)

del display_df


st.divider()

st.caption(
    "Vector is a demo built on synthetic data. A client engagement runs "
    "the identical pipeline against real CRM/marketing-automation "
    "exports — see about_the_project.md for how that swap works."
)
