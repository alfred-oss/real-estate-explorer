from pathlib import Path
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Real Estate Explorer", page_icon="🏠", layout="wide")
st.title("🏠 Real Estate Explorer")

# The data file lies next to app.py (a relative path works on any computer and on Streamlit Community Cloud).
# It is a gzip-compressed CSV (about 18 MB instead of 120 MB); pandas reads it directly.
DATA_PATH = Path(__file__).parent / "dsc.csv.gz"

# Identifier columns are numbers, but they are codes, not measurements: never used in charts or statistics.
ID_COLS = ["brokered_by", "zip_code"]
# Columns that have one value per state (the same for every listing in the state).
STATE_COLS = ["crime_rate", "median_income", "flood_risk"]

# User-friendly names for tables and charts.
FRIENDLY = {
    "status": "Status",
    "price": "Price ($)",
    "bed": "Bedrooms",
    "bath": "Bathrooms",
    "acre_lot": "Lot size (acres)",
    "city": "City",
    "state": "State",
    "zip_code": "ZIP code",
    "house_size": "House size (sqft)",
    "prev_sold_date": "Previous sale date",
    "price_per_sqft": "Price per sqft ($)",
    "crime_rate": "Crime rate (per 100k residents)",
    "median_income": "Median household income ($)",
    "flood_risk": "Flood risk (0–100)",
}


def pretty(col):
    return FRIENDLY.get(col, col)


# NOTE: cache_resource returns the same DataFrame object on every rerun (cache_data would copy ~1M rows each time,
# which made the page freeze). Because of that, never modify `df` in place.
@st.cache_resource(show_spinner="Loading data...")
def load_csv(path):
    # brokered_by (an anonymous broker id) is useless for the reader, so it is not loaded at all
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                    usecols=lambda c: c != "brokered_by")
    # text columns as categories: the table takes ~100 MB instead of ~360 MB (Streamlit Cloud has limited memory)
    for c in ["status", "city", "state", "prev_sold_date"]:
        if c in d.columns:
            d[c] = d[c].astype("category")
    return d


if not DATA_PATH.exists():
    st.error(f"File not found: {DATA_PATH}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Options")
    page = st.radio("📂 Navigation", ["Home", "EDA", "Maps", "Insights"], index=0)

df = load_csv(DATA_PATH)


def numeric_cols(d):
    return [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]


def analysis_cols(d):
    """Numeric columns that are real measurements (identifier codes excluded)."""
    return [c for c in numeric_cols(d) if c not in ID_COLS]


def strength(r):
    a = abs(r)
    if a < 0.1:
        return "very weak"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    return "strong"


# ----- Cached helpers (the underscore in `_d` tells Streamlit not to hash the huge DataFrame) -----------------
@st.cache_resource
def get_sample(_d, n=20_000):
    return _d.sample(min(n, len(_d)), random_state=42)


@st.cache_data
def hist_counts(_d, col, bins, clip):
    s = _d[col].dropna()
    if clip:
        lo, hi = s.quantile([0.01, 0.99])
        s = s[(s >= lo) & (s <= hi)]
    return np.histogram(s, bins=bins)


@st.cache_data
def corr_matrix(_d, cols):
    return _d[list(cols)].corr()


@st.cache_data
def state_table(_d):
    t = (_d.groupby("state", as_index=False)
           .agg(price_per_sqft=("price_per_sqft", "median"),
                acre_lot=("acre_lot", "median"),
                house_size=("house_size", "median"),
                bed=("bed", "median"),
                bath=("bath", "median"),
                crime_rate=("crime_rate", "median"),
                median_income=("median_income", "median"),
                flood_risk=("flood_risk", "median"),
                listings=("price", "size")))
    t["code"] = t["state"].map(STATE_ABBREV)
    return t


@st.cache_data
def bed_bath_pivot(_d, min_count=30):
    """Median price per sqft for every bedrooms x bathrooms pair; pairs with few listings are hidden."""
    g = _d.groupby(["bed", "bath"])["price_per_sqft"].agg(["median", "size"]).reset_index()
    g = g[g["size"] >= min_count]
    pv = g.pivot(index="bed", columns="bath", values="median").sort_index().sort_index(axis=1)
    pv.index, pv.columns = pv.index.astype(int), pv.columns.astype(int)
    return pv


@st.cache_data
def describe_table(_d):
    return _d[analysis_cols(_d)].describe().T.rename(index=FRIENDLY)


STATE_ABBREV = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT',
    'Delaware':'DE','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI',
    'Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH',
    'Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
    'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV',
    'Wisconsin':'WI','Wyoming':'WY','District of Columbia':'DC'
}


def page_home():
    st.subheader("🧩 Problem statement")
    st.write(
        "**Goal:** In which US state is it more advantageous to buy a home, taking into account "
        "the price per square foot, the income level, the crime rate and the flood risk?"
    )

    st.divider()

    # Why this matters
    st.subheader("Why this matters")
    st.markdown("""
- Housing is typically the largest purchase most people make.
- Prices vary widely with fundamentals (bed/bath, size, lot) and with **context** (crime, income, flood risk).
- Using **price_per_sqft** lets us compare locations more fairly than raw prices, because it removes the effect of house size.
- State-level context (crime rate, median income, flood risk) helps set expectations, but it is a coarse indicator: use it as a first filter, then check the specific city and neighborhood.
    """)

    st.divider()

    # Data Source & Limitations
    st.subheader("Data Source & Limitations")
    st.markdown(f"""
**Source:** USA Real Estate Dataset (property listings) from `kaggle.com`, file dated July 2025 (about 2.2 million raw rows).

**Coverage:** all 50 states and the District of Columbia, {len(df):,} listings after cleaning.

**What was missing:**
- `price_per_sqft`, `crime_rate`, `median_income`, `flood_risk` — had to search and add manually for more context and better decision making.
- `crime_rate` is the total crime rate in 2025 (violent + property crimes per 100,000 residents, FBI data via [USAFacts](https://usafacts.org/articles/which-states-have-the-least-and-most-crime/)). Florida and North Dakota have incomplete 2025 reporting.
- `median_income` is the 2025 median household income by state (U.S. Census Bureau, American Community Survey 2025 1-year estimates, brief ACSBR-028).
- `flood_risk` is a flood-frequency score (0–100), and it is not only floods: it counts FEMA major disaster declarations in 2000–2025 linked to floods **and hurricanes / tropical and coastal storms**, scaled so that the highest state = 100 (source: OpenFEMA).
- `crime_rate`, `median_income` and `flood_risk` are one value per state, the same for every listing in the state.
- **Cleaning:** rows with missing values and duplicate listings (the same house as `for_sale` and later `sold`; the `sold` record is kept) were removed. Outliers were then removed in two ways: hard limits (price outside $50k–$10M, house size outside 400–20,000 sqft, lots over 10 acres, more than 10 beds or baths, inconsistent bed/bath/size combinations such as 1 bed with 10 baths) and an IQR rule on `price_per_sqft` (log scale, within each state). Studios/1-bedroom homes and luxury homes are kept, and states are compared by the **median** rather than the mean.
- The raw file also contained anonymous codes for the street and the broker (not useful for analysis), missing values and extreme errors (for example 444 bedrooms or a $1 price).
    """)

    st.divider()

    st.markdown(
        "**Rows and columns.** The size of the cleaned dataset: each *row* is one property listing "
        "(a home for sale or recently sold) and each *column* is one fact about it (price, bedrooms, state, ...)."
    )
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")

    st.markdown(
        "**Head.** The first 6 listings, to show what a single record looks like. "
        "The last three columns (crime, income, flood risk) describe the state, so they repeat for every listing in that state."
    )
    st.dataframe(df.head(6).rename(columns=FRIENDLY), width="stretch")

    st.markdown(
        "**Describe.** Summary statistics for every numeric column: **count** (listings with a value), "
        "**mean** (average), **std** (typical spread around the mean), **min** and **max** (smallest and largest value), "
        "and **25% / 50% / 75%** (quartiles; 50% is the median, the middle listing). "
        "It helps to see typical values and how extreme the extremes are."
    )
    if analysis_cols(df):
        st.dataframe(describe_table(df), width="stretch")
    else:
        st.info("No numeric columns found.")


def page_eda():
    st.subheader("🔎 EDA")
    num = analysis_cols(df)
    if not num:
        st.info("No numeric columns.")
        return

    # ---- Histogram (binned with numpy: fast, and only a few bars are sent to the browser) ----
    st.markdown("**Histogram**")
    st.caption("How many listings fall into each value range of the chosen column.")
    h1, h2, h3 = st.columns([2, 2, 2])
    col = h1.selectbox("Column", num, format_func=pretty)
    bins = h2.slider("Bins", 10, 100, 40)
    clip = h3.checkbox("Hide extreme values (keep the middle 98%)", value=True,
                       help="Cuts the lowest 1% and highest 1% so that a few extreme listings do not squash the chart.")
    counts, edges = hist_counts(df, col, bins, clip)
    fig_h = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges)))
    fig_h.update_layout(xaxis_title=pretty(col), yaxis_title="Number of listings", bargap=0.02)
    st.plotly_chart(fig_h, width="stretch")
    if col in STATE_COLS:
        st.caption("This value is the same for every listing in a state, so the bars show how many listings "
                   "have each value, not how many states.")

    # ---- Scatter plot (random sample: plotting a million points freezes the browser) ----
    st.markdown("**Scatter plot**")
    sample = get_sample(df)
    st.caption(f"Each dot is one listing. A random sample of {len(sample):,} listings (out of {len(df):,}) is shown to keep the page fast.")
    c1, c2 = st.columns(2)
    xcol = c1.selectbox("X", num, index=num.index("house_size") if "house_size" in num else 0,
                        key="x", format_func=pretty)
    ycol = c2.selectbox("Y", num, index=num.index("price") if "price" in num else 0,
                        key="y", format_func=pretty)
    st.plotly_chart(
        px.scatter(sample, x=xcol, y=ycol, opacity=0.4, labels={xcol: pretty(xcol), ycol: pretty(ycol)}),
        width="stretch",
    )

    # ---- Correlation heatmap + state-level scatter ----
    st.markdown("**Correlation Heatmap and income vs. price per sqft**")
    left, right = st.columns(2)

    with left:
        st.caption("Correlation (Pearson): from −1 to 1. Close to 0 means no linear link; "
                   "close to 1 means the two grow together. Identifier columns (broker id, ZIP code) are excluded.")
        corr = corr_matrix(df, tuple(num))
        fig_c = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                          color_continuous_scale="RdBu_r", labels=dict(color="Correlation"))
        fig_c.update_xaxes(side="bottom", tickangle=-45)
        fig_c.update_layout(height=520)
        st.plotly_chart(fig_c, width="stretch")

    with right:
        t = state_table(df).dropna(subset=["median_income", "price_per_sqft"])
        # Spearman = Pearson correlation of the ranks (pandas' own method="spearman" needs scipy, which is not installed on Streamlit Cloud)
        rho = t["median_income"].rank().corr(t["price_per_sqft"].rank())
        st.caption(f"One dot = one state (median price per sqft vs. the state's median household income). "
                   f"Spearman ρ = {rho:.2f}: richer states tend to be more expensive per sqft.")
        slope, intercept = np.polyfit(t["median_income"], t["price_per_sqft"], 1)
        xs = np.array([t["median_income"].min(), t["median_income"].max()])
        fig_s = px.scatter(t, x="median_income", y="price_per_sqft", text="code", hover_name="state",
                           labels={"median_income": pretty("median_income"), "price_per_sqft": "Median price per sqft ($)"})
        fig_s.update_traces(textposition="top center", marker=dict(size=9))
        fig_s.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="trend",
                                   line=dict(color="red")))
        fig_s.update_layout(showlegend=False, height=520)
        st.plotly_chart(fig_s, width="stretch")

    # ---- Bedrooms x bathrooms heatmap ----
    st.markdown("**Price per sqft by bedrooms and bathrooms**")
    st.caption("Each cell is the median price per sqft of listings with that number of bedrooms (vertical) and "
               "bathrooms (horizontal). Cells with fewer than 30 listings are left empty. "
               "More bathrooms at the same bedroom count usually means a higher price per sqft.")
    pv = bed_bath_pivot(df)
    fig_bb = px.imshow(pv, origin="lower", text_auto=".0f", aspect="auto", color_continuous_scale="Viridis",
                       labels=dict(x="Bathrooms", y="Bedrooms", color="Median $/sqft"))
    fig_bb.update_xaxes(dtick=1)
    fig_bb.update_yaxes(dtick=1)
    st.plotly_chart(fig_bb, width="stretch")


METRIC_LABELS = {
    "price_per_sqft": "Median price per sqft ($)",
    "acre_lot": "Median lot size (acres)",
    "crime_rate": "Crime rate (per 100k residents)",
    "median_income": "Median household income ($)",
    "flood_risk": "Flood risk (0–100)",
}
MAP_SCALES = {
    "price_per_sqft": "Viridis",
    "acre_lot": "Greens",
    "crime_rate": "Reds",
    "median_income": "Blues",
    "flood_risk": "Plasma",
}


def page_maps():
    st.subheader("🗺️ Maps")
    # Which metric to map?
    metric = st.selectbox(
        "Choose a metric to map:",
        ["price_per_sqft", "crime_rate", "median_income", "flood_risk", "acre_lot"],
        index=0,
        format_func=lambda m: METRIC_LABELS[m],
    )

    # Aggregate by state
    g = state_table(df)[["state", "code", metric]].dropna(subset=["code"])

    # Build the choropleth
    fig = px.choropleth(
        g,
        locations="code",
        locationmode="USA-states",
        color=metric,
        scope="usa",
        color_continuous_scale=MAP_SCALES[metric],
        labels={metric: METRIC_LABELS[metric]},
        hover_name="state",
        title=f"{METRIC_LABELS[metric]} by State",
    )
    st.plotly_chart(fig, width="stretch")

    # ----- Regional Comparisons -------------------------------------------------
    st.markdown("---")
    st.subheader("🏁 Regional Comparisons")

    # pick states to compare
    all_states = sorted(df["state"].dropna().unique().tolist())
    compare_states = st.multiselect(
        "Pick 2–5 states to compare (e.g., New York, Georgia):",
        options=all_states,
        default=["New York", "Georgia"],
        max_selections=5,
    )

    # pick metrics to compare
    metrics = list(METRIC_LABELS)
    pick_metrics = st.multiselect(
        "Pick metrics to compare:",
        options=metrics,
        default=["price_per_sqft", "median_income", "crime_rate"],
        format_func=lambda m: METRIC_LABELS[m],
    )

    norm = st.checkbox(
        "Normalize selected metrics to [0, 1] for fair visual comparison",
        value=False,
        help="The metrics are measured in very different units (dollars, acres, points), so raw bars are not comparable. "
             "Normalization puts every metric on the same 0–1 scale.",
    )
    if norm:
        st.caption(
            "**What normalization does:** the metrics have very different scales (income is in tens of thousands of dollars, "
            "lot size is a fraction of an acre), so on one chart the biggest numbers would hide everything else. "
            "Normalization rescales each metric across the selected states: **0 = the lowest of the chosen states, "
            "1 = the highest**. The bars then show *who is higher or lower*, not the real values "
            "(the real values are in the table below the chart)."
        )

    if len(compare_states) >= 2 and len(pick_metrics) >= 1:
        agg = state_table(df)

        # restrict to chosen states
        view = agg[agg["state"].isin(compare_states)][["state"] + pick_metrics].copy()
        actual = view.copy()  # real values, shown in the table

        # optional normalization across chosen states
        if norm:
            # min-max per metric (only over chosen states so bars remain intuitive for the comparison)
            for m in pick_metrics:
                mmin, mmax = view[m].min(), view[m].max()
                if mmax > mmin:
                    view[m] = (view[m] - mmin) / (mmax - mmin)
                else:
                    # avoid divide-by-zero if all values equal
                    view[m] = 0.5

        # long format for grouped bars
        long = view.melt(id_vars="state", value_vars=pick_metrics,
                         var_name="metric", value_name="value")
        long["metric"] = long["metric"].map(METRIC_LABELS)

        fig_cmp = px.bar(
            long,
            x="state", y="value", color="metric",
            barmode="group",
            labels={"value": "value (0 = lowest, 1 = highest)" if norm else "value", "metric": "Metric"},
            title="State comparison" + (" (normalized)" if norm else ""),
        )
        st.plotly_chart(fig_cmp, width="stretch")

        # small numeric table under the chart (always the real values)
        with st.expander("Show numeric table (real values)"):
            st.dataframe(actual.set_index("state").rename(columns=METRIC_LABELS))
    else:
        st.info("Pick at least two states and one metric to see the comparison.")


@st.cache_data
def factor_table(_d):
    """Correlation of price_per_sqft with each factor: across listings (Pearson) and across states (Spearman)."""
    factors = ["price", "house_size", "bed", "bath", "acre_lot", "median_income", "crime_rate", "flood_risk"]
    listing_r = _d[["price_per_sqft"] + factors].corr()["price_per_sqft"]
    st_tab = (_d.groupby("state")[["price_per_sqft"] + factors].median())
    state_rho = st_tab.rank().corr()["price_per_sqft"]  # Spearman = Pearson on ranks (no scipy needed)
    rows = []
    for f in factors:
        rows.append({"factor": f, "r": listing_r[f], "rho": state_rho[f]})
    return pd.DataFrame(rows).set_index("factor")


@st.cache_data
def state_variance_share(_d):
    """Share of the variation of log(price_per_sqft) that is explained by the state alone."""
    lg = np.log10(_d["price_per_sqft"])
    within = ((lg - lg.groupby(_d["state"]).transform("mean")) ** 2).sum()
    return 1 - within / ((lg - lg.mean()) ** 2).sum()


def page_insights():
    st.subheader("💡 Insights & Next steps")

    # --- Prep: state-level aggregates
    cols_needed = ["state", "price_per_sqft", "crime_rate", "median_income", "flood_risk"]
    d = df[cols_needed].dropna()

    state_agg = (
        d.groupby("state", as_index=False)
         .agg(price_per_sqft=("price_per_sqft", "median"),
              crime_rate=("crime_rate", "median"),
              median_income=("median_income", "median"),
              flood_risk=("flood_risk", "median"),
              listings=("price_per_sqft", "size"))
         .sort_values("price_per_sqft")
         .reset_index(drop=True)
    )

    # --- 1) Where is it cheaper / more expensive?
    st.markdown("### 1) Where is it cheaper / more expensive?")
    k = st.slider("How many states to show (each list)", 3, 10, 5, key="insights_topk")

    cheap = state_agg.nsmallest(k, "price_per_sqft")[["state", "price_per_sqft", "listings"]]
    expensive = state_agg.nlargest(k, "price_per_sqft")[["state", "price_per_sqft", "listings"]]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cheapest (by median price_per_sqft)**")
        st.dataframe(cheap.set_index("state"), width="stretch")
    with c2:
        st.markdown("**Most expensive (by median price_per_sqft)**")
        st.dataframe(expensive.set_index("state"), width="stretch")

    # optional bars
    fig_bar = px.bar(state_agg.sort_values("price_per_sqft").head(15),
                     x="state", y="price_per_sqft",
                     title="Lowest 15 price_per_sqft (median by state)")
    st.plotly_chart(fig_bar, width="stretch")

    # --- 2) What factors influence price_per_sqft?
    st.subheader("2) What influences price_per_sqft?")
    ft = factor_table(df)
    eta = state_variance_share(df)

    shown = pd.DataFrame({
        "Across listings: Pearson r": ft["r"].round(2),
        "Strength (listings)": ft["r"].map(strength),
        "Across states: Spearman ρ": ft["rho"].round(2),
        "Strength (states)": ft["rho"].map(strength),
    })
    shown.index = [pretty(f) + (" *" if f == "price" else "") for f in ft.index]
    st.dataframe(shown, width="stretch")
    st.caption(
        "**How to read:** the numbers are correlations with `price_per_sqft` (from −1 to 1, 0 = no link). "
        "*Across listings* uses all listings; state-wide values (income, crime, flood) repeat for each listing, "
        "so big states count more. *Across states* compares the 51 state medians, which matches the question "
        "“which state?”. Strength: below 0.1 very weak, 0.1–0.3 weak, 0.3–0.5 moderate, 0.5 and above strong. "
        "A correlation shows an association, not a cause. "
        "\\* Price is partly built in: price_per_sqft = price ÷ house size."
    )

    r = ft["r"]
    rho = ft["rho"]
    st.markdown(f"""
**Key signals:**

- **Median income** is the strongest contextual factor: **{strength(r['median_income'])}** across listings (r = {r['median_income']:+.2f}) and **{strength(rho['median_income'])}** across states (ρ = {rho['median_income']:+.2f}). Wealthier states are more expensive per sqft.
- **House size, beds and baths** are **{strength(max(abs(r['house_size']), abs(r['bed']), abs(r['bath'])))}** drivers of price per sqft across listings (|r| ≤ {max(abs(r['house_size']), abs(r['bed']), abs(r['bath'])):.2f}): a bigger house costs more in total, but not more per sqft.
- **Crime rate** is **{strength(r['crime_rate'])}** across listings (r = {r['crime_rate']:+.2f}) and **{strength(rho['crime_rate'])}** across states (ρ = {rho['crime_rate']:+.2f}). Higher recorded crime does not make a state cheaper per sqft: expensive places such as the District of Columbia have high crime rates too.
- **Flood risk** gives mixed signals: r = {r['flood_risk']:+.2f} across listings but ρ = {rho['flood_risk']:+.2f} across states. Across states, more flood-prone states tend to be cheaper per sqft; across listings, large expensive markets (for example California and Florida) pull the number up. So there is no consistent link.
- **Lot size** (acres): r = {r['acre_lot']:+.2f} across listings and ρ = {rho['acre_lot']:+.2f} across states, so states with bigger typical lots tend to be a bit cheaper per sqft.

**Reading:** `price_per_sqft` is driven mainly by **location**: the state alone explains about **{eta:.0%}** of the variation in price per sqft, while house size, beds and baths explain almost nothing once the price is divided by square feet. Among the context factors we have, income matters most; crime and flood risk are weak or mixed signals.
    """)

    # --- 3) Trade-offs (low price, high risk)
    st.markdown("### 3) Trade-offs")
    st.caption("States that are **cheap** (bottom quartile of price_per_sqft) but **riskier** (top quartile of crime_rate or flood_risk).")
    q_price_low = state_agg["price_per_sqft"].quantile(0.25)
    q_crime_high = state_agg["crime_rate"].quantile(0.75)
    q_flood_high = state_agg["flood_risk"].quantile(0.75)

    tradeoffs = state_agg.query(
        "(price_per_sqft <= @q_price_low) and (crime_rate >= @q_crime_high or flood_risk >= @q_flood_high)"
    )[["state", "price_per_sqft", "crime_rate", "flood_risk"]].sort_values("price_per_sqft")

    if tradeoffs.empty:
        st.info("No strong trade-off states under the chosen quartile rule.")
    else:
        st.dataframe(tradeoffs.set_index("state"), width="stretch")

    # --- 4) “Best to buy” vs “Expensive” (balanced shortlists)
    st.markdown("### 4) Best states to buy vs expensive states")
    st.caption("Balanced score (min–max normalized, 0–1): low price_per_sqft (weight 40%), higher median_income (25%), "
               "lower crime_rate (20%), lower flood_risk (15%). The weights are a judgment call, not a statistical result.")

    # normalize 0–1 (higher is better after sign flips)
    norm = state_agg.copy()
    def mm(x):
        x = x.astype(float)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    norm["p_inv"] = 1 - mm(norm["price_per_sqft"])
    norm["income"] = mm(norm["median_income"])
    norm["crime_inv"] = 1 - mm(norm["crime_rate"])
    norm["flood_inv"] = 1 - mm(norm["flood_risk"])
    norm["score"] = (
        0.40*norm["p_inv"] +
        0.25*norm["income"] +
        0.20*norm["crime_inv"] +
        0.15*norm["flood_inv"]
    )
    shortlist = norm.sort_values("score", ascending=False).head(k)[
        ["state", "score", "price_per_sqft", "median_income", "crime_rate", "flood_risk"]
    ]
    pricey = state_agg.nlargest(k, "price_per_sqft")[["state", "price_per_sqft", "median_income", "crime_rate", "flood_risk"]]

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Best states to buy (multi-factor score)**")
        st.dataframe(shortlist.set_index("state"), width="stretch")
    with cc2:
        st.markdown("**Expensive states (by price_per_sqft)**")
        st.dataframe(pricey.set_index("state"), width="stretch")

    # --- 5) Future Enhancements
    st.markdown("### 🔮 Future Enhancements")
    st.markdown(
        """
- **Add richer features**
  - School quality (ratings, distance to top schools)
  - Job market (unemployment, new jobs, industry mix)
  - Transportation & accessibility (transit score, commute time)
  - Amenities (parks, hospitals, grocery density), environmental risks beyond floods
  - Property specifics (year built, HOA fees, parking/garage, renovation status)
        """
    )

    # --- Summary bullets (fast talking points)
    st.markdown("### Quick summary:")
    st.markdown(
        f"""
- **Cheapest now:** {', '.join(cheap['state'].head(3))} &nbsp;&nbsp;|&nbsp;&nbsp; **Most expensive:** {', '.join(expensive['state'].head(3))}
- **Drivers:** location explains about {eta:.0%} of the variation in price per sqft; **median_income** is the strongest context factor (higher income → higher price per sqft). Crime and flood risk are weak or mixed signals.
- **Trade-offs:** {len(tradeoffs)} low-price states have high crime or flood risk, so screen area-level safety before deciding.
- **Best shortlist (multi-factor):** {', '.join(shortlist['state'].head(5))}
        """
    )

if page == "Home":
    page_home()
elif page == "EDA":
    page_eda()
elif page == "Maps":
    page_maps()
else:
    page_insights()
