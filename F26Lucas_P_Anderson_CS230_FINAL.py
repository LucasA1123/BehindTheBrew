"""
CS 230 Final Project | CS230-7
Lucas P. Anderson
Behind the Brew: Manhattan Coffee Shop Health Explorer
Dataset: Manhattan Coffee Shop Violations (NYC restaurant inspections)
AI coding partners: ChatGPT, Claude
Streamlit URL: Add after publishing.

Intended user: a Manhattan college student who wants a coffee shop and
knows nothing about health inspections.

Three questions the app answers (each one is driven by user input):
  1. Which coffee shops in my neighborhood have a low inspection score?
  2. How do ZIP codes (or neighborhoods) compare on average score?
  3. Which violations are most common for a ZIP code and grade I pick?

Other references: NYC Open Data restaurant inspection results, and the
NYC Health Department's explanation of the A / B / C grading system. (As AI Suggested)
"""

from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

# [DEPLOY] IDENTIFY & EXPLAIN
# AI used relative file paths so my application can locate the file regardless of the computer its on
# Using the Path(_file_).parent the folder containing the python file and file name specifies where the csv is located
# This makes it so the file can be stored in a different location
# Example: As long as the Python File and CSV file are in the same folder, the program can locate the dataset without changing file path
BASE_FOLDER = Path(__file__).parent
FILE_NAME = BASE_FOLDER / "manhattan_coffee_shop_violations.csv"

# Each dictionary describes a score range, including its name, maximum score,
# and colors, which is used in the graction of the graphs and map to have consistent
# coloring
SCORE_BANDS = [
    {"name": "Low (0-13 points)", "max_score": 13,
     "rgb": [25, 135, 99, 220], "hex": "#198763"},
    {"name": "Moderate (14-27 points)", "max_score": 27,
     "rgb": [236, 158, 50, 220], "hex": "#EC9E32"},
    {"name": "High (28+ points)", "max_score": float("inf"),
     "rgb": [206, 80, 68, 220], "hex": "#CE5044"},
]

# This dictionary is used to match the neighborhood codes with the respecitive neighborhood names (easier for users
# to identify)
NEIGHBORHOOD_NAMES = {
    "MN01": "Marble Hill / Inwood",
    "MN03": "Central Harlem (North)",
    "MN04": "Hamilton Heights",
    "MN06": "Manhattanville",
    "MN09": "Morningside Heights",
    "MN11": "Central Harlem",
    "MN12": "Upper West Side",
    "MN13": "Chelsea / Flatiron / Union Square",
    "MN14": "Lincoln Square",
    "MN15": "Hell's Kitchen (Clinton)",
    "MN17": "Midtown",
    "MN19": "Turtle Bay / East Midtown",
    "MN20": "Murray Hill / Kips Bay",
    "MN21": "Gramercy",
    "MN22": "East Village",
    "MN23": "West Village",
    "MN24": "SoHo / TriBeCa / Little Italy",
    "MN25": "Battery Park City / Financial District",
    "MN27": "Chinatown",
    "MN28": "Lower East Side",
    "MN31": "Lenox Hill / Roosevelt Island",
    "MN32": "Yorkville",
    "MN33": "East Harlem (South)",
    "MN34": "East Harlem (North)",
    "MN35": "Washington Heights (North)",
    "MN36": "Washington Heights (South)",
    "MN40": "Upper East Side",
    "MN50": "Stuyvesant Town",
    "MN99": "Parks / Other",
}

#[DICT_DRIVEN] AUGMENT - added letter/actually meaning to random letters used associated with values gathered by AI generate code

# This dictionary was not initially created when AI generated the code, however, I added this piece
# to give users a better sense of what each letter of the grade means (espcailly for N and Z)
# Although A-C already correspond to letters, N and Z do not, so it is beneficial that users
# undertsand what these letters mean
# this is used to match the letter grades to the meaning of them (in this case A-C remain their letters
# but N and Z change to what they truly mean)
GRADE_LABELS = {
    "A": "A",
    "B": "B",
    "C": "C",
    "N": "Not yet graded",
    "Z": "Grade pending",
}

# This dictionary connects the button titles to the column name of the dataset.
# For example, choosing "ZIP code" selects the "zipcode" column.
COMPARE_OPTIONS = {
    "ZIP code": "zipcode",
    "Neighborhood": "neighborhood",
}

# Creates the choices for the violation types to be used on the violations page
SEVERITY_OPTIONS = [
    "All violations",
    "Critical only (higher food-safety risk)",
    "Not critical only",
]

ALL_AREAS = "All neighborhoods"
ALL_ZIPS = "All ZIP codes"
ALL_GRADES = "All grades"

#This function is used to shorten the desciription of a value from the data set so that it can be
#practically used in on charts, etc. 
def short_description(text, limit=70):
    """Shorten a long violation description so it fits on a chart label."""
    if pd.isna(text):
        return "No description"
    # Keep only the first sentence or clause (as Claude recomeneded)
    first_part = text.replace(";", ".").split(".")[0].strip()
    if len(first_part) > limit:
        first_part = first_part[:limit].rstrip() + "..."
    return first_part

@st.cache_data #
def load_data():
    """Return (shops, violations).

    shops: one row per coffee shop, using its latest inspection that has a score.
    violations: the violation rows that belong to exactly those inspections.
    """
    data = pd.read_csv(FILE_NAME, dtype={"camis": str, "zipcode": str})
    data.columns = [name.strip().lower() for name in data.columns]

    required = ["camis", "dba", "boro", "cuisine_description", "inspection_date",
                "inspection_type", "score", "grade", "zipcode", "nta", "building",
                "street", "latitude", "longitude", "violation_code",
                "violation_description", "critical_flag"]
    
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Keep Manhattan coffee/tea shops only (the file should already be this) *************
    data = data[
        (data["boro"].fillna("").str.strip().str.lower() == "manhattan") &
        (data["cuisine_description"].fillna("").str.strip().str.lower() == "coffee/tea")
    ].copy()

    # Turn text into dates and numbers. Invalid values become NaN/NaT
    data["inspection_date"] = pd.to_datetime(data["inspection_date"], errors="coerce")

    for name in ["score", "latitude", "longitude"]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["camis", "inspection_date"])

    # Some rows use a placeholder year like 1900 for "never inspected" (AI Suggested)
    data = data[data["inspection_date"].dt.year > 1900].copy()

    # A shop has several rows per inspection (one per violation), and many
    # inspections. Sort newest first, then keep the first row for each shop.
    scored = data.dropna(subset=["score"]).copy()
    scored = scored.sort_values(["inspection_date", "score"], ascending=[False, False])
    shops = scored.drop_duplicates("camis").copy()

    # [STR_CLEAN] IDENTIFY & EXPLAIN
    # This cleans ZIP codes stored as text, and missing/empty ZIP codes become "Unknown",
    # spaces around a value are removed using .strip(), and a trailing ".0" is deleted (as suggested by AI)
    # Example: " 10001.0 " becomes "10001", which will then allow the  ZIP filtering code to work
    shops["zipcode"] = (shops["zipcode"].fillna("Unknown").str.strip()
                        .str.replace(r"\.0$", "", regex=True))
    shops["nta"] = shops["nta"].fillna("").str.strip() 
    shops["neighborhood"] = shops["nta"].apply(
        lambda code: NEIGHBORHOOD_NAMES.get(code, code if code else "Unknown area"))
    shops["grade"] = shops["grade"].fillna("").str.strip()
    shops["grade_label"] = shops["grade"].apply(
        lambda letter: GRADE_LABELS.get(letter, "No grade"))
    shops["dba"] = shops["dba"].fillna("Unnamed coffee shop").str.strip().str.title()
    shops["address"] = (
        shops["building"].fillna("").astype(str).str.strip() + " " +
        shops["street"].fillna("").astype(str).str.strip().str.title()
    ).str.strip().replace("", "Address unavailable")
    shops["date_label"] = shops["inspection_date"].dt.strftime("%b %d, %Y") 

    # Keep only the violation rows from the inspection chosen for each shop. 
    chosen = shops[["camis", "inspection_date", "inspection_type", "score"]]
    violations = scored.merge(
        chosen, on=["camis", "inspection_date", "inspection_type", "score"])
    violations = violations[["camis", "violation_code", "violation_description",
                             "critical_flag"]]
    violations = violations.dropna(subset=["violation_code", "violation_description"])
    violations["short_description"] = violations["violation_description"].apply(
        short_description)
    # Add the shop's ZIP code, neighborhood and grade to every violation row.
    violations = violations.merge(
        shops[["camis", "zipcode", "neighborhood", "grade_label"]], on="camis")
    return shops, violations


def score_band(score):
    """Return the band (dictionary) a score belongs to."""
    for band in SCORE_BANDS:
        if score <= band["max_score"]:
            return band
    return SCORE_BANDS[-1]


def score_color(score):
    """RGB color list used for one map dot."""
    return score_band(score)["rgb"]


# [REUSE_FUNC] IDENTIFY & EXPLAIN
# AI wrote filter_shops() once instead of creating new filters on each page.
# Its default parameters (setting neighborhood=ALL_AREAS, max score to none, etc.) let the user apply only the filters specified
# The map page uses the neghiborhood, the score, and name, wherease the violations page only uses the ZIP.
# Parameters not filled in when the function is called keep their default values, which in turn does not filter the data to the unspecified regions
# Example: filter_shops(shops, zipcodes=["10001"]) returns only coffee shops located within ZIP code 10001
def filter_shops(shops, neighborhood=ALL_AREAS, max_score=None,
                 name_text="", zipcodes=None):
    """Apply whichever filters were given; skipped filters change nothing."""
    result = shops.copy()
    if neighborhood != ALL_AREAS:
        result = result[result["neighborhood"] == neighborhood]
    if max_score is not None:
        result = result[result["score"] <= max_score]
    if name_text.strip():
        result = result[result["dba"].str.contains(
            name_text.strip(), case=False, regex=False, na=False)]
    if zipcodes is not None:
        result = result[result["zipcode"].isin(zipcodes)]
    return result


def legend_html(): 
    """A row of colored dots with labels, built from SCORE_BANDS."""
    pieces = []
    for band in SCORE_BANDS:
        dot = f'<span class="bt-swatch" style="background:{band["hex"]}"></span>'
        pieces.append(dot + band["name"])
    return '<div class="bt-legend">' + "".join(pieces) + "</div>"

#This portion of code uses CSS to change the colors and other asepcts, which the charts and "widgets" are through Streamlit
def website_style():
    st.markdown("""
    <style>
    .stApp {background: #FBF9F5; color: #20352F;}
    .block-container {max-width: 1160px; padding-top: 2rem; padding-bottom: 4rem;}
    h1, h2, h3 {color: #20352F !important; letter-spacing: -0.02em;}
    p {line-height: 1.55;}
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E8E4DE;
        border-radius: 10px; padding: 1rem 1.2rem;
    }
    div[data-testid="stMetricValue"] {color: #1D6B54;}
    .bt-hero {
        background: #203D34; color: #FFFFFF; border-radius: 14px;
        padding: 34px 38px;
    }
    .bt-hero h1 {color: white !important; margin: 0 0 .6rem; font-size: 2.6rem !important;}
    .bt-hero p {color: #E1ECE6; max-width: 640px; margin: 0; font-size: 1.06rem;}
    .bt-title {color: #20352F; font-weight: 700; font-size: 1.6rem; margin: 0 0 .3rem;}
    .bt-sub {color: #56675F; font-size: .97rem; margin: 0 0 .8rem;}
    .bt-legend {font-size: .92rem; margin: .2rem 0 .6rem;}
    .bt-swatch {
        display: inline-block; width: 14px; height: 14px; border-radius: 50%;
        margin: 0 7px 0 0; vertical-align: -2px;
    }
    .bt-legend .bt-swatch:not(:first-child) {margin-left: 22px;}
    .bt-foot {color: #627168; font-size: .83rem; padding-top: 16px;}
    </style>
    """, unsafe_allow_html=True)


def section_intro(title, question):
    st.markdown(f'<div class="bt-title">{title}</div>'
                f'<p class="bt-sub">{question}</p>', unsafe_allow_html=True)

#Creates a basic guide so users can read every chart
def show_basics_guide():
    with st.expander("New to health inspections? Start here"):
        st.markdown(
            "**What is an inspection score?** NYC health inspectors add up "
            "*violation points* during a visit. **More points means more problems, "
            "so a lower score is better.** A score of 0 means nothing was found.")
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.markdown(
            "**What are the letter grades?** 0-13 points earns an **A**, "
            "14-27 a **B**, and 28 or more a **C**. *Grade pending* means the shop "
            "is waiting on a re-inspection; *Not yet graded* means it has not "
            "received a grade yet.")
        st.markdown(
            "**What does Critical mean?** A critical violation is one more likely "
            "to make someone sick (for example, food kept at the wrong temperature "
            "or signs of pests). Not-critical violations are things like missing "
            "signs or equipment that is hard to clean.")
        st.markdown(
            "**Cautions.** Each shop appears once, using its most recent inspection "
            "that has a score. Scores are a snapshot of one day, not a promise about "
            "today. Shops with no scored inspection are not shown, and a few shops "
            "have no map location.")


#Question #1

def show_map(shops):
    section_intro("Question 1: Which coffee shops near me have a good inspection score?",
                  "Pick a neighborhood and the highest score you are comfortable with. "
                  "Hover over a dot to see the shop's name, address and grade.")
    
    left, middle, right = st.columns(3)
    with left:
        counts = shops["neighborhood"].value_counts()
        areas = [ALL_AREAS] + sorted(counts.index.tolist())
        area = st.selectbox(
            "Neighborhood", areas,
            format_func=lambda name: name if name == ALL_AREAS
            else f"{name} ({counts[name]} shops)")
    with middle:
        highest = max(1, int(shops["score"].max()))
        score_limit = st.slider(
            "Highest score to show (lower is better)", 0, highest, min(27, highest),
            help="27 or lower means an A or B grade. Slide left to see only the "
                 "cleanest shops.")
    with right:
        name_text = st.text_input("Search by shop name (optional)",
                                  placeholder="for example: Starbucks")

    matching = filter_shops(shops, neighborhood=area, max_score=score_limit,
                            name_text=name_text)
    # Only keep dots that fall inside the Manhattan area, so bad coordinates
    # do not stretch the map, in the case another data set was used (AI Suggested)
    mapped = matching[
        matching["latitude"].between(40.68, 40.89) &
        matching["longitude"].between(-74.05, -73.90)
    ].copy()

    one, two, three = st.columns(3)
    one.metric("Matching shops", len(matching))
    two.metric("Shown on map", len(mapped))
    three.metric("Average score",
                 f"{matching['score'].mean():.1f}" if not matching.empty else "-")

    if matching.empty:
        st.info("No shops match these choices. Try raising the score limit, "
                "choosing another neighborhood, or clearing the name search.")
        return
    if mapped.empty:
        st.warning("The matching shops do not have usable map locations.")
        return
    if len(mapped) < len(matching):
        st.caption(f"{len(matching) - len(mapped)} matching shop(s) have no usable "
                   "location and are left off the map.")

    st.markdown(legend_html(), unsafe_allow_html=True)

    mapped["color"] = mapped["score"].apply(score_color)
    layer = pdk.Layer(
        "ScatterplotLayer", data=mapped,
        get_position="[longitude, latitude]", get_fill_color="color",
        get_radius=65, radius_min_pixels=5, radius_max_pixels=16,
        pickable=True)
    view = pdk.ViewState(latitude=float(mapped["latitude"].mean()),
                         longitude=float(mapped["longitude"].mean()),
                         zoom=12 if area == ALL_AREAS and not name_text.strip() else 13.5)
    tooltip = {
        "html": "<b>{dba}</b><br/>{address}<br/>{neighborhood}, {zipcode}<br/>"
                "Score: {score} points<br/>Grade: {grade_label}<br/>"
                "Inspected: {date_label}",
        "style": {"backgroundColor": "#203D34", "color": "#FFFFFF"}
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                             map_style="light", tooltip=tooltip),
                    use_container_width=True)

    # This creates a list that users can view the address and other information for the shops
    # 
    with st.expander("See these shops as a list (best score first)"):
        table = matching.sort_values(["score", "dba"])[
            ["dba", "address", "neighborhood", "score", "grade_label", "date_label"]]
        table.columns = ["Shop", "Address", "Neighborhood", "Score", "Grade",
                         "Last scored inspection"]
        st.dataframe(table, hide_index=True, use_container_width=True)

# Question 2

def show_area_comparison(shops):
    section_intro("Question 2: Which areas have the best (lowest) average score?",
                  "Choose ZIP codes or neighborhoods to compare. Shorter bars mean "
                  "fewer violation points on average.")

    compare_by = st.radio("Compare by", list(COMPARE_OPTIONS), horizontal=True)
    column = COMPARE_OPTIONS[compare_by]

    # Leaves out shops whose ZIP code or neighborhood is not known.
    known = shops[~shops[column].isin(["Unknown", "Unknown area"])]
    shop_counts = known[column].value_counts()
    if shop_counts.empty:
        st.warning("No areas are available to compare.")
        return

    left, middle, right = st.columns([2, 1, 1])
    with left:
        # The widget key changes with the radio*** choice so the two lists never mix.
        selected = st.multiselect(
            f"Choose {compare_by.lower()}s to compare",
            sorted(shop_counts.index.tolist()),
            default=shop_counts.head(6).index.tolist(),
            key="compare_" + column)
    with middle:
        min_shops = st.slider(
            "Minimum shops per area", 1, 20, 3,
            help="Averages built from only one or two shops can be misleading, "
                 "so small areas are hidden.")
    with right:
        sort_choice = st.selectbox(
            "Sort bars by",
            ["Best score first", "Worst score first", "Most shops first"])

    if not selected:
        st.info(f"Choose at least one {compare_by.lower()} to see the comparison.")
        return

    matching = known[known[column].isin(selected)]

    # [LOOP_VS_PANDAS] AUGMENT - AI initially just generated the PANDAS calculation for this. However, in order to make it apply to
    # This tag, I manually added the for loop in order to mannually count the number of coffee shops in 
    # each selected area

    # First, I use a loop and dictionary to count rows by area
    # Then, value_counts() answers the same question using Pandas
    # I compare the dictionaries to confirm that both methods agree and contain the same values
    loop_counts = {}
    for area_name in matching[column]:
        if area_name not in loop_counts:
            loop_counts[area_name] = 0
        loop_counts[area_name] += 1
    pandas_counts = matching[column].value_counts().to_dict()
    counts_match = loop_counts == pandas_counts #checks to see if manual and pandas match
    if not counts_match:
        st.error("The manual counts and Pandas counts do not match.")
        return

    summary = matching.groupby(column)["score"].agg(["mean", "count"]).reset_index()
    summary.columns = ["area", "average_score", "shop_count"]
    hidden = len(summary[summary["shop_count"] < min_shops])
    summary = summary[summary["shop_count"] >= min_shops].copy()
    if summary.empty:
        st.info("Every selected area has fewer shops than the minimum. "
                "Lower the minimum or choose areas with more shops.")
        return

    if sort_choice == "Best score first":
        summary = summary.sort_values("average_score") #sorts by scores in ascending order
    elif sort_choice == "Worst score first":
        summary = summary.sort_values("average_score", ascending=False) #sorts by scores in decending order
    else:
        summary = summary.sort_values("shop_count", ascending=False) #sorts by shopcount in decending order
    summary["band"] = summary["average_score"].apply(lambda s: score_band(s)["name"])

    best = summary.loc[summary["average_score"].idxmin()] #minimum value is the "best"
    worst = summary.loc[summary["average_score"].idxmax()] #maximum value is the "worst"
    one, two, three = st.columns(3)
    one.metric("Areas compared", len(summary))
    two.metric("Coffee shops included", int(summary["shop_count"].sum()))
    three.metric("Gap between best and worst",
                 f"{worst['average_score'] - best['average_score']:.1f} pts")

    bars = alt.Chart(summary).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("area:N", sort=summary["area"].tolist(), title=compare_by),
        x=alt.X("average_score:Q", title="Average inspection score (points)",
                scale=alt.Scale(zero=True)),
        color=alt.Color(
            "band:N", title="Score range",
            scale=alt.Scale(domain=[b["name"] for b in SCORE_BANDS],
                            range=[b["hex"] for b in SCORE_BANDS])),
        tooltip=[alt.Tooltip("area:N", title=compare_by),
                 alt.Tooltip("average_score:Q", title="Average score", format=".1f"),
                 alt.Tooltip("shop_count:Q", title="Coffee shops")])
    # Dashed line at 13 points, the top of the A range.
    cutoff = alt.Chart(pd.DataFrame({"x": [13]})).mark_rule(
        strokeDash=[6, 4], color="#20352F").encode(x="x:Q")
    chart = (bars + cutoff).properties(
        title=f"Average inspection score by {compare_by.lower()}",
        height=max(220, 48 * len(summary)))
    st.altair_chart(chart, use_container_width=True)

    st.success(f"Lowest average: **{best['area']}** at {best['average_score']:.1f} points "
               f"({int(best['shop_count'])} shops). Highest average: **{worst['area']}** "
               f"at {worst['average_score']:.1f} points ({int(worst['shop_count'])} shops).")
    st.caption("The dashed line marks 13 points, the top of the A range. Each shop counts "
               "once, using its most recent scored inspection. Hover over a bar to see "
               "how many shops are behind each average.")
    if hidden:
        st.caption(f"{hidden} selected area(s) were hidden because they have fewer than "
                   f"{min_shops} shops.")

# Question 3

def show_violations(shops, violations):
    section_intro("Question 3: What problems do inspectors find most often?",
                  "Pick a ZIP code, a grade and a severity to see which violations "
                  "show up in the most shops.")

    # [LIST_COMP] IDENTIFY & EXPLAIN
    # This list comprehension builds unique ZIP code dropdown choices (finds all possible zip codes in the data set)
    # It loops through unique ZIP codes and stores them in zips, exlcuding those "Unknown" set from a previous function
    # Example: ["10001", "Unknown", "10002"] becomes ["10001", "10002"].
    # Adds "All ZIPs" as the first option followed by unique zips
    zips = [ALL_ZIPS] + sorted([z for z in shops["zipcode"].unique()
                               if z != "Unknown"])
    left, middle, right = st.columns(3)
    with left:
        zipcode = st.selectbox("ZIP code", zips)
    with middle:
        in_zip = (filter_shops(shops) if zipcode == ALL_ZIPS
                  else filter_shops(shops, zipcodes=[zipcode]))
        grades = [ALL_GRADES] + sorted(in_zip["grade_label"].unique().tolist())
        grade = st.selectbox("Inspection grade", grades)
    with right:
        severity = st.radio("Severity", SEVERITY_OPTIONS)

    # filters by shop names and then by zip codes
    chosen_shops = in_zip
    chosen = violations if zipcode == ALL_ZIPS else violations[violations["zipcode"] == zipcode]
    if grade != ALL_GRADES:
        chosen_shops = chosen_shops[chosen_shops["grade_label"] == grade]
        chosen = chosen[chosen["grade_label"] == grade]
    if severity == SEVERITY_OPTIONS[1]:
        chosen = chosen[chosen["critical_flag"] == "Critical"]
    elif severity == SEVERITY_OPTIONS[2]:
        chosen = chosen[chosen["critical_flag"] == "Not Critical"]
    chosen = chosen.drop_duplicates(["camis", "violation_code"])

    one, two = st.columns(2)
    one.metric("Shops in this selection", len(chosen_shops))
    two.metric("Shops with a listed violation", chosen["camis"].nunique())
    if chosen.empty:
        st.info("No violations are recorded for this combination. Try 'All ZIP codes', "
                "'All grades', or a different severity.")
        return

    # Count how many different shops had each violation, then add a percentage.
    counts = chosen.groupby("violation_code").agg(
        shop_count=("camis", "nunique"),
        short_description=("short_description", "first"),
        full_description=("violation_description", "first")).reset_index()
    # This percentage is used in the chart tooltip and summary below.
    counts["percent_of_shops"] = counts["shop_count"] / len(chosen_shops) * 100
    counts["label"] = counts["violation_code"] + ": " + counts["short_description"]
    counts = counts.sort_values("shop_count", ascending=False).head(10)

    chart = alt.Chart(counts).mark_bar(color="#C67B4F", cornerRadiusEnd=4).encode(
        y=alt.Y("label:N", sort=counts["label"].tolist(), title="Violation",
                axis=alt.Axis(labelLimit=420)),
        x=alt.X("shop_count:Q", title="Number of shops with this violation",
                scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("violation_code:N", title="Code"),
                 alt.Tooltip("full_description:N", title="What it means"),
                 alt.Tooltip("shop_count:Q", title="Shops"),
                 alt.Tooltip("percent_of_shops:Q", title="% of shops", format=".0f")]
    ).properties(title="Ten most common violations", height=420)
    st.altair_chart(chart, use_container_width=True)

    top = counts.iloc[0]
    st.success(f"Most common: **{top['short_description']}**, found in "
               f"{int(top['shop_count'])} of {len(chosen_shops)} shops "
               f"({top['percent_of_shops']:.0f}%).")
    st.caption("Each bar counts different shops, not repeat visits. A shop can appear "
               "in several bars. Hover over a bar for the full description.")

    with st.expander("Read the full description of each violation"):
        table = counts[["violation_code", "shop_count", "full_description"]]
        table.columns = ["Code", "Shops", "What the inspector looks for"]
        st.dataframe(table, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="Behind the Brew | Manhattan", page_icon="☕",
                       layout="wide")
    website_style()

    # Header banner
    st.markdown("""
    <div class="bt-hero">
        <h1>Behind the Brew</h1>
        <p>Good coffee starts with a clean counter. Find a Manhattan coffee shop
        with a good inspection record, compare neighborhoods, and see what
        inspectors actually find.</p>
    </div>
    """, unsafe_allow_html=True)

    # [INPUT_CHECK] IDENTIFY & EXPLAIN
    # starts by using the try except code to deal with missing or incorrectly formatted CSV files
    # The try statment attempts to load the coffee shop data, while the except cataches the errors
    # The if statement below  checks if the cleaning process leaves zero shops, and says there are no scored shops
    # If there is an issue and an error occurrs, and displays a message
    # Example: If the CSV file is missing, the program displays an error message ""The coffee shop data could not be loaded: FileNotFundError)"
    try:
        shops, violations = load_data()
    except (FileNotFoundError, ValueError, KeyError, OSError, pd.errors.ParserError) as error:
        st.error("The coffee shop data could not be loaded: " + str(error))
        st.caption("Make sure manhattan_coffee_shop_violations.csv is in the same "
                   "folder as this Python file.")
        st.stop()
    if shops.empty:
        st.warning("There are no scored coffee shop inspections in this dataset.")
        st.stop()

    # [MANUAL_CALC] REVISE/REWRITE
    # AI orginally created a variable for the total number of shops that would increase
    # at each iteration, however, this created two unnessary lines of code
    # (Note, this is another example of LOOPS vs. PANDAS, but this also classifies as a manual calc)
    # Using a loop, I add up all of the scores and divide by the total number of shops (len(shops)) to obtain the average
    # Additionally, I then compare it to the PANDAS average (which is not required for this particular tag)
    total_score = 0
   
    for score in shops["score"]:
        total_score += score
    manual_average = total_score / len(shops) ##REVISED##
    pandas_average = shops["score"].mean()
    if abs(manual_average - pandas_average) > 0.000001:
        st.error("The manual average and Pandas average do not match.")
        st.stop()

    st.write("")
    show_basics_guide()
    st.markdown("**What would you like to explore?**")
    choice = st.radio(
        "Choose a page",
        ["1. Find a coffee shop", "2. Compare areas", "3. Explore violations"],
        horizontal=True, label_visibility="collapsed")
    st.divider()

    if choice.startswith("1"):
        show_map(shops)
    elif choice.startswith("2"):
        show_area_comparison(shops)
    else:
        show_violations(shops, violations)

    st.markdown('<div class="bt-foot">Behind the Brew | CS 230 Final Project | '
                'Source: NYC Department of Health restaurant inspection data. '
                'Scores describe past inspections, not current conditions.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
