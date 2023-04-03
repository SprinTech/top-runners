from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid
from streamlit_card import card

st.set_page_config(layout="wide")

st.session_state["gender"] = False
st.session_state["country"] = False
st.session_state["event"] = False
st.session_state["name"] = False

# APP HEADER
header1, header2, header3 = st.columns((1, 8, 1))

with header1:
    st.image("https://i.gifer.com/3F3F.gif", width=100)

with header3:
    st.image("https://i.gifer.com/3F3F.gif", width=100)

with header2:
    st.markdown("<h1 style='text-align: center'>Who are the fastest runners in the world ?</h1>",
                unsafe_allow_html=True)
    st.markdown("---")


# PANDAS OPERATIONS
def calculate_age(birthdate: str, running_date: str) -> int:
    birthdate = datetime.strptime(birthdate, "%Y-%m-%d").date()
    running_date = datetime.strptime(running_date, "%Y-%m-%d").date()
    return running_date.year - birthdate.year - (
            (running_date.month, running_date.day) < (birthdate.month, birthdate.day))


def get_hours(time_str):
    h, m, s = time_str.split(":")
    s = s.split(".")[0]
    return int(h) + int(m) / 60 + int(s) / 3600


df = pd.read_csv("./data/data.csv")

df["Age"] = df.apply(lambda x: calculate_age(x["Date of Birth"], x["Date"]), axis=1)
df["Date"] = pd.to_datetime(df["Date"])
df["Place"] = df["Place"].fillna(-1)
df["Place"] = df["Place"].astype(int)
df["Year"] = pd.DatetimeIndex(df['Date']).year

df["Distance"] = df["Event"].str.replace(",", "")
df["Distance"] = df.apply(
    lambda x: "42195" if x["Event"] == "Marathon" else "21100" if x["Event"] == "Half marathon" else x["Distance"],
    axis=1)
df["Distance"] = df["Distance"].str.extract("(\d+)")
df["Distance"] = df["Distance"].astype(int)
df["Distance"] = df["Distance"] / 1000

df["Time"] = df["Time"].apply(lambda x: x.split(".")[0])
df["Time_Hour"] = df["Time"].apply(get_hours)
df["Speed"] = round(df["Distance"] / df["Time_Hour"], 1)

df_filtered = df.copy()


# create filter callbacks
def update_country():
    st.session_state.country = True


def update_gender():
    st.session_state.gender = True


def update_event():
    st.session_state.event = True


def update_name():
    st.session_state.name = True


# FILTER LIST
with st.sidebar:
    st.title("Filters")
    st.markdown("---")

    gender = st.selectbox(label="Gender", options=df_filtered["Gender"].unique(), on_change=update_gender())
    df_filtered = df_filtered.query(f"Gender == '{gender}'")

    if st.session_state["gender"]:
        country = st.selectbox(label="Country", options=df_filtered["Country"].unique(), on_change=update_country())
        df_filtered = df_filtered.query(f"Country == '{country}'")

    if st.session_state["country"]:
        event = st.selectbox(label="Type of race", options=df_filtered["Event"].unique(), on_change=update_event())
        df_filtered = df_filtered.query(f"Event == '{event}'")

    if st.session_state["event"]:
        name = st.selectbox(label="Runner name", options=df_filtered["Name"].unique(), on_change=update_name())
        df_filtered = df_filtered.query(f"Name == '{name}'")

    if st.session_state["name"]:
        date_range = st.date_input(label="Date range", min_value=df_filtered["Date"].min(),
                                   max_value=df_filtered["Date"].max(),
                                   value=(df_filtered["Date"].min(), df_filtered["Date"].max()))
        df_filtered = df_filtered.query(f"Date >= '{date_range[0]}' and Date <= '{date_range[1]}'")

    submitted = st.button("Submit")

if submitted:
    st.title(f"Racing list")
    date_of_birth = df_filtered["Date of Birth"].values[0]

    pronoun = "He" if gender == "Men" else "She"
    possessive_pronoun = "His" if gender == "Mean" else "Her"

    df_filtered = df_filtered.drop(columns=["Rank", "Country", "Name", "Gender", "Date of Birth"])
    gb = GridOptionsBuilder.from_dataframe(df_filtered)
    gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
    gb.configure_side_bar()  # Add a sidebar
    gridOptions = gb.build()

    grid_response = AgGrid(
        df_filtered,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT',
        update_mode='MODEL_CHANGED',
        fit_columns_on_grid_load=True,
        theme='alpine',  # Add theme color to the table
        enable_enterprise_modules=True,
        height=350,
        width='100%',
        reload_data=True
    )

    df_stats = df.copy()
    df_stats = df_stats[df_stats["Event"] == df_filtered["Event"].values[0]]
    df_stats = df_stats.sort_values(by="Time")

    # RESUME
    st.title("Runner informations")

    card1, card2, card3 = st.columns(3)

    with card1:
        country_card = card(
            title="🌎 Country",
            text=country,
        )

    with card2:
        birth_date_card = card(
            title="👶 Birthdate",
            text=date_of_birth,
        )

    with card3:
        position_card = card(
            title="🏃Running category",
            text=df_filtered['Event'].values[0],
        )

    # DATA VISUALIZATION
    st.title("Statistics")

    chart_col_1, chart_col_2 = st.columns(2)

    total_races = df_filtered.shape[0]
    min_age = df_filtered["Age"].min()
    max_age = df_filtered["Age"].max()
    age_diff = (max_age + 1) - min_age

    avg_race_by_year = round(total_races / age_diff, 2)

    # Run by year
    with chart_col_1:
        st.subheader("Age distribution")
        fig = px.histogram(df_filtered, x="Age", color_discrete_sequence=["rgb(102,194,165)"],
                           title=f"<span style='font-size: 20px'>Runs by age</span>", barmode="group")
        fig.update_xaxes(range=[df_filtered["Age"].min(), df_filtered["Age"].max()])
        st.plotly_chart(fig)

        st.markdown(f"""
        <span style='font-size: 20px'><strong>{name}</strong> has run a total of <strong>{total_races} races</strong> in 
        <strong>{age_diff} year</strong>.
        This represents an average of <strong>{avg_race_by_year}</strong> races by year.</div>
        """, unsafe_allow_html=True)

    speed_peak_age = df.iloc[df_filtered["Speed"].idxmax()]["Age"]
    min_max_speed_diff = round(df_filtered["Age"].max() / df_filtered["Age"].min(), 1)

    filtered_sex = df.query(f"Gender == '{gender}'")
    filtered_event = filtered_sex.query(f"Event == '{event}'")

    runner_position = filtered_event.index.get_indexer_for(filtered_event[filtered_event["Name"] == name].index)[0]
    position = round((runner_position + 1) / (df_stats.shape[0] + 1) * 100, 2)

    with chart_col_1:
        st.subheader(f"How fast is {name} ?")
        fig = px.line(round(df_filtered.groupby(["Age"]).mean().reset_index(), 1), x="Age", y="Speed",
                      title="<span style='font-size: 20px'>Average Speed by Year</span>",
                      color_discrete_sequence=px.colors.sequential.algae, text="Speed")
        fig.update_traces(textposition="top center")
        fig.update_layout(font={
            "size": 18
        })
        st.plotly_chart(fig)

        st.markdown(f"""
        <div style='font-size: 20px'>
            <strong>{name}</strong> has reach it's speed peak at age <strong>{speed_peak_age}</strong>.
            There is an increase of <strong>{min_max_speed_diff}%</strong> between the slower time and the fastest time.<br/>
            {pronoun} belongs to the top <strong>{position}% percentile</strong> of fastest runners all time in it's category.
        </div>
            """, unsafe_allow_html=True)

    unique_cities = len(pd.unique(df_filtered["City"]))
    most_frequent_city = df_filtered["City"].mode()[0]

    with chart_col_2:
        st.subheader("What are the favorite cities ?")
        fig = px.histogram(df_filtered, x="City", color_discrete_sequence=["rgb(102,194,165)"],
                           title=f"<span style='font-size: 20px'>Runs by city</span>")
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig)

        st.markdown(f"""
            <span style='font-size: 20px'><strong>{name}</strong> favorite city is <strong>{most_frequent_city}</strong>.
            {pronoun} has run in <strong>{unique_cities} different cities</strong> in total.
            """, unsafe_allow_html=True)

    best_rank = df_filtered.sort_values(by="Place")["Place"].values[0]
    best_rank_times = df_filtered[df_filtered["Place"] == best_rank].shape[0]
    average_rank = df_filtered["Place"].mean()

    with chart_col_2:
        st.subheader(f"How is {name} ranked ?")
        fig = px.pie(df_filtered, values="Place", names="Place",
                     title=f"<span style='font-size: 20px'>Rank repartition</span>",
                     labels="<strong>label></strong>",
                     color_discrete_sequence=px.colors.sequential.algae)
        fig.update_traces(
            textfont_size=16,
            texttemplate="<b>%{percent}</b>",
            marker={
                "line": {
                    "width": 1
                }
            })
        st.plotly_chart(fig)

        st.markdown(f"""
                <div style='font-size: 20px'>
                <strong>{name}</strong> best rank is <strong>{best_rank}</strong>. {pronoun} reach this position 
                <strong>{best_rank_times}
                times</strong> during it's career.
                {pronoun} average rank is <strong>{average_rank}</strong>.
                </div>
                """, unsafe_allow_html=True)
