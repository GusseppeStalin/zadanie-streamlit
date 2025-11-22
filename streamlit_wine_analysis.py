# streamlit_wine_analysis.py
# Zaawansowana aplikacja Streamlit z możliwością uploadu plików CSV

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wine Analysis & Food Pairing", layout="wide")

st.sidebar.header("Wczytaj pliki CSV")
wine_file = st.sidebar.file_uploader("Wgraj winequality-red.csv", type=["csv"])
food_file = st.sidebar.file_uploader("Wgraj wine_food_pairings.csv", type=["csv"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

if wine_file is not None:
    wine = load_csv(wine_file)
else:
    st.error("Wgraj plik winequality-red.csv aby kontynuować.")
    st.stop()

if food_file is not None:
    pairings = load_csv(food_file)
else:
    st.error("Wgraj plik wine_food_pairings.csv aby kontynuować.")
    st.stop()

mode = st.sidebar.radio("Tryb:", ["Eksploracja danych", "Model ML (predykcja jakości)", "Food–Wine Pairings"])
show_raw = st.sidebar.checkbox("Pokaż surowe dane (odpowiedni zestaw)")

def corr_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Macierz korelacji (pearson)")
    return fig

if mode == "Eksploracja danych":
    st.title("Eksploracja danych: winequality-red.csv")
    st.subheader("Podstawowe statystyki")
    st.write(wine.describe())

    st.subheader("Rozkład jakości")
    fig_hist = px.histogram(wine, x='quality')
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Interaktywne wykresy")
    numeric_cols = wine.select_dtypes(include=np.number).columns.tolist()

    x_axis = st.selectbox("X", numeric_cols, index=numeric_cols.index('alcohol'))
    y_axis = st.selectbox("Y", numeric_cols, index=numeric_cols.index('quality'))
    color_by = st.selectbox("Kolor", [None] + numeric_cols)

    scatter = px.scatter(wine, x=x_axis, y=y_axis, color=color_by)
    st.plotly_chart(scatter, use_container_width=True)

    st.subheader("Heatmap korelacji")
    st.plotly_chart(corr_heatmap(wine, numeric_cols), use_container_width=True)

    if show_raw:
        st.subheader("Surowe dane")
        st.dataframe(wine)

elif mode == "Model ML (predykcja jakości)":
    st.title("Model ML: RandomForestRegressor")

    features = wine.select_dtypes(include=np.number).columns.tolist()
    features.remove('quality')
    X = wine[features]
    y = wine['quality']

    test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)
    random_state = st.sidebar.number_input("Random state", 0, 9999, 42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)

    do_scale = st.checkbox("Standaryzuj cechy")
    if do_scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 150, 10)
    max_depth = st.sidebar.slider("max_depth", 2, 30, 8)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("RMSE", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.metric("R²", r2_score(y_test, y_pred))

    st.subheader("Feature Importances")
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    st.plotly_chart(px.bar(importances, orientation='h'), use_container_width=True)

elif mode == "Food–Wine Pairings":
    st.title("Analiza food–wine pairings")

    st.subheader("Statystyki")
    st.write(pairings.describe(include='all'))

    st.subheader("Filtrowanie")
    wine_types = ['Wszystkie'] + sorted(pairings['wine_type'].unique().tolist())
    selected_wine = st.selectbox("Wybierz wino", wine_types)

    df_filtered = pairings.copy()
    if selected_wine != 'Wszystkie':
        df_filtered = df_filtered[df_filtered['wine_type'] == selected_wine]

    st.dataframe(df_filtered)

    if show_raw:
        st.subheader("Surowe dane")
        st.dataframe(pairings)

st.caption("Aplikacja Streamlit — wersja z uploadem plików CSV.")
