# streamlit_wine_analysis.py
# Automatyczne wczytywanie winequality-red.csv i wine_food_pairings.csv

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Wine Analysis & Food Pairing", layout="wide")

# --- AUTOMATYCZNE WYSZUKIWANIE PLIKÓW ---
def find_file(filename):
    for root, dirs, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Szukamy lokalnych plików
quality_path = find_file("winequality-red.csv")
food_path = find_file("wine_food_pairings.csv")

if quality_path is None:
    st.error("Nie znaleziono pliku: winequality-red.csv — umieść plik w katalogu aplikacji.")
    st.stop()

if food_path is None:
    st.error("Nie znaleziono pliku: wine_food_pairings.csv — umieść plik w katalogu aplikacji.")
    st.stop()

# Wczytanie CSV
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

wine = load_csv(quality_path)
pairings = load_csv(food_path)

# Sidebar mode
mode = st.sidebar.radio("Tryb:", ["Eksploracja danych", "Model ML (predykcja jakości)", "Food–Wine Pairings"])
show_raw = st.sidebar.checkbox("Pokaż surowe dane")

# Heatmap function
def corr_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    return px.imshow(corr, text_auto=True, aspect="auto", title="Macierz korelacji")

# EXPLORATION MODE
if mode == "Eksploracja danych":
    st.title("Eksploracja winequality-red.csv")
    st.subheader("Podstawowe statystyki")
    st.write(wine.describe())

    st.subheader("Histogram jakości")
    st.plotly_chart(px.histogram(wine, x='quality'), use_container_width=True)

    st.subheader("Interaktywny scatterplot")
    numeric_cols = wine.select_dtypes(include=np.number).columns.tolist()
    x_axis = st.selectbox("X", numeric_cols, index=numeric_cols.index('alcohol'))
    y_axis = st.selectbox("Y", numeric_cols, index=numeric_cols.index('quality'))
    color_by = st.selectbox("Kolor", [None] + numeric_cols)

    st.plotly_chart(px.scatter(wine, x=x_axis, y=y_axis, color=color_by), use_container_width=True)

    st.subheader("Korelacje")
    st.plotly_chart(corr_heatmap(wine, numeric_cols), use_container_width=True)

    if show_raw:
        st.subheader("Surowe dane")
        st.dataframe(wine)

# MODEL MODE
elif mode == "Model ML (predykcja jakości)":
    st.title("Model ML — RandomForestRegressor")

    features = wine.select_dtypes(include=np.number).columns.tolist()
    features.remove('quality')

    X = wine[features]
    y = wine['quality']

    test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)
    seed = st.sidebar.number_input("Random state", 0, 9999, 42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=seed)

    do_scale = st.checkbox("Standaryzuj cechy")
    if do_scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 150, 10)
    max_depth = st.sidebar.slider("max_depth", 2, 30, 8)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("RMSE", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.metric("R²", r2_score(y_test, y_pred))

    st.subheader("Ważność cech")
    feat_imp = pd.Series(model.feature_importances_, index=features).sort_values()
    st.plotly_chart(px.bar(feat_imp, orientation='h'), use_container_width=True)

# FOOD–WINE PAIRINGS MODE
elif mode == "Food–Wine Pairings":
    st.title("Food–Wine Pairings")

    st.subheader("Statystyki")
    st.write(pairings.describe(include='all'))

    wine_types = ['Wszystkie'] + sorted(pairings['wine_type'].unique().tolist())
    selected = st.selectbox("Wybierz wino", wine_types)

    df_filtered = pairings.copy()
    if selected != 'Wszystkie':
        df_filtered = df_filtered[df_filtered['wine_type'] == selected]

    st.dataframe(df_filtered)

    if show_raw:
        st.subheader("Surowe dane")
        st.dataframe(pairings)

st.caption("Aplikacja działa — pliki CSV są automatycznie wyszukiwane w katalogu aplikacji.")
