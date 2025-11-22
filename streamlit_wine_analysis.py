# streamlit_wine_analysis.py
# Zaawansowana aplikacja Streamlit do analizy zbiorów:
# - /mnt/data/winequality-red.csv
# - /mnt/data/wine_food_pairings.csv

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

# --- Paths to the uploaded files (local) ---
WINE_QUALITY_PATH = '/mnt/data/winequality-red.csv'
WINE_FOOD_PATH = '/mnt/data/wine_food_pairings.csv'

@st.cache_data
def load_wine_quality(path=WINE_QUALITY_PATH):
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_wine_food(path=WINE_FOOD_PATH):
    df = pd.read_csv(path)
    return df

wine = load_wine_quality()
pairings = load_wine_food()

# -----------------
# Sidebar controls
# -----------------
st.sidebar.header("Ustawienia aplikacji")
mode = st.sidebar.radio("Tryb:", ["Eksploracja danych", "Model ML (predykcja jakości)", "Food–Wine Pairings"])

# Common: show raw data checkbox
show_raw = st.sidebar.checkbox("Pokaż surowe dane (odpowiedni zestaw)")

# -----------------
# Helper functions
# -----------------

def corr_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Macierz korelacji (pearson)")
    return fig

# -----------------
# Mode: Exploration
# -----------------
if mode == "Eksploracja danych":
    st.title("Eksploracja: winequality-red.csv")

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Podstawowe statystyki")
        st.write(wine.describe())

    with col2:
        st.subheader("Rozkład jakości")
        fig_hist = px.histogram(wine, x='quality', nbins=6, title='Histogram quality', labels={'quality':'Quality'})
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Interaktywne wykresy")
    numeric_cols = wine.select_dtypes(include=np.number).columns.tolist()

    x_axis = st.selectbox("Wybierz zmienną X", numeric_cols, index=numeric_cols.index('alcohol') if 'alcohol' in numeric_cols else 0)
    y_axis = st.selectbox("Wybierz zmienną Y", numeric_cols, index=numeric_cols.index('quality') if 'quality' in numeric_cols else 1)
    color_by = st.selectbox("Koloruj według", [None] + numeric_cols, index=0)

    scatter = px.scatter(wine, x=x_axis, y=y_axis, color=color_by if color_by else None, trendline='ols', title=f"{y_axis} vs {x_axis}")
    st.plotly_chart(scatter, use_container_width=True)

    st.subheader("Macierz korelacji")
    hm = corr_heatmap(wine, numeric_cols)
    st.plotly_chart(hm, use_container_width=True)

    st.markdown("---")
    if show_raw:
        st.subheader("Surowe dane - winequality-red.csv")
        st.dataframe(wine)

# -----------------
# Mode: ML
# -----------------
elif mode == "Model ML (predykcja jakości)":
    st.title("Model ML: Predykcja jakości wina czerwonego")

    st.markdown("#### 1) Przygotowanie danych")
    # Features and target
    features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
                'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    X = wine[features].copy()
    y = wine['quality'].copy()

    st.write("Liczba próbek:", X.shape[0])
    test_size = st.sidebar.slider("Wielkość zbioru testowego (%)", 10, 50, 25)
    random_state = st.sidebar.number_input("Random state (seed)", value=42, step=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size/100), random_state=int(random_state))

    st.markdown("#### 2) Standaryzacja (opcjonalna)")
    do_scale = st.checkbox("Standaryzuj cechy (StandardScaler)", value=False)
    if do_scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

    st.markdown("#### 3) Trenowanie modelu")
    n_estimators = st.sidebar.slider("Liczba drzew (RandomForest)", 50, 500, 150, step=10)
    max_depth = st.sidebar.slider("Max depth", 2, 30, 8)

    rf = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=int(random_state))
    with st.spinner("Trenuję model..."):
        rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.metric("RMSE", f"{rmse:.3f}")
    st.metric("R^2", f"{r2:.3f}")

    st.markdown("#### 4) Predykcje: próbka vs prawda")
    comp = pd.DataFrame({'true': y_test, 'pred': y_pred})
    comp = comp.reset_index(drop=True)
    st.dataframe(comp.head(20))

    st.markdown("#### 5) Ważność cech")
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    fig_imp = px.bar(importances, x=importances.values, y=importances.index, orientation='h', title='Feature importances (RF)')
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### 6) Permutation importance (bardziej niezależna miara)")
    perm = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=int(random_state))
    perm_importances = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
    fig_perm = px.bar(perm_importances, x=perm_importances.values, y=perm_importances.index, orientation='h', title='Permutation importances')
    st.plotly_chart(fig_perm, use_container_width=True)

    st.markdown("#### 7) Interaktywna predykcja pojedynczego wina")
    st.write("Przesuń wartości cech, aby otrzymać predykcję jakości")
    input_cols = st.columns(3)
    sample = {}
    for i, feat in enumerate(features):
        col = input_cols[i % 3]
        min_v = float(np.floor(wine[feat].min()))
        max_v = float(np.ceil(wine[feat].max()))
        median = float(wine[feat].median())
        sample[feat] = col.slider(feat, min_value=min_v, max_value=max_v, value=median)

    sample_df = pd.DataFrame([sample])
    if do_scale:
        sample_df = pd.DataFrame(scaler.transform(sample_df), columns=features)

    pred_single = rf.predict(sample_df)[0]
    st.subheader(f"Predykowana jakość: {pred_single:.2f}")

    if show_raw:
        st.subheader("Surowe dane - winequality-red.csv")
        st.dataframe(wine)

# -----------------
# Mode: Food–Wine Pairings
# -----------------
elif mode == "Food–Wine Pairings":
    st.title("Analiza food–wine pairings")

    st.markdown("Dane pochodzą z pliku: `/mnt/data/wine_food_pairings.csv`")

    # Quick overview
    st.subheader("Podstawowe statystyki i widoki")
    c1, c2 = st.columns(2)
    with c1:
        st.write(pairings.describe(include='all'))
    with c2:
        st.write(pairings['pairing_quality'].value_counts().sort_index())

    st.markdown("---")
    st.subheader("Filtrowanie i wyszukiwanie pairings")
    wine_types = ['Wszystkie'] + sorted(pairings['wine_type'].dropna().unique().tolist())
    selected_wine = st.selectbox("Wybierz typ wina", wine_types)
    sel_cat = st.multiselect("Wybierz kategorię jedzenia", options=sorted(pairings['food_category'].dropna().unique().tolist()), default=None)
    pq_min, pq_max = st.slider("Zakres pairing_quality", 1, 5, (1,5))

    df_filtered = pairings.copy()
    if selected_wine != 'Wszystkie':
        df_filtered = df_filtered[df_filtered['wine_type'] == selected_wine]
    if sel_cat:
        df_filtered = df_filtered[df_filtered['food_category'].isin(sel_cat)]
    df_filtered = df_filtered[(df_filtered['pairing_quality'] >= pq_min) & (df_filtered['pairing_quality'] <= pq_max)]

    st.subheader(f"Wyniki: {len(df_filtered)} wierszy")
    st.dataframe(df_filtered.reset_index(drop=True))

    st.markdown("---")
    st.subheader("Top food pairings według średniej jakości")
    top_by_food = pairings.groupby('food_item').agg(mean_quality=('pairing_quality','mean'), count=('pairing_quality','size')).reset_index()
    top_by_food = top_by_food[top_by_food['count']>=1].sort_values(['mean_quality','count'], ascending=[False, False]).head(20)
    fig_top = px.bar(top_by_food, x='mean_quality', y='food_item', orientation='h', labels={'mean_quality':'Średnia ocena','food_item':'Jedzenie'}, title='Top jedzeń według średniej oceny pairing_quality')
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")
    st.subheader("Mapa kategorii — ile pairów w każdej kategorii")
    cat_counts = pairings['food_category'].value_counts().reset_index()
    cat_counts.columns = ['food_category', 'count']
    fig_cat = px.bar(cat_counts, x='food_category', y='count', title='Ilość pairów wg kategorii jedzenia')
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")
    st.subheader("Szybka rekomendacja dla wybranego wina")
    sel_wine_for_rec = st.selectbox("Wybierz wino do rekomendacji", sorted(pairings['wine_type'].dropna().unique().tolist()))
    rec_n = st.slider("Ile propozycji?", 1, 10, 5)
    recs = pairings[pairings['wine_type'] == sel_wine_for_rec].sort_values('pairing_quality', ascending=False).head(rec_n)
    st.table(recs[['wine_type','food_item','food_category','cuisine','pairing_quality','description']].reset_index(drop=True))

    if show_raw:
        st.subheader("Surowe dane - wine_food_pairings.csv")
        st.dataframe(pairings)

# -----------------
# Footer / Notes
# -----------------
st.sidebar.markdown("---")
st.sidebar.write("Aplikacja: analiza danych + prosty model RandomForest do regresji jakości.")
st.sidebar.write("Pliki użyte lokalnie:")
st.sidebar.code(WINE_QUALITY_PATH)
st.sidebar.code(WINE_FOOD_PATH)

st.markdown("---")
st.caption("Uwaga: model ML jest przykładowy — do zastosowań produkcyjnych zaleca się bardziej gruntowną walidację i optymalizację.")
