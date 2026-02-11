import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.cluster.hierarchy as sch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="News Topic Discovery", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.markdown(
    "This system uses Hierarchical Clustering to automatically group similar news articles based on textual similarity."
)
st.info("👉 Discover hidden themes without defining categories upfront.")


# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Dataset Upload")
uploaded_file = st.sidebar.file_uploader("all-data.csv", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file, encoding="latin1", header=None)
df.columns = [f"col_{i}" for i in range(df.shape[1])]
df["text"] = df.iloc[:, -1].astype(str)


# ---------------- TF-IDF CONTROLS ----------------
st.sidebar.header("📝 TF-IDF Settings")

max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000)
remove_stopwords = st.sidebar.checkbox("Remove English Stopwords", True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)


# ---------------- CLUSTERING CONTROLS ----------------
st.sidebar.header("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

dendrogram_sample = st.sidebar.slider(
    "Articles for Dendrogram",
    20, 200, 100
)

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2, 10, 5
)


# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(
    stop_words="english" if remove_stopwords else None,
    max_features=max_features,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(df["text"])


# ---------------- DENDROGRAM ----------------
if st.button("🟦 Generate Dendrogram"):
    st.subheader("🌳 Dendrogram")

    sample = X[:dendrogram_sample].toarray()

    fig, ax = plt.subplots(figsize=(12, 5))
    sch.dendrogram(
        sch.linkage(sample, method=linkage_method)
    )
    plt.xlabel("Article Index")
    plt.ylabel("Distance")

    st.pyplot(fig)
    st.success("Inspect large vertical gaps to decide cluster count.")


# ---------------- APPLY CLUSTERING ----------------
if st.button("🟩 Apply Clustering"):

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )

    clusters = model.fit_predict(X.toarray())
    df["Cluster"] = clusters

    # -------- PCA VISUALIZATION --------
    st.subheader("📊 Cluster Visualization (PCA)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    plot_df = pd.DataFrame({
        "PCA1": X_pca[:, 0],
        "PCA2": X_pca[:, 1],
        "Cluster": clusters,
        "Snippet": df["text"].str[:150]
    })

    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color=plot_df["Cluster"].astype(str),
        hover_data=["Snippet"]
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------- SILHOUETTE SCORE --------
    st.subheader("📊 Silhouette Score")
    score = silhouette_score(X, clusters)
    st.metric("Score", round(score, 3))

    if score > 0.5:
        st.success("Clusters are well separated.")
    elif score > 0.2:
        st.warning("Clusters have moderate separation.")
    else:
        st.error("Clusters overlap significantly.")

    # -------- CLUSTER SUMMARY --------
    st.subheader("📋 Cluster Summary")

    terms = vectorizer.get_feature_names_out()
    summary = []

    for i in range(n_clusters):
        indices = np.where(clusters == i)
        cluster_mean = X[indices].mean(axis=0)
        top_words = np.argsort(cluster_mean.A1)[-10:]
        keywords = ", ".join([terms[j] for j in top_words])

        summary.append([i, len(indices[0]), keywords])

    summary_df = pd.DataFrame(
        summary,
        columns=["Cluster ID", "Article Count", "Top Keywords"]
    )

    st.dataframe(summary_df)

    st.info(
        "Articles grouped in the same cluster share similar vocabulary and themes. "
        "These clusters can be used for automatic tagging, recommendations, and content organization."
    )
