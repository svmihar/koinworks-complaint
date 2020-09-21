from cluster_topic import get_top_k_word
import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
from train_sklearn import rf_model


@st.cache(suppress_st_warning=True)
def load_data():
    df = pd.read_csv("./asset/eda.csv")
    df.dropna(inplace=True, subset=["cleaned"])
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(inplace=True, subset=["date"])
    return df


# @st.cache
def load_model():
    return rf_model()


@st.cache
def wordcloud(df: pd.DataFrame):
    wc = WordCloud(background_color="white", width=800, height=400)
    wc.generate(" ".join(df["cleaned"].values))
    img = wc.to_image()
    return img


def chart_trending_tweets(df):
    df_ = df.drop_duplicates(subset=["username"])
    df_ = df.groupby(df["date"].dt.date)["username"].count().reset_index()
    df_.columns = ["date", "tweet_count"]
    fig = px.line(df_, x="date", y="tweet_count")
    return fig


def chart_trending_complaints(df):
    df_ = df[df["complaint"] == 1].drop_duplicates(subset=["username"])
    df_ = df_.groupby(df_["date"].dt.date)["username"].count().reset_index()
    df_.columns = ["date", "complaint_count"]
    fig = px.line(df_, x="date", y="complaint_count")
    return fig


def top_words_per_topic(df):
    clustering_columns = [
        "kmeans_flair",
        "dbscan_flair",
        "dbscan_umap",
        "kmeans_umap",
        "dbscan_pca",
        "kmeans_pca",
    ]
    result = {}
    for topic_cluster in clustering_columns:
        result[topic_cluster] = []
        for topic_id in df[topic_cluster].unique():
            result[topic_cluster].append(
                {
                    "description": "this is a non apt description of this model and embedding",  # TODO: explaining shit about how this clustering and "dimensionality reduction works"
                    "top_words": ", ".join(
                        get_top_k_word(df[df[topic_cluster] == topic_id].cleaned.values)
                    ),
                    "topic_id": int(topic_id),
                }
            )
    for cluster_method, item in result.items():
        st.markdown(f"## {cluster_method}\n{item[0]['description']}")
        for x in item:
            st.markdown(f'{x["topic_id"]}: {x["top_words"]}')
        st.markdown("---")


def eda(df):
    st.header("Analysis")
    st.subheader("wordcloud")
    # TODO:introduction text her
    chart = wordcloud(df)
    st.image(chart)
    st.header("tweets about koinworks")
    st.write(chart_trending_tweets(df))
    st.header("complaint tweets trending")
    st.write(chart_trending_complaints(df))


def load_sklearn():
    return load_model()


def classification(df):
    model = load_model()
    # TODO: explaining what's the model and how it predicts
    st.header("Koinworks complaint classifier")
    st.subheader("predict, whether the tweet is a koinwork complaint or not")
    st.write("you can learn the how it is made here: ")
    query_tweet = st.text_input("insert a tweet here")
    if st.button("predict"):
        hasil = model.predict(query_tweet)
        st.write(hasil[0])


def generation(df):
    st.write("TBA")


MENU = {
    "EDA": eda,
    "Complaint Topics": top_words_per_topic,
    "Classification": classification,
    "Text Generation": generation,
}


def main():
    df = load_data()
    df_display = df[["cleaned", "date", "complaint"]]
    menu_choice = st.sidebar.radio("Menu", list(MENU.keys()))
    st.title("collection of koinworks complaints")
    st.write(df_display)
    d = st.date_input("Insert date")
    if st.button("search by date"):
        st.write(df_display[df_display["date"] == d.strftime("%Y-%m-%d")])
    MENU[menu_choice](df)


if __name__ == "__main__":
    main()
