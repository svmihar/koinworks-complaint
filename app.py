from cluster_topic import get_top_k_word
import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud


def load_data():
    df = pd.read_csv("./data/eda.csv")
    df.dropna(inplace=True, subset=["cleaned"])
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(inplace=True, subset=["date"])
    return df


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
                    "top_words": ", ".join(
                        get_top_k_word(df[df[topic_cluster] == topic_id].cleaned.values)
                    ),
                    "topic_id": int(topic_id),
                }
            )
    for cluster_method, item in result.items():
        st.markdown(f'## {cluster_method}')
        for x in item:
            st.markdown(f'{x["topic_id"]}: {x["top_words"]}')
        st.markdown('---')



def eda(df):
    st.header("Analysis")
    st.subheader("wordcloud")
    chart = wordcloud(df)
    # TODO: add date picker here
    st.image(chart)
    st.header("tweets about koinworks")
    st.write(chart_trending_tweets(df))
    st.header("complaint tweets trending")
    st.write(chart_trending_complaints(df))


def classification(df):
    st.write("TBA")


def generation(df):
    st.write("TBA")


MENU = {"EDA": eda,'Complaint Topics':top_words_per_topic,"Classification": classification, "Text Generation": generation}


def main():
    df = load_data()
    menu_choice = st.sidebar.radio("Menu", list(MENU.keys()))
    st.title("collection of koinworks complaints")
    st.write(df)
    d= st.date_input('Insert date')
    if st.button('search by date'):
        st.write(df[df['date']==d.strftime('%Y-%m-%d')])
    st.subheader("haha")
    MENU[menu_choice](df)


if __name__ == "__main__":
    main()
