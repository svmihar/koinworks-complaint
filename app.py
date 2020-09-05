import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud

@st.cache
def load_data():
    df= pd.read_csv('./data/eda.csv')
    df.dropna(inplace=True, subset=['cleaned'])
    df['date'] = pd.to_datetime(df['date'])
    df.dropna(inplace=True, subset=['date'])
    return df

@st.cache
def wordcloud(df:pd.DataFrame):
    wc = WordCloud(background_color='white', width=800, height=400)
    wc.generate(' '.join(df['cleaned'].values))
    img = wc.to_image()
    return img

def chart_trending_tweets(df):
    df_ = df.groupby(df['date'].dt.month)['username'].count().reset_index()
    df_.columns=['month', 'tweet_count']
    fig = px.line(df_, x="month", y="tweet_count")
    return fig

def chart_trending_complaints(df):
    df_ = df[df['complaint']==1]
    df_ = df_.groupby(df_['date'].dt.month)['username'].count().reset_index()
    df_.columns=['month', 'complaint_count']
    fig = px.line(df_, x="month", y="complaint_count")
    return fig



def main():
    df = load_data()
    st.title('collection of koinworks complaints')
    st.subheader('haha')
    st.write(df)

    st.header('wordcloud')
    chart = wordcloud(df)
    st.image(chart)
    st.header('tweets about koinworks')
    st.write(chart_trending_tweets(df))
    # insert time series tweet count here
    st.header('complaint tweets trending')
    st.write(chart_trending_complaints(df))
    # insert times series of complaint tweets here


if __name__ == "__main__":
    main()