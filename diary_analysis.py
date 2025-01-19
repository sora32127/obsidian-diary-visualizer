import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import numpy as np

st.set_page_config(page_title="日記分析", layout="wide")

st.title("日記分析ダッシュボード")

# 色の定義
SENTIMENT_COLORS = {
    'Very Negative': '#1a237e',  # 濃い青
    'Negative': '#4051b5',       # 青
    'Neutral': '#9e9e9e',        # グレー（中間色）
    'Positive': '#e53935',       # 赤
    'Very Positive': '#b71c1c'   # 濃い赤
}

# 感情のカテゴリーを定義（順序を保持）
SENTIMENT_CATEGORIES = [
    'Very Positive',
    'Positive',
    'Neutral',
    'Negative',
    'Very Negative'
]

# データの読み込みと前処理
@st.cache_data
def load_data():
    df = pd.read_csv('output/raw_contents.csv')
    df['date'] = pd.to_datetime(df['ymd'])
    df = df[~df['date'].dt.month.isin([1, 8])]
    df['month'] = df['date'].dt.strftime('%Y-%m')
    df['weekday'] = df['date'].dt.day_name()
    return df

df = load_data()

# タブを作成
tab1, tab2, tab3 = st.tabs(["月次分析", "曜日別分析", "基本統計情報"])

with tab1:
    st.header("月次分析")
    
    # 月ごとの日数を計算
    monthly_stats = []
    for month in sorted(df['month'].unique()):  # 月を昇順にソート
        year, month_num = map(int, month.split('-'))
        total_days = calendar.monthrange(year, int(month_num))[1]
        diary_days = df[df['month'] == month].shape[0]
        monthly_stats.append({
            'month': month,  # YYYY-MM形式を保持
            'total_days': total_days,
            'diary_days': diary_days,
            'completion_rate': (diary_days / total_days) * 100
        })
    
    monthly_df = pd.DataFrame(monthly_stats)
    
    # 充足率のグラフ
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=monthly_df['month'],  # YYYY-MM形式をそのまま使用
        y=monthly_df['completion_rate'],
        text=monthly_df['completion_rate'].round(1).astype(str) + '%',
        textposition='auto',
    ))
    fig1.update_layout(
        title='月別日記充足率',
        xaxis_title='月',
        yaxis_title='充足率 (%)',
        yaxis_range=[0, 100],
        xaxis=dict(
            tickformat='%Y-%m',  # X軸のフォーマットを指定
            tickmode='array',
            ticktext=monthly_df['month'],  # 表示するテキスト
            tickvals=monthly_df['month']   # 表示する位置
        )
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # 感情分布のグラフ
    monthly_sentiment = pd.crosstab(
        df['month'], 
        df['sentiment_label'], 
        normalize='index'
    ) * 100
    
    fig2 = go.Figure()
    for sentiment in SENTIMENT_CATEGORIES:
        if sentiment in monthly_sentiment.columns:
            fig2.add_trace(go.Bar(
                name=sentiment,
                x=monthly_sentiment.index,  # indexはすでにYYYY-MM形式
                y=monthly_sentiment[sentiment],
                text=monthly_sentiment[sentiment].round(1).astype(str) + '%',
                textposition='auto',
                marker_color=SENTIMENT_COLORS[sentiment]
            ))
    
    fig2.update_layout(
        barmode='stack',
        title='月別感情分布',
        xaxis_title='月',
        yaxis_title='割合 (%)',
        yaxis_range=[0, 100],
        xaxis=dict(
            tickformat='%Y-%m',  # X軸のフォーマットを指定
            tickmode='array',
            ticktext=monthly_sentiment.index,  # 表示するテキスト
            tickvals=monthly_sentiment.index   # 表示する位置
        )
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 月次分析タブの最後に追加
    st.header("日記詳細")
    
    # フィルター用のコントロール
    col1, col2 = st.columns(2)
    with col1:
        selected_months = st.multiselect(
            "月を選択",
            options=sorted(df['month'].unique()),
            default=sorted(df['month'].unique())
        )
    
    with col2:
        selected_sentiments = st.multiselect(
            "感情を選択",
            options=SENTIMENT_CATEGORIES,
            default=SENTIMENT_CATEGORIES
        )
    
    # データのフィルタリング
    filtered_df = df[
        (df['month'].isin(selected_months)) &
        (df['sentiment_label'].isin(selected_sentiments))
    ]
    
    # 表示用のデータフレームを作成
    display_df = filtered_df[['date', 'raw_content', 'sentiment_label']].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    
    # データテーブルの表示
    for _, row in display_df.iterrows():
        with st.expander(f"{row['date']} - {row['sentiment_label']}"):
            st.text_area(
                "内容",
                value=row['raw_content'],
                height=150,
                disabled=True
            )

with tab2:
    st.header("曜日別分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weekday_counts = df['weekday'].value_counts()
        total_weeks = len(df['month'].unique()) * 4
        weekday_completion = (weekday_counts / total_weeks) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weekday_completion.index,
            y=weekday_completion.values,
            text=weekday_completion.round(1).astype(str) + '%',
            textposition='auto',
        ))
        fig.update_layout(
            title='曜日別日記充足率',
            xaxis_title='曜日',
            yaxis_title='充足率 (%)',
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        weekday_sentiment = pd.crosstab(
            df['weekday'], 
            df['sentiment_label'], 
            normalize='index'
        ) * 100
        
        fig = go.Figure()
        for sentiment in SENTIMENT_CATEGORIES:
            if sentiment in weekday_sentiment.columns:
                fig.add_trace(go.Bar(
                    name=sentiment,
                    x=weekday_sentiment.index,
                    y=weekday_sentiment[sentiment],
                    text=weekday_sentiment[sentiment].round(1).astype(str) + '%',
                    textposition='auto',
                    marker_color=SENTIMENT_COLORS[sentiment]
                ))
        
        fig.update_layout(
            barmode='stack',
            title='曜日別感情分布',
            xaxis_title='曜日',
            yaxis_title='割合 (%)',
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("基本統計情報")
    
    total_days = sum(calendar.monthrange(2024, month)[1] for month in range(9, 13))
    total_completion_rate = (len(df) / total_days) * 100
    
    sentiment_distribution = df['sentiment_label'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("総日記数", len(df))
        st.metric("全体充足率", f"{total_completion_rate:.1f}%")
    
    with col2:
        sentiment_distribution = sentiment_distribution.reindex(SENTIMENT_CATEGORIES)
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_distribution.index,
            values=sentiment_distribution.values,
            text=sentiment_distribution.round(1).astype(str) + '%',
            textposition='inside',
            marker=dict(colors=[SENTIMENT_COLORS[cat] for cat in sentiment_distribution.index])
        )])
        
        fig.update_layout(
            title='感情分布',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("感情分布の詳細")
    sentiment_df = pd.DataFrame({
        '感情': sentiment_distribution.index,
        '割合 (%)': sentiment_distribution.values.round(1)
    })
    st.dataframe(sentiment_df, hide_index=True)