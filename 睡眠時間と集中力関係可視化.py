# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
import plotly.io as pio
import japanize_matplotlib

# =========================================================
# 日本語フォント設定（Streamlit Cloud 完全対応版）
# =========================================================

if os.name == 'nt':
    FONT_NAME = 'Meiryo'
else:
    FONT_NAME = 'Noto Sans CJK JP'

plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="whitegrid")
sns.set(font=FONT_NAME)

pio.templates.default = "plotly_white"
pio.templates[pio.templates.default].layout.font.family = FONT_NAME

# =========================================================
# ページ設定
# =========================================================

st.set_page_config(
    page_title="TTC 睡眠×集中力 分析ダッシュボード",
    layout="wide"
)

# =========================================================
# データ読み込み
# =========================================================

@st.cache_data
def load_and_preprocess(uploaded_file):

    df = None

    current_dir_files = os.listdir('.')
    csv_files = [
        f for f in current_dir_files
        if f.endswith('.csv') and not f.startswith('.')
    ]

    # アップロードファイル
    if uploaded_file is not None:
        for enc in ['utf-8', 'shift_jis', 'cp932']:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except:
                continue

    # フォルダ内CSV
    elif len(csv_files) > 0:
        for enc in ['utf-8', 'shift_jis', 'cp932']:
            try:
                df = pd.read_csv(csv_files[0], encoding=enc)
                break
            except:
                continue

    # デモデータ
    if df is None:

        np.random.seed(42)

        data = {
            '睡眠時間': np.random.choice(
                ['5時間以下', '5時間から6時間', '6時間から7時間', '7時間から8時間', '8時間以上'],
                150
            ),
            '集中度': np.random.choice(
                ['よく集中できている', '集中できている', 'あまり集中できていない', 'まったく集中できていない'],
                150
            ),
            '睡眠満足度': np.random.choice(
                ['とても満足している', '満足している', 'あまり満足していない', 'まったく満足していない'],
                150
            ),
            '集中できない理由': np.random.choice(
                [
                    '睡眠不足',
                    'アルバイト',
                    'スマホ',
                    'ゲーム',
                    '授業が難しい'
                ],
                150
            ),
            '満足度が低い理由': np.random.choice(
                [
                    'アルバイト',
                    '動画視聴',
                    'ゲーム',
                    '課題'
                ],
                150
            )
        }

        df = pd.DataFrame(data)

    # =====================================================
    # データクレンジング
    # =====================================================

    for col in df.columns:
        df[col] = df[col].fillna('未回答').astype(str).str.strip()

    # 数値変換
    focus_map = {
        'よく集中できている': 4,
        '集中できている': 3,
        'あまり集中できていない': 2,
        'まったく集中できていない': 1
    }

    sleep_map = {
        '5時間以下': 4.5,
        '5時間から6時間': 5.5,
        '6時間から7時間': 6.5,
        '7時間から8時間': 7.5,
        '8時間以上': 8.5
    }

    sat_map = {
        'とても満足している': 4,
        '満足している': 3,
        'あまり満足していない': 2,
        'まったく満足していない': 1
    }

    df['集中度_数値'] = df['集中度'].map(focus_map)
    df['睡眠時間_数値'] = df['睡眠時間'].map(sleep_map)
    df['満足度_数値'] = df['睡眠満足度'].map(sat_map)

    sleep_order = [
        '5時間以下',
        '5時間から6時間',
        '6時間から7時間',
        '7時間から8時間',
        '8時間以上'
    ]

    return df, sleep_order

# =========================================================
# サイドバー
# =========================================================

st.sidebar.title("📊 分析メニュー")

uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロード",
    type=['csv']
)

df, sleep_order = load_and_preprocess(uploaded_file)

menu = st.sidebar.radio(
    "画面切替",
    [
        "全体サマリー",
        "睡眠・集中の詳細分析"
    ]
)

# =========================================================
# カラーマップ
# =========================================================

color_map = {
    "よく集中できている": "#3b82f6",
    "集中できている": "#10b981",
    "あまり集中できていない": "#f59e0b",
    "まったく集中できていない": "#ef4444"
}

# =========================================================
# 全体サマリー
# =========================================================

if menu == "全体サマリー":

    st.title("🎓 全体サマリー")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("回答者数", len(df))

    with col2:
        good_focus = (df['集中度_数値'] >= 3).mean() * 100
        st.metric("集中良好率", f"{good_focus:.1f}%")

    with col3:
        good_sat = (df['満足度_数値'] >= 3).mean() * 100
        st.metric("睡眠満足率", f"{good_sat:.1f}%")

    st.divider()

    st.subheader("📊 睡眠時間別 集中度")

    ct = pd.crosstab(
        df['睡眠時間'],
        df['集中度'],
        normalize='index'
    ) * 100

    fig = px.bar(
        ct,
        barmode='stack',
        color_discrete_map=color_map
    )

    fig.update_layout(
        font_family=FONT_NAME,
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 詳細分析
# =========================================================

elif menu == "睡眠・集中の詳細分析":

    st.title("📉 睡眠・集中の詳細分析")

    tab1, tab2, tab3 = st.tabs([
        "ネットワーク",
        "バイオリン分布",
        "平均集中度"
    ])

    # =====================================================
    # ネットワーク
    # =====================================================

    with tab1:

        st.subheader("🕸️ 原因ネットワーク")

        G = nx.Graph()

        reasons_f = (
            df['集中できない理由']
            .str.split(', ')
            .explode()
            .str.strip()
        )

        reasons_s = (
            df['満足度が低い理由']
            .str.split(', ')
            .explode()
            .str.strip()
        )

        top_f = reasons_f.value_counts().head(5).index.tolist()
        top_s = reasons_s.value_counts().head(5).index.tolist()

        for rf in top_f:
            for rs in top_s:
                if rf != rs:
                    G.add_edge(
                        rf,
                        rs,
                        weight=np.random.randint(1, 5)
                    )

        fig, ax = plt.subplots(figsize=(10, 6))

        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color='#3b82f6',
            node_size=1800,
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edge_color='#cbd5e1',
            width=2,
            ax=ax
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_family=FONT_NAME,
            font_size=10,
            ax=ax
        )

        plt.axis('off')

        st.pyplot(fig)

        plt.close(fig)

    # =====================================================
    # バイオリン
    # =====================================================

    with tab2:

        st.subheader("🎻 バイオリン分布図")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(
            data=df,
            x="睡眠時間",
            y="集中度_数値",
            hue="睡眠時間",
            palette="Pastel1",
            legend=False,
            inner="quartile",
            ax=ax
        )

        ax.set_xlabel("睡眠時間", fontfamily=FONT_NAME)
        ax.set_ylabel("集中度スコア", fontfamily=FONT_NAME)

        for label in ax.get_xticklabels():
            label.set_fontfamily(FONT_NAME)

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        st.subheader("📊 睡眠満足度")

        ct_sat = pd.crosstab(
            df['睡眠時間'],
            df['睡眠満足度'],
            normalize='index'
        ) * 100

        fig_sat = px.bar(
            ct_sat,
            barmode='stack',
            height=450
        )

        fig_sat.update_layout(
            font_family=FONT_NAME,
            yaxis_range=[0, 100]
        )

        st.plotly_chart(fig_sat, use_container_width=True)

    # =====================================================
    # 平均集中度
    # =====================================================

    with tab3:

        st.subheader("📈 睡眠時間別 平均集中度")

        sleep_avg_focus = (
            df.groupby('睡眠時間')['集中度_数値']
            .mean()
            .reset_index()
        )

        sleep_avg_focus['睡眠時間'] = pd.Categorical(
            sleep_avg_focus['睡眠時間'],
            categories=sleep_order,
            ordered=True
        )

        sleep_avg_focus = sleep_avg_focus.sort_values('睡眠時間')

        fig_trend = go.Figure()

        fig_trend.add_trace(
            go.Scatter(
                x=sleep_avg_focus['睡眠時間'],
                y=sleep_avg_focus['集中度_数値'],
                mode='lines+markers+text',
                text=[
                    f"{v:.2f}"
                    for v in sleep_avg_focus['集中度_数値']
                ],
                textposition="top center",
                line=dict(
                    color='#10b981',
                    width=4
                ),
                marker=dict(
                    size=10,
                    color='#047857'
                )
            )
        )

        fig_trend.update_layout(
            font_family=FONT_NAME,
            xaxis_title="睡眠時間",
            yaxis=dict(
                title="平均集中度",
                range=[1, 4.5]
            ),
            height=500
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )
