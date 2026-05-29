# 修正版（IndentationError 修正済み）
# 主な修正:
# 1. with tab1: のインデント修正
# 2. plotly template font 部分の安全化
# 3. 一部のインデント崩れを統一

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
import urllib.request
import plotly.io as pio

# --- 0. グラフのフォント・スタイル設定（クロスプラットフォーム対応版） ---
def setup_japanese_font():
    """
    日本語フォントをセットアップする関数。
    Windows / Mac / Linux (GitHub, Streamlit Cloud) すべてで動作する。
    """
    preferred_fonts = [
        'Noto Sans CJK JP',
        'Noto Sans JP',
        'IPAGothic',
        'IPAPGothic',
        'Yu Gothic',
        'Meiryo',
        'MS Gothic',
        'AppleGothic',
        'Hiragino Sans',
        'Hiragino Kaku Gothic Pro',
    ]

    available = {f.name for f in fm.fontManager.ttflist}

    for font in preferred_fonts:
        if font in available:
            return font

    font_dir = os.path.expanduser("~/.fonts")
    font_path = os.path.join(font_dir, "NotoSansJP-Regular.ttf")

    if not os.path.exists(font_path):
        os.makedirs(font_dir, exist_ok=True)

        url = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"

        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            return "DejaVu Sans"

    try:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        return prop.get_name()
    except Exception:
        return "DejaVu Sans"


FONT_NAME = setup_japanese_font()

plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['axes.unicode_minus'] = False

sns.set(font=FONT_NAME, style='whitegrid')

pio.templates.default = "plotly_white"

try:
    pio.templates[pio.templates.default].layout.font.family = FONT_NAME
except Exception:
    pass

# --- 1. ページ設定 ---
st.set_page_config(
    page_title="TTC 睡眠x集中力 分析ダッシュボード",
    layout="wide"
)

# --- 2. データ読み込み ---
@st.cache_data
def load_and_preprocess(uploaded_file):

    df = None

    current_dir_files = os.listdir('.')
    csv_files = [
        f for f in current_dir_files
        if f.endswith('.csv') and not f.startswith('.')
    ]

    if uploaded_file is not None:

        for enc in ['utf-8', 'shift_jis', 'cp932']:

            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break

            except Exception:
                continue

    elif len(csv_files) > 0:

        for enc in ['utf-8', 'shift_jis', 'cp932']:

            try:
                df = pd.read_csv(csv_files[0], encoding=enc)
                break

            except Exception:
                continue

    # デモデータ
    if df is None:

        np.random.seed(42)

        data = {
            '学科': np.random.choice(
                ['データサイエンス＋AI科', '情報処理科', '建築科'],
                120
            ),
            '学年': np.random.choice(['1年生', '2年生'], 120),
            '属性': np.random.choice(['日本人', '留学生'], 120),
            'バイト有無': np.random.choice(['はい', 'いいえ'], 120),
            '平日バイト曜日': np.random.choice(
                ['月,水', '金', 'なし'],
                120
            ),
            '睡眠時間': np.random.choice(
                [
                    '5時間以下',
                    '5時間から6時間',
                    '6時間から7時間',
                    '7時間から8時間',
                    '8時間以上'
                ],
                120
            ),
            '睡眠満足度': np.random.choice(
                [
                    'とても満足している',
                    '満足している',
                    'あまり満足していない',
                    'まったく満足していない'
                ],
                120
            ),
            '満足度が低い理由': np.random.choice(
                ['ゲーム', 'Youtube', 'アルバイト'],
                120
            ),
            '時間の使い方': np.random.choice(
                ['2時間', '3時間', '4時間'],
                120
            ),
            '集中度': np.random.choice(
                [
                    'よく集中できている',
                    '集中できている',
                    'あまり集中できていない',
                    'まったく集中できていない'
                ],
                120
            ),
            '集中できない理由': np.random.choice(
                ['睡眠不足', '疲れ', 'スマホ'],
                120
            )
        }

        df = pd.DataFrame(data)

    # 欠損処理
    for col in df.columns:
        df[col] = df[col].fillna('未回答').astype(str)

    # 数値化
    sleep_map = {
        '5時間以下': 4.5,
        '5時間から6時間': 5.5,
        '6時間から7時間': 6.5,
        '7時間から8時間': 7.5,
        '8時間以上': 8.5
    }

    focus_map = {
        'よく集中できている': 4,
        '集中できている': 3,
        'あまり集中できていない': 2,
        'まったく集中できていない': 1
    }

    sat_map = {
        'とても満足している': 4,
        '満足している': 3,
        'あまり満足していない': 2,
        'まったく満足していない': 1
    }

    df['睡眠時間_数値'] = df['睡眠時間'].map(sleep_map)
    df['集中度_数値'] = df['集中度'].map(focus_map)
    df['満足度_数値'] = df['睡眠満足度'].map(sat_map)

    sleep_order = [
        '5時間以下',
        '5時間から6時間',
        '6時間から7時間',
        '7時間から8時間',
        '8時間以上'
    ]

    focus_order = [
        'よく集中できている',
        '集中できている',
        'あまり集中できていない',
        'まったく集中できていない'
    ]

    sat_order = [
        'とても満足している',
        '満足している',
        'あまり満足していない',
        'まったく満足していない'
    ]

    return df, sleep_order, focus_order, sat_order


# --- サイドバー ---
st.sidebar.title("🔍 分析メニュー")

uploaded_file = st.sidebar.file_uploader(
    "CSVアップロード",
    type=["csv"]
)

df, sleep_order, focus_order, sat_order = load_and_preprocess(uploaded_file)

menu = st.sidebar.radio(
    "表示切替",
    [
        "全体サマリー",
        "睡眠・集中の詳細分析"
    ]
)

color_map = {
    "よく集中できている": "#3b82f6",
    "集中できている": "#10b981",
    "あまり集中できていない": "#f59e0b",
    "まったく集中できていない": "#ef4444",
}

# =========================
# 全体サマリー
# =========================
if menu == "全体サマリー":

    st.title("🎓 全体サマリー")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("回答者数", f"{len(df)} 名")

    with c2:
        focus_rate = (df['集中度_数値'] >= 3).mean() * 100
        st.metric("集中良好率", f"{focus_rate:.1f}%")

    with c3:
        sat_rate = (df['満足度_数値'] >= 3).mean() * 100
        st.metric("睡眠満足率", f"{sat_rate:.1f}%")

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

    fig.update_layout(font_family=FONT_NAME)

    st.plotly_chart(fig, use_container_width=True)

# =========================
# 詳細分析
# =========================
elif menu == "睡眠・集中の詳細分析":

    st.title("📉 睡眠・集中 詳細分析")

    df_plot = df[
        (df['睡眠時間'] != '未回答') &
        (df['集中度'] != '未回答')
    ]

    tab1, tab2, tab3 = st.tabs([
        "1 要因相関ネットワーク",
        "2 バイオリン分布図",
        "3 平均集中度推移"
    ])

    # ==================================================
    # tab1
    # ==================================================
    with tab1:

        st.subheader("🕸️ 要因ネットワーク")

        G = nx.Graph()

        sample_nodes = [
            "睡眠不足",
            "スマホ",
            "疲れ",
            "集中低下",
            "ゲーム"
        ]

        for i in range(len(sample_nodes)-1):
            G.add_edge(sample_nodes[i], sample_nodes[i+1])

        fig, ax = plt.subplots(figsize=(10, 6))

        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="#3b82f6",
            node_size=2500,
            ax=ax
        )

        nx.draw_networkx_edges(
            G,
            pos,
            width=2,
            edge_color="#94a3b8",
            ax=ax
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_family=FONT_NAME,
            font_size=10,
            ax=ax
        )

        plt.axis("off")

        st.pyplot(fig)

        plt.close(fig)

    # ==================================================
    # tab2
    # ==================================================
    with tab2:

        st.subheader("🎻 バイオリン分布図")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(
            data=df_plot,
            x="睡眠時間",
            y="集中度_数値",
            hue="睡眠時間",
            legend=False,
            inner="quartile",
            palette="Pastel1",
            ax=ax
        )

        ax.set_xlabel("睡眠時間")
        ax.set_ylabel("集中度")

        st.pyplot(fig)

        plt.close(fig)

    # ==================================================
    # tab3
    # ==================================================
    with tab3:

        st.subheader("📈 平均集中度推移")

        sleep_avg_focus = (
            df_plot.groupby('睡眠時間')['集中度_数値']
            .mean()
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=sleep_avg_focus['睡眠時間'],
                y=sleep_avg_focus['集中度_数値'],
                mode='lines+markers+text',
                text=[
                    f"{v:.2f}"
                    for v in sleep_avg_focus['集中度_数値']
                ],
                textposition="top center"
            )
        )

        fig.update_layout(
            font_family=FONT_NAME,
            yaxis=dict(range=[1, 4.5]),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "睡眠時間が増えると集中度がどう変化するかを示しています。"
        )
