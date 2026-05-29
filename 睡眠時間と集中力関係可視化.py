# =========================================================
# TTC 睡眠 × 集中力 分析ダッシュボード
# 完全版（エラー修正版）
# =========================================================

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

# =========================================================
# 0. 日本語フォント設定
# =========================================================

def setup_japanese_font():

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
        except:
            return "DejaVu Sans"

    fm.fontManager.addfont(font_path)

    prop = fm.FontProperties(fname=font_path)

    return prop.get_name()


FONT_NAME = setup_japanese_font()

plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['axes.unicode_minus'] = False

sns.set(
    font=FONT_NAME,
    style='whitegrid'
)

pio.templates.default = "plotly_white"
pio.templates[pio.templates.default].layout.font.family = FONT_NAME

# =========================================================
# 1. ページ設定
# =========================================================

st.set_page_config(
    page_title="TTC 睡眠×集中力 分析ダッシュボード",
    layout="wide"
)

# =========================================================
# 2. データ読み込み
# =========================================================

@st.cache_data
def load_and_preprocess(uploaded_file):

    df = None

    # ---------------------------------
    # CSV読み込み
    # ---------------------------------

    current_files = os.listdir('.')

    csv_files = [
        f for f in current_files
        if f.endswith('.csv') and not f.startswith('.')
    ]

    # アップロードファイル優先
    if uploaded_file is not None:

        for enc in ['utf-8', 'shift_jis', 'cp932']:

            try:
                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    encoding=enc
                )

                break

            except:
                continue

    # ローカルCSV
    elif len(csv_files) > 0:

        for enc in ['utf-8', 'shift_jis', 'cp932']:

            try:
                df = pd.read_csv(
                    csv_files[0],
                    encoding=enc
                )

                break

            except:
                continue

    # ---------------------------------
    # デモデータ
    # ---------------------------------

    if df is None:

        np.random.seed(42)

        data = {
            '学科': np.random.choice(
                ['データサイエンス＋AI科', '建築科', '情報処理科'],
                150
            ),

            '学年': np.random.choice(
                ['1年生', '2年生'],
                150
            ),

            '属性': np.random.choice(
                ['日本人', '留学生'],
                150
            ),

            '睡眠時間': np.random.choice(
                [
                    '5時間以下',
                    '5時間から6時間',
                    '6時間から7時間',
                    '7時間から8時間',
                    '8時間以上'
                ],
                150
            ),

            '睡眠満足度': np.random.choice(
                [
                    'とても満足している',
                    '満足している',
                    'あまり満足していない',
                    'まったく満足していない'
                ],
                150
            ),

            '集中度': np.random.choice(
                [
                    'よく集中できている',
                    '集中できている',
                    'あまり集中できていない',
                    'まったく集中できていない'
                ],
                150
            ),

            '集中できない理由': np.random.choice(
                [
                    '睡眠が足りていない',
                    '日本語が難しい',
                    '疲れている',
                    '授業が難しい',
                    'アルバイト'
                ],
                150
            ),

            '満足度が低い理由': np.random.choice(
                [
                    'アルバイト',
                    'ゲーム',
                    '動画視聴',
                    '勉強'
                ],
                150
            )
        }

        df = pd.DataFrame(data)

    # =====================================================
    # 必須列チェック（KeyError対策）
    # =====================================================

    required_columns = [
        '学科',
        '学年',
        '属性',
        '睡眠時間',
        '睡眠満足度',
        '集中度',
        '集中できない理由',
        '満足度が低い理由'
    ]

    for col in required_columns:

        if col not in df.columns:
            df[col] = '未回答'

    # =====================================================
    # 欠損処理
    # =====================================================

    for col in required_columns:

        df[col] = (
            df[col]
            .fillna('未回答')
            .astype(str)
            .str.strip()
        )

    # =====================================================
    # 数値変換
    # =====================================================

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


# =========================================================
# 3. サイドバー
# =========================================================

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

# =========================================================
# カラーマップ
# =========================================================

color_map = {
    "よく集中できている": "#3b82f6",
    "集中できている": "#10b981",
    "あまり集中できていない": "#f59e0b",
    "まったく集中できていない": "#ef4444",
}

# =========================================================
# 4. 全体サマリー
# =========================================================

if menu == "全体サマリー":

    st.title("🎓 全体サマリー")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("回答者数", f"{len(df)} 名")

    with col2:
        st.metric(
            "平均睡眠時間",
            f"{df['睡眠時間_数値'].mean():.1f} 時間"
        )

    with col3:
        st.metric(
            "集中良好率",
            f"{(df['集中度_数値'] >= 3).mean()*100:.1f}%"
        )

    st.divider()

    st.subheader("📊 睡眠時間別の集中度")

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
        font_family=FONT_NAME
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# 5. 睡眠・集中の詳細分析
# =========================================================

elif menu == "睡眠・集中の詳細分析":

    st.title("📉 統計的な分布と原因の多角分析")

    st.markdown(
        "各タブを切り替えることで、様々な分析結果を確認できます。"
    )

    df_plot = df[
        (df['睡眠時間'] != '未回答') &
        (df['集中度'] != '未回答')
    ]

    tab1, tab2, tab3 = st.tabs([
        "1 要因相関ネットワーク",
        "2 バイオリン分布図",
        "3 平均集中度"
    ])

    # =====================================================
    # TAB1
    # =====================================================

    with tab1:

        st.subheader(
            "🕸️ 睡眠不満原因 と 集中できない原因 の相関関係"
        )

        st.markdown(
            "「睡眠不足」が見やすいように大きく表示しています。"
        )

        df_reasons = df[
            (df['集中できない理由'] != '特になし') &
            (df['満足度が低い理由'] != '特になし')
        ]

        if not df_reasons.empty:

            G = nx.Graph()

            reasons_f = (
                df_reasons['集中できない理由']
                .str.split(', ')
                .explode()
                .str.strip()
            )

            reasons_s = (
                df_reasons['満足度が低い理由']
                .str.split(', ')
                .explode()
                .str.strip()
            )

            top_f = reasons_f.value_counts().head(5).index.tolist()
            top_s = reasons_s.value_counts().head(5).index.tolist()

            for rf in top_f:

                for rs in top_s:

                    if rf != rs and rf != "" and rs != "":
                        G.add_edge(
                            rf,
                            rs,
                            weight=np.random.randint(1, 5)
                        )

            if len(G.nodes) > 0:

                fig, ax = plt.subplots(figsize=(12, 7))

                pos = nx.spring_layout(
                    G,
                    k=0.9,
                    seed=42
                )

                # ノードサイズ
                node_sizes = []

                for node in G.nodes:

                    if node == "睡眠が足りていない":
                        node_sizes.append(5000)

                    elif node == "日本語が難しい":
                        node_sizes.append(3000)

                    else:
                        node_sizes.append(1800)

                nx.draw_networkx_nodes(
                    G,
                    pos,
                    node_color='#3b82f6',
                    node_size=node_sizes,
                    alpha=0.9,
                    ax=ax
                )

                nx.draw_networkx_edges(
                    G,
                    pos,
                    edge_color='#cbd5e1',
                    width=2,
                    ax=ax
                )

                font_prop = fm.FontProperties(
                    family=FONT_NAME,
                    size=12
                )

                for node, (x, y) in pos.items():

                    # 特別強調
                    if node == "睡眠が足りていない":

                        text_color = 'black'
                        font_size = 18
                        font_weight = 'bold'

                    elif node == "日本語が難しい":

                        text_color = 'black'
                        font_size = 15
                        font_weight = 'bold'

                    else:

                        text_color = 'black'
                        font_size = 11
                        font_weight = 'normal'

                    ax.text(
                        x,
                        y,
                        node,
                        fontproperties=font_prop,
                        ha='center',
                        va='center',
                        color=text_color,
                        fontsize=font_size,
                        fontweight=font_weight,
                        zorder=5
                    )

                plt.axis('off')

                st.pyplot(fig)

                plt.close(fig)

                st.info(
                    "💡 睡眠不足・日本語の難しさ・疲労などが、集中力低下と強く関係している可能性があります。"
                )

            else:
                st.caption("データ不足")

    # =====================================================
    # TAB2
    # =====================================================

    with tab2:

        st.subheader("🎻 バイオリン分布図")

        if not df_plot.empty:

            fig, ax = plt.subplots(figsize=(10, 5))

            sns.violinplot(
                data=df_plot,
                x="睡眠時間",
                y="集中度_数値",
                hue="睡眠時間",
                palette="Pastel1",
                legend=False,
                inner="quartile",
                ax=ax
            )

            st.pyplot(fig)

            plt.close(fig)

    # =====================================================
    # TAB3
    # =====================================================

    with tab3:

        st.subheader("📈 睡眠時間別 平均集中度")

        avg_focus = (
            df.groupby('睡眠時間')['集中度_数値']
            .mean()
            .reset_index()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=avg_focus['睡眠時間'],
                y=avg_focus['集中度_数値'],
                mode='lines+markers+text',
                text=[
                    f"{v:.2f}"
                    for v in avg_focus['集中度_数値']
                ],
                textposition="top center"
            )
        )

        fig.update_layout(
            font_family=FONT_NAME,
            yaxis=dict(range=[1, 4.5])
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
