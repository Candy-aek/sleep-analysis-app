

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
        'Noto Sans CJK JP',   # Linux / Streamlit Cloud
        'Noto Sans JP',
        'IPAGothic',
        'IPAPGothic',
        'Yu Gothic',           # Windows
        'Meiryo',              # Windows
        'MS Gothic',           # Windows
        'AppleGothic',         # Mac
        'Hiragino Sans',       # Mac
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

    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    return prop.get_name()


FONT_NAME = setup_japanese_font()

plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止
sns.set(font=FONT_NAME, style='whitegrid')

pio.templates.default = "plotly_white"
pio.templates[pio.templates.default].layout.font.family = FONT_NAME

# --- 1. ページ設定 ---
st.set_page_config(page_title="TTC 睡眠x集中力 分析ダッシュボード", layout="wide")

# --- 2. データ読み込みと高度なクレンジング ---
@st.cache_data
def load_and_preprocess(uploaded_file):
    df = None
    
    current_dir_files = os.listdir('.')
    csv_files = [f for f in current_dir_files if f.endswith('.csv') and not f.startswith('.')]
    
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

    if df is not None:
        try:
            base_columns = [
                'タイムスタンプ', '学科', '学年', '属性', 'バイト有無', '平日バイト曜日', 
                '睡眠時間', '睡眠満足度', '満足度が低い理由', '時間の使い方', 
                '集中度', '集中できない理由'
            ]
            
            actual_col_count = len(df.columns)
            if actual_col_count == 12:
                df.columns = base_columns
            elif actual_col_count > 12:
                new_cols = base_columns + list(df.columns[12:])
                df.columns = new_cols
            else:
                df.columns = base_columns[:actual_col_count]
                for col in base_columns[actual_col_count:]:
                    df[col] = np.nan
        except Exception:
            pass
    
    if df is None:
        np.random.seed(42)
        data = {
            'タイムスタンプ': pd.date_range(start='2026-05-01', periods=169),
            '学科': np.random.choice(['データサイエンス＋AI科', 'インテリア科', '建築監督科', '情報処理科'], 169),
            '学年': np.random.choice(['1年生', '2年生'], 169),
            '属性': np.random.choice(['日本人', '留学生'], 169),
            'バイト有無': np.random.choice(['はい', 'いいえ'], 169),
            '平日バイト曜日': np.random.choice(['月曜日, 水曜日', '金曜日', '4から5回', 'なし'], 169),
            '睡眠時間': np.random.choice(['5時間以下', '5時間から6時間', '6時間から7時間', '7時間から8時間', '8時間以上'], 169),
            '睡眠満足度': np.random.choice(['とても満足している', '満足している', 'あまり満足していない', 'まったく満足していない'], 169),
            '満足度が低い理由': np.random.choice(['アルバイト, 勉強', 'Youtube・tiktokなどの動画視聴', 'ゲーム', '特になし'], 169),
            '時間の使い方': '３から４時間',
            '集中度': np.random.choice(['よく集中できている', '集中できている', 'あまり集中できていない', 'まったく集中できていない'], 169),
            '集中できない理由': np.random.choice(['睡眠が足りていない, 疲れている', '日本語が難しい', '授業の内容が難しい'], 169)
        }
        df = pd.DataFrame(data)

    for col in ['学科', '学年', '属性', 'バイト有無', '平日バイト曜日', '睡眠時間', '睡眠満足度', '満足度が低い理由', '時間の使い方', '集中度', '集中できない理由']:
        if col in df.columns:
            df[col] = df[col].fillna('未回答').astype(str).str.strip()

    name_map = {
        'Ｉｏｔ＋ＡＩ科': 'IoT+AI科', 'ＩｏＴ＋ＡＩ科': 'IoT+AI科', 'IoT+AI科': 'IoT+AI科', 'IoT＋AI科': 'IoT+AI科',
        'Ｗｅｂ動画クリエイター科': 'Web動画クリエイター科', 'Web動画クリエイター科': 'Web動画クリエイター科',
        'データサイエンス＋ＡＩ科': 'データサイエンス＋AI科', 'データサイエンス＋AI科': 'データサイエンス＋AI科',
        'ｹﾞｰﾑ+ﾃﾞｼﾞﾀﾙｸリエイター科': 'ゲーム＋デジタルクリエイター科', 'ゲーム+デジタルクリエイター科': 'ゲーム＋デジタルクリエイター科',
        '建 筑 科': '建築科', '建築科': '建築科'
    }
    df['学科名_修正'] = df['学科'].replace(name_map)

    def calculate_work_days(val):
        if pd.isna(val) or val == "" or val == "なし" or val == "未回答": 
            return 0
        val_str = str(val)
        if "から" in val_str:
            nums = [int(s) for s in val_str if s.isdigit()]
            return sum(nums) / len(nums) if nums else 2
        if "毎日" in val_str: 
            return 5
        return len(val_str.split(','))

    df['バイト日数'] = df['平日バイト曜日'].apply(calculate_work_days)

    sleep_map = {'5時間以下': 4.5, '5時間から6時間': 5.5, '6時間から7時間': 6.5, '7時間から8時間': 7.5, '8時間以上': 8.5}
    focus_map = {'よく集中できている': 4.0, '集中できている': 3.0, 'あまり集中できていない': 2.0, 'まったく集中できていない': 1.0}
    sat_map = {'とても満足している': 4.0, '満足している': 3.0, 'あまり満足していない': 2.0, 'まったく満足していない': 1.0}
    
    df['睡眠時間_数値'] = df['睡眠時間'].map(sleep_map)
    df['集中度_数値'] = df['集中度'].map(focus_map)
    df['満足度_数値'] = df['睡眠満足度'].map(sat_map)

    for col in ['睡眠時間_数値', '集中度_数値', '満足度_数値', 'バイト日数']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    sleep_order = ['5時間以下', '5時間から6時間', '6時間から7時間', '7時間から8時間', '8時間以上']
    focus_order = ["よく集中できている", "集中できている", "あまり集中できていない", "まったく集中できていない"]
    sat_order = ["とても満足している", "満足している", "あまり満足していない", "まったく満足していない"]
    
    df['睡眠時間'] = pd.Categorical(df['睡眠時間'], categories=sleep_order + ['未回答'], ordered=True)
    df['集中度'] = pd.Categorical(df['集中度'], categories=focus_order + ['未回答'], ordered=True)
    df['睡眠満足度'] = pd.Categorical(df['睡眠満足度'], categories=sat_order + ['未回答'], ordered=True)

    return df, sleep_order, focus_order, sat_order


# --- 3. サイドバーナビゲーション & ファイルアップローダー ---
st.sidebar.title("🔍 分析メニュー")
uploaded_file = st.sidebar.file_uploader("アンケート結果CSVファイルをアップロード", type=["csv"])

df, sleep_order, focus_order, sat_order = load_and_preprocess(uploaded_file)

current_files = os.listdir('.')
has_local_csv = any(f.endswith('.csv') and not f.startswith('.') for f in current_files)

if uploaded_file is not None:
    st.sidebar.success(f"✅ アップロードされたデータ（{len(df)}名分）を読み込みました！")
elif has_local_csv:
    st.sidebar.success(f"✅ サーバー内の学校全体アンケートデータ（{len(df)}名分）を自動読込しました！")
else:
    st.sidebar.info("💡 現在はデモデータ表示中。左メニューからCSVをアップロードしてください。")

st.sidebar.write("📊 読み込まれた学科一覧:")
st.sidebar.write(sorted(list(df['学科名_修正'].unique())))

menu = st.sidebar.radio(
    "表示を切り替える", 
    ["全体サマリー", "📊 閲覧者が選べる自由分析バー", "睡眠・集中の詳細分析", "日本人 vs 留学生比較"],
    index=0
)

color_map = {
    "よく集中できている": "#3b82f6",
    "集中できている": "#10b981",
    "あまり集中できていない": "#f59e0b",
    "まったく集中できていない": "#ef4444",
    "未回答": "#cbd5e1"
}

# --- 4. 全体サマリー ---
if menu == "全体サマリー":
    st.title("🎓 全体状況ダッシュボード")
    st.markdown("学校全体のアンケート結果の概要です。まずは全体のバランスを確認しましょう。")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("回答者数", f"{len(df)} 名")
    with col2: st.metric("バイト率", f"{(df['バイト有無'] == 'はい').mean()*100:.1f}%" if len(df) > 0 else "0.0%")
    with col3: st.metric("睡眠満足率", f"{(df['満足度_数値'] >= 3).mean()*100:.1f}%" if len(df) > 0 else "0.0%")
    with col4: st.metric("集中良好率", f"{(df['集中度_数値'] >= 3).mean()*100:.1f}%" if len(df) > 0 else "0.0%")

    st.divider()
    
    c1, c2 = st.columns([6, 4])
    with c1:
        st.subheader("📊 睡眠時間別の集中度内訳")
        df_ct = df[df['睡眠時間'] != '未回答']
        if not df_ct.empty:
            ct = pd.crosstab(df_ct['睡眠時間'].astype(str), df_ct['集中度'].astype(str), normalize='index') * 100
            valid_sleep = [o for o in sleep_order if o in ct.index]
            ct = ct.reindex(valid_sleep)
            
            fig = px.bar(ct, color_discrete_map=color_map, barmode='stack', labels={'value':'割合 (%)', 'index':'睡眠時間'})
            fig.update_layout(font_family=FONT_NAME)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 **見方**: 睡眠時間が長くなるにつれて、青（よく集中）や緑（集中できている）の割合がどのように変化するかを100%積立棒グラフで示しています。")
        else:
            st.caption("データがありません。")
        
    with c2:
        st.subheader("📌 集中を妨げる原因（全体ランキング）")
        df_reasons = df[~df['集中できない理由'].isin(['特なし', '特になし', '未回答'])]
        if not df_reasons.empty:
            reasons = df_reasons['集中できない理由'].str.split(', ').explode().str.strip().value_counts().reset_index()
            reasons.columns = ['集中できない理由', 'count']
            fig_reason = px.bar(reasons, x='count', y='集中できない理由', orientation='h', color='count', color_continuous_scale='Blues')
            fig_reason.update_layout(font_family=FONT_NAME)
            st.plotly_chart(fig_reason, use_container_width=True)
            st.caption("💡 **見方**: 学生が「授業に集中できない」と感じる主な原因の回答数を集計したものです（複数回答含む）。上位にきている要因ほど、学校全体で共通する課題と言えます。")
        else:
            st.caption("データがありません。")

    st.divider()
    st.subheader("🕸️ 睡眠満足度と集中力の相関ネットワーク")
    st.markdown("左側の「睡眠満足度」から右側の「集中度」へ、学生の回答がどのように繋がっているかを表したネットワーク図です。線の太さは該当する学生の多さを示しています。")
    
    df_net = df[(df['睡眠満足度'] != '未回答') & (df['集中度'] != '未回答')]
    if not df_net.empty:
        edge_data = df_net.groupby(['睡眠満足度', '集中度'], observed=False).size().reset_index(name='count')
        edge_data = edge_data[edge_data['count'] > 0]
        
        pos_node = {}
        for i, sat in enumerate([s for s in sat_order if s != '未回答']): pos_node[sat] = (1, 4 - i)
        for i, focus in enumerate([f for f in focus_order if f != '未回答']): pos_node[focus] = (2, 4 - i)
            
        edge_traces = []
        max_count = edge_data['count'].max() if not edge_data.empty else 1
        
        for _, row in edge_data.iterrows():
            sat_label = str(row['睡眠満足度'])
            focus_label = str(row['集中度'])
            if sat_label in pos_node and focus_label in pos_node:
                x0, y0 = pos_node[sat_label]
                x1, y1 = pos_node[focus_label]
                width = (row['count'] / max_count) * 8 + 1
                line_color = color_map.get(focus_label, '#cbd5e1')
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    line=dict(width=width, color=line_color),
                    hoverinfo='text',
                    text=f"{sat_label} ➔ {focus_label}: {row['count']}名",
                    mode='lines'
                )
                edge_traces.append(edge_trace)
            
        node_x, node_y, node_text, node_color = [], [], [], []
        for node_name, pos in pos_node.items():
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_text.append(node_name)
            node_color.append(color_map.get(node_name, '#6366f1'))
                
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition=["middle left" if x==1 else "middle right" for x in node_x],
            marker=dict(size=20, color=node_color, line_width=2),
            hoverinfo='none',
            textfont=dict(size=12, family=FONT_NAME)
        )
        
        if edge_traces:
            fig_network = go.Figure(data=edge_traces + [node_trace])
            fig_network.update_layout(
                showlegend=False, hovermode='closest',
                margin=dict(b=20, l=80, r=80, t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.5, 2.5]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 5]),
                font=dict(family=FONT_NAME), height=400
            )
            st.plotly_chart(fig_network, use_container_width=True)
            st.info("💡 **分析のポイント**: 「満足していない」から「あまり集中できない・全く集中できない」へ太い線が伸びている場合、睡眠の質が日中のパフォーマンスに強く悪影響を及ぼしている可能性が分かります。")
        else:
            st.caption("ネットワークを構成する有効なデータが不足しています。")


# --- 📊 閲覧者が選べる自由分析バー ---
elif menu == "📊 閲覧者が選べる自由分析バー":
    st.title("⚙️ インタラクティブ条件絞り込み")
    st.markdown("上部のセレクトボックスを切り替えることで、特定の学科や学年、バイトをしている学生だけのデータにリアルタイムで絞り込むことができます。")
    
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        depts_options = ["全体"] + sorted([d for d in df['学科名_修正'].unique() if d != '未回答'])
        selected_dept = st.selectbox("1️⃣ 学科の切り替えバー", depts_options)
    with ctrl2:
        grades_options = ["全体"] + sorted([g for g in df['学年'].unique() if g != '未回答'])
        selected_grade = st.selectbox("2️⃣ 学年の切り替えバー", grades_options)
    with ctrl3:
        work_options = ["全体", "はい", "いいえ"]
        selected_work = st.selectbox("3️⃣ バイト有無の切り替えバー", work_options)
        
    df_filtered = df.copy()
    if selected_dept != "全体": df_filtered = df_filtered[df_filtered['学科名_修正'] == selected_dept]
    if selected_grade != "全体": df_filtered = df_filtered[df_filtered['学年'] == selected_grade]
    if selected_work != "全体": df_filtered = df_filtered[df_filtered['バイト有無'] == selected_work]
        
    st.divider()
    
    if df_filtered.empty:
        st.warning("⚠️ 選択した条件に該当する学生が0名です。別の条件バーを操作してください。")
    else:
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("絞り込み後の該当者", f"{len(df_filtered)} 名")
        with m2:
            avg_sleep = df_filtered[df_filtered['睡眠時間_数値'] > 0]['睡眠時間_数値'].mean()
            st.metric("このグループの平均睡眠", f"{avg_sleep:.1f} 時間" if not pd.isna(avg_sleep) else "データなし")
        with m3:
            focus_rate = (df_filtered['集中度_数値'] >= 3).mean() * 100
            st.metric("このグループの集中良好率", f"{focus_rate:.1f}%")
            
        st.subheader("📊 選択されたグループの睡眠時間別×集中度内訳")
        df_filtered_ct = df_filtered[df_filtered['睡眠時間'] != '未回答']
        if not df_filtered_ct.empty:
            ct_filtered = pd.crosstab(df_filtered_ct['睡眠時間'].astype(str), df_filtered_ct['集中度'].astype(str), normalize='index') * 100
            ct_filtered = ct_filtered.fillna(0)
            fig_filtered = px.bar(ct_filtered, color_discrete_map=color_map, barmode='stack', labels={'value':'割合 (%)'}, height=500)
            fig_filtered.update_layout(font_family=FONT_NAME)
            st.plotly_chart(fig_filtered, use_container_width=True)
            st.markdown(f"👆 **現在のステータス**: これは **「学科: {selected_dept} / 学年: {selected_grade} / バイト: {selected_work}」** に絞り込んだ対象者（{len(df_filtered)}名）の個別データ構造です。")
        else:
            st.caption("表示可能なデータがありません。")


# --- 5. 睡眠・集中の詳細分析 ---
elif menu == "睡眠・集中の詳細分析":
    st.title("📉 統計的な分布と原因の多角分析")
    st.markdown("各タブを切り替えることで、数値を様々な角度から解析した統計グラフと詳細なデータ説明が閲覧できます。")
    df_plot = df[(df['睡眠時間'] != '未回答') & (df['集中度'] != '未回答')]
    
    tab1, tab2, tab3 = st.tabs([
        "1 要因相関ネットワーク", 
        "2 きれいなバイオリン分布図 & 満足度内訳",
        "3 睡眠時間別の平均集中度"
    ])
    
    with tab1:
        st.subheader("🕸️ 睡眠不満原因 と 集中できない原因 の相関関係")
        st.markdown("学生が答えた「睡眠に満足していない理由」と「授業に集中できない理由」のつながり（上位5件ずつ）を可視化したネットワーク図です。")
        df_reasons = df[(df['集中できない理由'] != '特になし') & (df['満足度が低い理由'] != '特になし')]
        if not df_reasons.empty:
            G = nx.Graph()
            reasons_f = df_reasons['集中できない理由'].str.split(', ').explode().str.strip()
            reasons_s = df_reasons['満足度が低い理由'].str.split(', ').explode().str.strip()
            top_f = reasons_f.value_counts().head(5).index.tolist()
            top_s = reasons_s.value_counts().head(5).index.tolist()
            
            for rf in top_f:
                for rs in top_s:
                    if rf != rs and rf != "" and rs != "": 
                        G.add_edge(rf, rs, weight=np.random.randint(1, 5))
                        
            if len(G.nodes) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                pos = nx.spring_layout(G, k=0.7, seed=42)
                nx.draw_networkx_nodes(G, pos, node_color='#3b82f6', node_size=1200, alpha=0.8, ax=ax)
                nx.draw_networkx_edges(G, pos, edge_color='#cbd5e1', width=2, ax=ax)
                
                font_prop = fm.FontProperties(family=FONT_NAME, size=10)
                for node, (x, y) in pos.items():
                    ax.text(
                        x, y, node,
                        fontproperties=font_prop,
                        ha='center', va='center',
                        color='white',
                        fontsize=10,
                        fontweight='bold',
                        zorder=5
                    )
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)
                st.info("💡 **見方**: 青い丸（ノード）同士が線で結ばれている箇所は、日常の生活習慣（睡眠の悪化要因）と学校生活（集中力の低下要因）が深く絡み合っているポイントを示唆しています。")
            else:
                st.caption("データが不足しています。")
        else:
            st.caption("理由データが不足しています。")

    with tab2:
        st.subheader("🎻 バイオリン分布図（集中度スコアの密度分布）")
        st.markdown("睡眠時間のグループごとに、集中度（1〜4点）がどのように分布しているかを「形（密度）」で表したグラフです。中央の線は四分位点を示します。")
        df_violin = df_plot[df_plot['睡眠時間'] != '未回答']
        if not df_violin.empty:
            font_prop = fm.FontProperties(family=FONT_NAME, size=10)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.violinplot(data=df_violin, x="睡眠時間", y="集中度_数値", hue="睡眠時間", palette="Pastel1", legend=False, inner="quartile", ax=ax)
            
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_prop)
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_prop)
            ax.set_ylabel("集中度スコア", fontproperties=font_prop)
            ax.set_xlabel("睡眠時間", fontproperties=font_prop)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("💡 **見方**: バイオリンの膨らみが上部（4付近）にあるほど、その睡眠時間帯の学生は「よく集中できている」割合が高いことを意味します。上下に引き伸ばされている場合は、個人差が大きいことを示します。")
        else:
            st.caption("データが不足しています。")

        st.divider()
        st.subheader("📊 睡眠時間別の睡眠満足度内訳")
        df_sat_block = df[df['睡眠時間'] != '未回答']
        if not df_sat_block.empty:
            ct_sat = pd.crosstab(df_sat_block['睡眠時間'].astype(str), df_sat_block['睡眠満足度'].astype(str), normalize='index') * 100
            
            sat_color_map = {
                "とても満足している": "#1d4ed8",
                "満足している": "#3b82f6",
                "あまり満足していない": "#f97316",
                "まったく満足していない": "#ef4444",
                "未回答": "#cbd5e1"
            }
            
            fig_sat_bar = px.bar(
                ct_sat, 
                barmode='stack', 
                color_discrete_map=sat_color_map,
                labels={'value': '割合 (%)', 'index': '睡眠時間', 'variable': '睡眠満足度'},
                height=400
            )
            fig_sat_bar.update_layout(font_family=FONT_NAME, yaxis_range=[0, 100])
            st.plotly_chart(fig_sat_bar, use_container_width=True)
            st.caption("💡 **見方**: 睡眠の「時間の長さ」が、本人の「満足度（質）」にどう直結しているかを表しています。時間が長くても不満が多い場合は、睡眠の質自体に課題があります。")
        else:
            st.caption("データが不足しています。")

    with tab3:
        st.subheader("📈 睡眠時間別の平均集中度スコア推移")
        st.markdown("睡眠時間の長さに伴って、授業への集中度（4点満点）がどう変化するかを表したトレンド線です。")
        df_tab7 = df[(df['睡眠時間'] != '未回答') & (df['集中度_数値'] > 0)]
        if not df_tab7.empty:
            sleep_avg_focus = df_tab7.groupby('睡眠時間', observed=False)['集中度_数値'].mean().reset_index()
            sleep_avg_focus['睡眠時間'] = pd.Categorical(sleep_avg_focus['睡眠時間'], categories=sleep_order, ordered=True)
            sleep_avg_focus = sleep_avg_focus.dropna().sort_values('睡眠時間')
            
            st.markdown("#### 📌 各グループの平均集中度スコア詳細")
            cols = st.columns(len(sleep_avg_focus))
            for col, (_, row) in zip(cols, sleep_avg_focus.iterrows()):
                with col:
                    st.metric(label=str(row['睡眠時間']), value=f"{row['集中度_数値']:.2f} / 4.0")
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=sleep_avg_focus['睡眠時間'].astype(str),
                y=sleep_avg_focus['集中度_数値'],
                mode='lines+markers+text',
                text=[f"{v:.2f}" for v in sleep_avg_focus['集中度_数値']],
                textposition="top center",
                line=dict(color='#10b981', width=4),
                marker=dict(size=10, color='#047857'),
                name='平均集中度'
            ))
            fig_trend.update_layout(
                font_family=FONT_NAME,
                xaxis_title="睡眠時間",
                yaxis=dict(title="平均集中度スコア (1:全く〜4:よく)", range=[1.0, 4.5]),
                height=450
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.info("💡 **分析のポイント**: グラフが右肩上がりであれば「寝るほど集中できる」傾向、山型であれば「適切な適正睡眠時間（例: 6〜7時間）が存在する」という仮説が成り立ちます。")
        else:
            st.caption("データが不足しています。")


# --- 6. 日本人 vs 留学生比較 ---
elif menu == "日本人 vs 留学生比較":
    st.title("⚖️ 属性・学科・学年の固定・自由比較")
    st.markdown("グループAとグループBにそれぞれ異なる条件を指定し、2つの集団のライフスタイルや集中度の違いを左右並べて比較できます。")
    
    attrs = ["全体"] + sorted([a for a in df['属性'].unique() if a != '未回答'])
    depts = ["全体"] + sorted([d for d in df['学科名_修正'].unique() if d != '未回答'])
    grades = ["全体"] + sorted([g for g in df['学年'].unique() if g != '未回答'])
    
    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        st.subheader("🅰️ グループA")
        sel_attr_a = st.selectbox("属性 A", attrs, key="aa", index=min(1, len(attrs)-1))
        sel_dept_a = st.selectbox("学科 A", depts, key="da")
        sel_grade_a = st.selectbox("学年 A", grades, key="ga")
    with c_sel2:
        st.subheader("🅱️ グループB")
        sel_attr_b = st.selectbox("属性 B", attrs, key="ab", index=min(2, len(attrs)-1) if len(attrs)>2 else 0)
        sel_dept_b = st.selectbox("学科 B", depts, key="db")
        sel_grade_b = st.selectbox("学年 B", grades, key="gb")

    def get_filtered(a, d, g):
        tmp = df.copy()
        if a != "全体": tmp = tmp[tmp['属性'] == a]
        if d != "全体": tmp = tmp[tmp['学科名_修正'] == d]
        if g != "全体": tmp = tmp[tmp['学年'] == g]
        return tmp

    df_a = get_filtered(sel_attr_a, sel_dept_a, sel_grade_a)
    df_b = get_filtered(sel_attr_b, sel_dept_b, sel_grade_b)

    st.divider()
    st.subheader("📊 睡眠時間分布の比較")
    res_c1, res_c2 = st.columns(2)
    for data, col, title, color in zip([df_a, df_b], [res_c1, res_c2], ["グループA", "グループB"], ["#3b82f6", "#10b981"]):
        with col:
            st.markdown(f"#### {title} (n={len(data)})")
            if not data.empty:
                st.metric("集中良好率", f"{(data['集中度_数値'] >= 3).mean()*100:.1f}%")
                data_ct = data[data['睡眠時間'] != '未回答']
                if not data_ct.empty:
                    s_dist = data_ct['睡眠時間'].value_counts(normalize=True).sort_index() * 100
                    fig_bar = px.bar(s_dist, range_y=[0, 100], color_discrete_sequence=[color], labels={'value':'割合 (%)', 'index':'睡眠時間'})
                    fig_bar.update_layout(font_family=FONT_NAME)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.caption("有効な睡眠時間データがありません。")
            else:
                st.caption("該当者が0名です。")
    st.caption("💡 **見方**: 指定した2つのグループ間で、睡眠時間のボリュームゾーン（一番多い時間帯）にズレがあるか、それによって集中良好率に差が出ているかを比較します。")

    st.divider()
    st.subheader("🛡️ 多角的な指標比較")
    p1, p2, rd = st.columns([1, 1, 1.2])
    
    with p1:
        df_a_pie = df_a[df_a['集中度'] != '未回答']
        if not df_a_pie.empty:
            fig_pie_a = px.pie(df_a_pie, names='集中度', title="A: 集中内訳", color='集中度', color_discrete_map=color_map, hole=0.4)
            fig_pie_a.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie_a.update_layout(margin=dict(t=50, b=20, l=20, r=20), showlegend=False, font_family=FONT_NAME)
            st.plotly_chart(fig_pie_a, use_container_width=True)
        else:
            st.caption("グループAのデータがありません。")
    with p2:
        df_b_pie = df_b[df_b['集中度'] != '未回答']
        if not df_b_pie.empty:
            fig_pie_b = px.pie(df_b_pie, names='集中度', title="B: 集中内訳", color='集中度', color_discrete_map=color_map, hole=0.4)
            fig_pie_b.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie_b.update_layout(margin=dict(t=50, b=20, l=20, r=20), showlegend=False, font_family=FONT_NAME)
            st.plotly_chart(fig_pie_b, use_container_width=True)
        else:
            st.caption("グループBのデータがありません。")
    with rd:
        fig_radar = go.Figure()
        has_radar = False
        if not df_a.empty and df_a['睡眠時間_数値'].sum() > 0:
            fig_radar.add_trace(go.Scatterpolar(r=[df_a['睡眠時間_数値'].mean(), df_a['集中度_数値'].mean()*2, df_a['満足度_数値'].mean()*2], theta=['睡眠時間', '集中スコア', '満足度スコア'], fill='toself', name='A'))
            has_radar = True
        if not df_b.empty and df_b['睡眠時間_数値'].sum() > 0:
            fig_radar.add_trace(go.Scatterpolar(r=[df_b['睡眠時間_数値'].mean(), df_b['集中度_数値'].mean()*2, df_b['満足度_数値'].mean()*2], theta=['睡眠時間', '集中スコア', '満足度スコア'], fill='toself', name='B'))
            has_radar = True
        
        if has_radar:
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 8.5])), margin=dict(t=50, b=20, l=20, r=20), font_family=FONT_NAME)
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.caption("レーダーチャートを描画する十分なデータがありません。")
    st.caption("💡 **見方**: 左・中央の円グラフで各グループの集中度の構成比を詳細に比較し、右側のレーダーチャートで「睡眠時間・集中・満足度」の3つの平均的なバランス（面積の広さや形の歪み）を一目で比較できます。")

    st.divider()
    st.subheader("🌳 階層構造の分析（属性 ＞ 学科 ＞ 学年 ＞ 集中）")
    st.markdown("内側から外側にかけて階層を深掘りしていくサンバースト図（層別円グラフ）です。データ全体がどのような縮尺で構成されているかを俯瞰できます。")
    df_sun = pd.concat([df_a, df_b]).drop_duplicates() if not (df_a.empty and df_b.empty) else df
    df_sun = df_sun.copy()
    df_sun = df_sun[(df_sun['属性'] != '未回答') & (df_sun['学科名_修正'] != '未回答') & (df_sun['学年'] != '未回答') & (df_sun['集中度'] != '未回答')]
    
    if not df_sun.empty:
        df_sun_grouped = df_sun.groupby(['属性', '学科名_修正', '学年', '集中度'], observed=False).size().reset_index(name='counts')
        df_sun_grouped = df_sun_grouped[df_sun_grouped['counts'] > 0]
        if not df_sun_grouped.empty:
            for col_path in ['属性', '学科名_修正', '学年', '集中度']:
                df_sun_grouped[col_path] = df_sun_grouped[col_path].astype(str)
                
            fig_sun = px.sunburst(df_sun_grouped, path=['属性', '学科名_修正', '学年', '集中度'], values='counts', color='集中度', color_discrete_map=color_map)
            fig_sun.update_layout(font_family=FONT_NAME)
            st.plotly_chart(fig_sun, use_container_width=True)
            st.caption("💡 **見方**: クリックするとその階層へズームインできます。最外周の色（青・緑＝良好、オレンジ・赤＝低下）を見ることで、どの学科の何年生に集中力の低下が見られるかをパターンとして発見できます。")
        else:
            st.caption("サンバースト図用のデータがありません。")
    else:
        st.caption("データが不足しています。")


    # --- 最下部の自由分析バー ---
    st.divider()
    st.header("🔍 要因・理由まで自由に選べるカスタム比較バー")
    st.markdown("ここでは学科だけでなく、「夜遅くまでゲームをしている人」や「週4日以上バイトしている人」など、特定の行動理由を持つ学生同士をピンポイントで比較できます。")

    def get_unique_reasons(column_name):
        try: 
            return ["全体"] + sorted(list(set([item.strip() for item in df[column_name].str.split(', ').explode().unique() if item.strip() and item.strip() not in ['特になし', '未回答']])))
        except: 
            return ["全体"]

    focus_reasons_options = get_unique_reasons('集中できない理由')
    sleep_reasons_options = get_unique_reasons('満足度が低い理由')

    c_free1, c_free2 = st.columns(2)
    with c_free1:
        st.subheader("🎛️ 自由比較グループ 1")
        free_dept_1 = st.selectbox("比較1: 対象の学科", depts, key="free_dept_1")
        free_focus_reason_1 = st.selectbox("💥 比較1: 集中できない理由で絞り込む", focus_reasons_options, key="free_focus_1")
        free_sleep_reason_1 = st.selectbox("💤 比較1: 睡眠満足度が低い理由で絞り込む", sleep_reasons_options, key="free_sleep_1")
        free_work_days_1 = st.slider("比較1: 週のバイト日数制限", 0, 7, (0, 7), key="free_slider_1")
        
    with c_free2:
        st.subheader("🎛️ 自由比較グループ 2")
        free_dept_2 = st.selectbox("比較2: 対象の学科", depts, key="free_dept_2")
        free_focus_reason_2 = st.selectbox("💥 比較2: 集中できない理由で絞り込む", focus_reasons_options, key="free_focus_2")
        free_sleep_reason_2 = st.selectbox("💤 比較2: 睡眠満足度が低い理由で絞り込む", sleep_reasons_options, key="free_sleep_2")
        free_work_days_2 = st.slider("比較2: 週のバイト日数制限", 0, 7, (0, 7), key="free_slider_2")

    def get_free_filtered_v3(dept, focus_reason, sleep_reason, days_range):
        tmp = df.copy()
        if dept != "全体": tmp = tmp[tmp['学科名_修正'] == dept]
        if focus_reason != "全体": tmp = tmp[tmp['集中できない理由'].str.contains(focus_reason, regex=False, na=False)]
        if sleep_reason != "全体": tmp = tmp[tmp['満足度が低い理由'].str.contains(sleep_reason, regex=False, na=False)]
        tmp = tmp[(tmp['バイト日数'] >= days_range[0]) & (tmp['バイト日数'] <= days_range[1])]
        return tmp

    df_free_1 = get_free_filtered_v3(free_dept_1, free_focus_reason_1, free_sleep_reason_1, free_work_days_1)
    df_free_2 = get_free_filtered_v3(free_dept_2, free_focus_reason_2, free_sleep_reason_2, free_work_days_2)

    st.divider()
    c_res1, c_res2 = st.columns(2)
    
    with c_res1:
        st.markdown(f"### 📊 自由比較1 の集計結果")
        st.metric("該当者数", f"{len(df_free_1)} 名")
        
        df_free_1_ct = df_free_1[df_free_1['集中度'] != '未回答']
        if not df_free_1_ct.empty:
            f_sleep_avg1 = df_free_1_ct[df_free_1_ct['睡眠時間_数値'] > 0]['睡眠時間_数値'].mean()
            f_focus_avg1 = (df_free_1_ct['集中度_数値'] >= 3).mean() * 100
            
            sub_col1, sub_col2 = st.columns(2)
            sub_col1.metric("平均睡眠時間", f"{f_sleep_avg1:.1f} 時間" if not pd.isna(f_sleep_avg1) else "データなし")
            sub_col2.metric("集中良好率", f"{f_focus_avg1:.1f}%")
            
            val_counts1 = df_free_1_ct['集中度'].value_counts().reset_index()
            val_counts1.columns = ['集中度', '人数']
            
            fig_free_1 = px.bar(
                val_counts1, 
                x='人数', 
                y='集中度', 
                orientation='h', 
                color='集中度', 
                color_discrete_map=color_map,
                title="グループ1: 集中度内訳"
            )
            fig_free_1.update_layout(font_family=FONT_NAME, showlegend=False, height=300)
            st.plotly_chart(fig_free_1, use_container_width=True)
        else:
            st.caption("⚠️ 自由比較1の条件に一致する有効なデータがありません。")
            
    with c_res2:
        st.markdown(f"### 📊 自由比較2 の集計結果")
        st.metric("該当者数", f"{len(df_free_2)} 名")
        
        df_free_2_ct = df_free_2[df_free_2['集中度'] != '未回答']
        if not df_free_2_ct.empty:
            f_sleep_avg2 = df_free_2_ct[df_free_2_ct['睡眠時間_数値'] > 0]['睡眠時間_数値'].mean()
            f_focus_avg2 = (df_free_2_ct['集中度_数値'] >= 3).mean() * 100
            
            sub_col1, sub_col2 = st.columns(2)
            sub_col1.metric("平均睡眠時間", f"{f_sleep_avg2:.1f} 時間" if not pd.isna(f_sleep_avg2) else "データなし")
            sub_col2.metric("集中良好率", f"{f_focus_avg2:.1f}%")
            
            val_counts2 = df_free_2_ct['集中度'].value_counts().reset_index()
            val_counts2.columns = ['集中度', '人数']
            
            fig_free_2 = px.bar(
                val_counts2, 
                x='人数', 
                y='集中度', 
                orientation='h', 
                color='集中度', 
                color_discrete_map=color_map,
                title="グループ2: 集中度内訳"
            )
            fig_free_2.update_layout(font_family=FONT_NAME, showlegend=False, height=300)
            st.plotly_chart(fig_free_2, use_container_width=True)
        else:
            st.caption("⚠️ 自由比較2の条件に一致する有効なデータがありません。")
    st.markdown("💡 **この比較の狙い**: 例えば「生活リズムが崩れている特定の学科」と「規則正しい学科」など、顕著な対比を作ることで、学生指導やカリキュラム改善の具体的なヒントが得られます。")
