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

# --- 0. グラフのフォント・スタイル設定（超強力版） ---

# 1. OSに合わせてフォント候補のリストを作成
if os.name == 'nt':  # Windowsの場合
    # 英語名と日本語名の両方を候補に入れる
    font_candidates = ['MS Gothic', 'MS Gothic', 'Yu Gothic', 'YuGothic', 'Meiryo', 'sans-serif']
else:  # Mac / Linux (Streamlit Cloud) の場合
    font_candidates = ['Noto Sans CJK JP', 'AppleGothic', 'sans-serif']

# 2. Matplotlib / Seaborn / NetworkX 全体に候補リストを設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = font_candidates

# 3. 代表として最初に見つかったフォント名をPlotly等に渡すための設定
FONT_NAME = font_candidates[0]
sns.set(font=FONT_NAME, style='whitegrid')
plt.rcParams['font.family'] = FONT_NAME

# Plotlyの設定
pio.templates.default = "plotly_white"
pio.templates[pio.templates.default].layout.font.family = FONT_NAME

# --- 0. グラフのフォント・スタイル設定（OS自動判別・日本語化・文字化け対策） ---

# 1. 実行環境（OS）に合わせて最適な日本語フォントを自動選択
# Windowsなら 'MS Gothic'、Macなら 'AppleGothic'、Linux(Streamlit Cloud)なら 'Noto Sans CJK JP'
if os.name == 'nt':
    FONT_NAME = 'MS Gothic'
else:
    # サーバー環境（Linux等）やMacを考慮したフォントフォールバック
    FONT_NAME = 'Noto Sans CJK JP'

# 2. Matplotlib / Seaborn / NetworkX 用の日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [FONT_NAME, 'MS Gothic', 'AppleGothic', 'TakaoPGothic', 'sans-serif']

# Seabornのスタイル適用（フォントが上書きされないように再設定）
sns.set(font=FONT_NAME, style='whitegrid')
# NetworkXや一部のグラフバグを防ぐため、family側にも直接代入
plt.rcParams['font.family'] = FONT_NAME

# 3. Plotly 用の日本語フォント設定
pio.templates.default = "plotly_white"
pio.templates[pio.templates.default].layout.font.family = FONT_NAME

# --- 1. ページ設定 ---
st.set_page_config(page_title="TTC 睡眠x集中力 分析ダッシュボード", layout="wide")

# --- 2. データ読み込みと高度なクレンジング ---
@st.cache_data
def load_and_preprocess(uploaded_file):
    df = None
    
    base_columns = [
        'タイムスタンプ', '学科', '学年', '属性', 'バイト有無', '平日バイト曜日', 
        '睡眠時間', '睡眠満足度', '満足度が低い理由', '時間の使い方', 
        '集中度', '集中できない理由'
    ]
    
    # 🌟 アップロードされたファイルを最優先で読み込む
    if uploaded_file is not None:
        for enc in ['utf-8', 'shift_jis', 'cp932']:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except Exception:
                continue
                
        if df is not None:
            # 🌟 列数に応じて安全に列名をマッピング
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

    # 🌟 ファイルが未アップロード、または読み込みに失敗した場合のみデモデータを生成
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
        df.columns = base_columns

    # 🌟【null/空白対策】ベース列の未回答（NaN）を一括で補完して「null」を無くす
    df['学科'] = df['学科'].fillna('未回答').astype(str).str.strip()
    df['学年'] = df['学年'].fillna('未回答').astype(str).str.strip()
    df['属性'] = df['属性'].fillna('未回答').astype(str).str.strip()
    df['バイト有無'] = df['バイト有無'].fillna('未回答').astype(str).str.strip()
    df['平日バイト曜日'] = df['平日バイト曜日'].fillna('なし').astype(str).str.strip()
    df['睡眠時間'] = df['睡眠時間'].fillna('未回答').astype(str).str.strip()
    df['睡眠満足度'] = df['睡眠満足度'].fillna('未回答').astype(str).str.strip()
    df['満足度が低い理由'] = df['満足度が低い理由'].fillna('特になし').astype(str).str.strip()
    df['時間の使い方'] = df['時間の使い方'].fillna('未回答').astype(str).str.strip()
    df['集中度'] = df['集中度'].fillna('未回答').astype(str).str.strip()
    df['集中できない理由'] = df['集中できない理由'].fillna('特になし').astype(str).str.strip()

    # 学科名の表記揺れ吸収
    name_map = {
        'Ｉｏｔ＋ＡＩ科': 'IoT+AI科', 'ＩｏＴ＋ＡＩ科': 'IoT+AI科', 'Ｗｅｂ動画クリエイター科': 'Web動画クリエイター科',
        'データサイエンス＋ＡＩ科': 'データサイエンス＋AI科', 'データサイエンス＋AI科': 'データサイエンス＋AI科',
        'ｹﾞｰﾑ+ﾃﾞｼﾞﾀﾙｸリエイター科': 'ゲーム＋デジタルクリエイター科', '建 筑 科': '建築科'
    }
    df['学科名_修正'] = df['学科'].replace(name_map)

    # バイト日数の計算ロジック
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

    sleep_order = ['5時間以下', '5時間から6時間', '6時間から7時間', '7時間から8時間', '8時間以上', '未回答']
    focus_order = ["よく集中できている", "集中できている", "あまり集中できていない", "まったく集中できていない", "未回答"]
    sat_order = ["とても満足している", "満足している", "あまり満足していない", "まったく満足していない", "未回答"]
    
    df['睡眠時間'] = pd.Categorical(df['睡眠時間'], categories=sleep_order, ordered=True)
    df['集中度'] = pd.Categorical(df['集中度'], categories=focus_order, ordered=True)
    df['睡眠満足度'] = pd.Categorical(df['睡眠満足度'], categories=sat_order, ordered=True)

    return df, sleep_order, focus_order, sat_order


# --- 3. サイドバーナビゲーション & ファイルアップローダー ---
st.sidebar.title("🔍 分析メニュー")

uploaded_file = st.sidebar.file_uploader("アンケートCSVファイルをアップロードしてください", type=["csv"])

# データを読み込む
df, sleep_order, focus_order, sat_order = load_and_preprocess(uploaded_file)

if uploaded_file is not None:
    st.sidebar.success(f"✅ 実際のデータ（{len(df)}名分）を読み込みました！")
else:
    st.sidebar.info("💡 現在はデモデータ表示中。左メニューからCSVをアップロードしてください。")

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
            # 描画バグを防ぐため一時的に文字列型にキャストしてクロス集計
            ct = pd.crosstab(df_ct['睡眠時間'].astype(str), df_ct['集中度'].astype(str), normalize='index') * 100
            valid_sleep = [o for o in sleep_order if o in ct.index]
            ct = ct.reindex(valid_sleep)
            
            fig = px.bar(ct, color_discrete_map=color_map, barmode='stack', labels={'value':'割合 (%)', 'index':'睡眠時間'})
            fig.update_layout(font_family=FONT_NAME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("データがありません。")
        
    with c2:
        st.subheader("📌 集中を妨げる原因（全体ランキング）")
        df_reasons = df[~df['集中できない理由'].isin(['特になし', '未回答'])]
        if not df_reasons.empty:
            reasons = df_reasons['集中できない理由'].str.split(', ').explode().str.strip().value_counts().reset_index()
            reasons.columns = ['集中できない理由', 'count']
            fig_reason = px.bar(reasons, x='count', y='集中できない理由', orientation='h', color='count', color_continuous_scale='Blues')
            fig_reason.update_layout(font_family=FONT_NAME)
            st.plotly_chart(fig_reason, use_container_width=True)
        else:
            st.caption("データがありません。")

    st.divider()
    st.subheader("🕸️ 睡眠満足度と集中力の相関ネットワーク")
    
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
        else:
            st.caption("ネットワークを構成する有効なデータが不足しています。")


# --- 📊 閲覧者が選べる自由分析バー ---
elif menu == "📊 閲覧者が選べる自由分析バー":
    st.title("⚙️ インタラクティブ条件絞り込み")
    
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
        else:
            st.caption("表示可能なデータがありません。")


# --- 5. 睡眠・集中の詳細分析 ---
elif menu == "睡眠・集中の詳細分析":
    st.title("📉 統計的な分布と原因の多角分析")
    st.markdown("各タブを切り替えることで、数値を様々な角度から解析した統計グラフと詳細なデータ説明が閲覧できます。")
    df_plot = df[(df['睡眠時間'] != '未回答') & (df['集中度'] != '未回答')]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "① 要因ネットワークグラフ", 
        "② 集中度の円グラフ", 
        "③ 六角形ヒートマップ", 
        "📈 ④ 要因別の相関トレンド散布図", 
        "✨ ⑤ きれいなバイオリン分布図"
    ])
    
    with tab1:
        st.subheader("🕸️ 睡眠不満・集中できない理由の相関関係")
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
                nx.draw_networkx_labels(G, pos, font_family=FONT_NAME, font_size=10, ax=ax)
                nx.draw_networkx_edges(G, pos, edge_color='#cbd5e1', width=2, ax=ax)
                plt.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.caption("データが不足しています。")
        else:
            st.caption("理由データが不足しています。")

    with tab2:
        st.subheader("🍕 全体の集中度内訳（割合）")
        if not df_plot.empty:
            fig_pie = px.pie(df_plot, names='集中度', color='集中度', color_discrete_map=color_map, hole=0.4)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(font_family=FONT_NAME)
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        st.subheader("⬢ 六角形ヒートマップ（密集地帯）")
        if not df_plot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            hb = ax.hexbin(df_plot["睡眠時間_数値"], df_plot["集中度_数値"], gridsize=10, cmap="Blues", mincnt=1)
            ax.set_xlabel("睡眠時間 (時間)", fontfamily=FONT_NAME)
            ax.set_ylabel("集中度スコア", fontfamily=FONT_NAME)
            fig.colorbar(hb, ax=ax, label='学生数')
            st.pyplot(fig)
            plt.close(fig)

    with tab4:
        st.subheader("📊 アルバイト有無別の相関トレンド散布図")
        from plotly.subplots import make_subplots
        fig_custom = make_subplots(rows=1, cols=2, subplot_titles=("アルバイトあり (はい)", "アルバイトなし (いいえ)"), shared_yaxes=True)
        categories = ["はい", "いいえ"]
        colors = ["#ef4444", "#3b82f6"] 
        
        has_data = False
        for i, cat in enumerate(categories):
            sub_df = df_plot[df_plot["バイト有無"] == cat]
            if len(sub_df) > 1:
                has_data = True
                x_data = sub_df["睡眠時間_数値"].to_numpy()
                y_data = sub_df["集中度_数値"].to_numpy()
                x_jitter = x_data + np.random.uniform(-0.15, 0.15, size=len(x_data))
                y_jitter = y_data + np.random.uniform(-0.15, 0.15, size=len(y_data))
                
                fig_custom.add_trace(
                    go.Scatter(x=x_jitter, y=y_jitter, mode='markers', name=f"学生データ ({cat})",
                               marker=dict(color=colors[i], size=8, opacity=0.7, line=dict(width=1, color='DimGray'))),
                    row=1, col=i+1
                )
                try:
                    a, b = np.polyfit(x_data, y_data, 1)
                    x_line = np.linspace(4, 9, 10)
                    y_line = a * x_line + b
                    fig_custom.add_trace(
                        go.Scatter(x=x_line, y=y_line, mode='lines', name=f"トレンド傾向線",
                                   line=dict(color=colors[i], width=3, dash='dash')),
                        row=1, col=i+1
                    )
                except:
                    pass
        
        if has_data:
            fig_custom.update_layout(font_family=FONT_NAME, height=500, showlegend=False,
                                     yaxis=dict(title="集中度スコア (1〜4)", range=[0.5, 4.5]))
            fig_custom.update_xaxes(title_text="睡眠時間 (時間)", range=[3.5, 9.5])
            st.plotly_chart(fig_custom, use_container_width=True)
        else:
            st.caption("散布図を描画するためのデータが不足しています。")

    with tab5:
        st.subheader("🎻 きれいなバイオリン分布図（密度の波形）")
        df_violin = df_violin = df_plot[df_plot['睡眠時間'] != '未回答']
        if not df_violin.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(data=df_violin, x="睡眠時間", y="集中度_数値", palette="Pastel1", inner="quartile", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), fontfamily=FONT_NAME)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.caption("データが不足しています。")


# --- 6. 日本人 vs 留学生比較 ---
elif menu == "日本人 vs 留学生比較":
    st.title("⚖️ 属性・学科・学年の固定・自由比較")
    
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

    st.divider()
    st.subheader("🌳 階層構造の分析（属性 ＞ 学科 ＞ 学年 ＞ 集中）")
    df_sun = pd.concat([df_a, df_b]).drop_duplicates() if not (df_a.empty and df_b.empty) else df
    df_sun = df_sun.copy()
    
    df_sun = df_sun[(df_sun['属性'] != '未回答') & (df_sun['学科名_修正'] != '未回答') & (df_sun['学年'] != '未回答') & (df_sun['集中度'] != '未回答')]
    
    if not df_sun.empty:
        df_sun_grouped = df_sun.groupby(['属性', '学科名_修正', '学年', '集中度'], observed=False).size().reset_index(name='counts')
        df_sun_grouped = df_sun_grouped[df_sun_grouped['counts'] > 0]
        if not df_sun_grouped.empty:
            # 🌟【最重要修正】Plotlyの階層計算時のCategoricalエラーを防ぐため文字列にキャスト
            for col_path in ['属性', '学科名_修正', '学年', '集中度']:
                df_sun_grouped[col_path] = df_sun_grouped[col_path].astype(str)
                
            fig_sun = px.sunburst(df_sun_grouped, path=['属性', '学科名_修正', '学年', '集中度'], values='counts', color='集中度', color_discrete_map=color_map)
            fig_sun.update_layout(font_family=FONT_NAME)
            st.plotly_chart(fig_sun, use_container_width=True)
        else:
            st.caption("サンバースト図用のデータがありません。")
    else:
        st.caption("データが不足しています。")

    # --- 最下部の自由分析バー ---
    st.divider()
    st.header("🔍 要因・理由まで自由に選べるカスタム比較バー")

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
        st.markdown(f"**📊 自由比較1の集計結果 (該当: {len(df_free_1)}名)**")
        df_free_1_ct = df_free_1[df_free_1['集中度'] != '未回答']
        if not df_free_1_ct.empty:
            f_sleep_avg1 = df_free_1_ct[df_free_1_ct['睡眠時間_数値'] > 0]['睡眠時間_数値'].mean()
            f_focus_avg1 = (df_free_1_ct['集中度_数値'] >= 3).mean() * 100
            st.write(f"💡 平均睡眠時間: `{f_sleep_avg1:.1f}時間` / 集中良好率: `{f_focus_avg1:.1f}%`")
            
            val_counts = df_free_1_ct['集中度'].value_counts().reset_index()
            val_counts.columns = ['集中度', 'count']
            fig_free_1 = px.bar(val_counts, x='count', y='集中度', orientation='h', color='集中度', color_discrete_map=color_map, height=250)
            fig_free_1.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), font_family=FONT_NAME)
            st.plotly_chart(fig_free_1, use_container_width=True, key="free_chart_left_unique")
        else:
            st.caption("⚠️ 条件に一致するデータがありません。")

    with c_res2:
        st.markdown(f"**📊 自由比較2の集計結果 (該当: {len(df_free_2)}名)**")
        df_free_2_ct = df_free_2[df_free_2['集中度'] != '未回答']
        if not df_free_2_ct.empty:
            f_sleep_avg2 = df_free_2_ct[df_free_2_ct['睡眠時間_数値'] > 0]['睡眠時間_数値'].mean()
            f_focus_avg2 = (df_free_2_ct['集中度_数値'] >= 3).mean() * 100
            st.write(f"💡 平均睡眠時間: `{f_sleep_avg2:.1f}時間` / 集中良好率: `{f_focus_avg2:.1f}%`")
            
            val_counts2 = df_free_2_ct['集中度'].value_counts().reset_index()
            val_counts2.columns = ['集中度', 'count']
            fig_free_2 = px.bar(val_counts2, x='count', y='集中度', orientation='h', color='集中度', color_discrete_map=color_map, height=250)
            fig_free_2.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), font_family=FONT_NAME)
            st.plotly_chart(fig_free_2, use_container_width=True, key="free_chart_right_unique")
        else:
            st.caption("⚠️ 条件に一致するデータがありません。")
