import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ページ設定
st.set_page_config(
    page_title="Business Game 2025 分析ツール", page_icon="📊", layout="wide"
)

# カスタムCSS
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# タイトルとヘッダー
st.title("🎮 Business Game 2025 データ分析・予測ツール")
st.markdown("### 📊 Alexander諸島ビジネスゲーム分析")

# サイドバー
st.sidebar.header("📁 データファイルをアップロード")
st.sidebar.markdown("---")

# ファイルアップローダー
fs_data_file = st.sidebar.file_uploader(
    "bzgame-fs-data CSV",
    type=["csv"],
    help="財務データのCSVファイルをアップロードしてください",
)
result_info_file = st.sidebar.file_uploader(
    "bzgame-result-info CSV",
    type=["csv"],
    help="ゲーム結果データのCSVファイルをアップロードしてください",
)


# データ分析用の関数
def calculate_market_share(df):
    """市場シェアを計算"""
    market_share = df.pivot_table(
        index="ラウンド",
        columns="Player",
        values=["S島販売数量", "H島販売数量"],
        aggfunc="sum",
    )

    s_total = market_share["S島販売数量"].sum(axis=1)
    s_share = market_share["S島販売数量"].div(s_total, axis=0) * 100

    h_total = market_share["H島販売数量"].sum(axis=1)
    h_share = market_share["H島販売数量"].div(h_total, axis=0) * 100

    return s_share, h_share, market_share


def predict_demand(demand_data, current_round):
    """複数の手法で需要を予測"""
    predictions = {}

    # 1. 単純移動平均（過去3期）
    if len(demand_data) >= 3:
        predictions["ma3_s"] = demand_data["S島販売数量"].tail(3).mean()
        predictions["ma3_h"] = demand_data["H島販売数量"].tail(3).mean()
    else:
        predictions["ma3_s"] = demand_data["S島販売数量"].mean()
        predictions["ma3_h"] = demand_data["H島販売数量"].mean()

    # 2. 加重移動平均
    if len(demand_data) >= 3:
        weights = [0.5, 0.3, 0.2]
        predictions["wma_s"] = np.average(
            demand_data["S島販売数量"].tail(3), weights=weights
        )
        predictions["wma_h"] = np.average(
            demand_data["H島販売数量"].tail(3), weights=weights
        )
    else:
        predictions["wma_s"] = demand_data["S島販売数量"].mean()
        predictions["wma_h"] = demand_data["H島販売数量"].mean()

    # 3. 線形トレンド予測
    if len(demand_data) >= 2:
        x = np.arange(len(demand_data))
        z_s = np.polyfit(x, demand_data["S島販売数量"], 1)
        p_s = np.poly1d(z_s)
        predictions["trend_s"] = p_s(len(demand_data))

        z_h = np.polyfit(x, demand_data["H島販売数量"], 1)
        p_h = np.poly1d(z_h)
        predictions["trend_h"] = p_h(len(demand_data))

        predictions["z_s"] = z_s
        predictions["z_h"] = z_h
    else:
        predictions["trend_s"] = demand_data["S島販売数量"].iloc[-1]
        predictions["trend_h"] = demand_data["H島販売数量"].iloc[-1]
        predictions["z_s"] = [0, 0]
        predictions["z_h"] = [0, 0]

    # 4. 指数平滑法
    alpha = 0.3
    if len(demand_data) >= 2:
        predictions["exp_s"] = (
            alpha * demand_data["S島販売数量"].iloc[-1]
            + (1 - alpha) * demand_data["S島販売数量"].iloc[-2]
        )
        predictions["exp_h"] = (
            alpha * demand_data["H島販売数量"].iloc[-1]
            + (1 - alpha) * demand_data["H島販売数量"].iloc[-2]
        )
    else:
        predictions["exp_s"] = demand_data["S島販売数量"].iloc[-1]
        predictions["exp_h"] = demand_data["H島販売数量"].iloc[-1]

    return predictions


# メイン処理
if fs_data_file and result_info_file:
    try:
        # データ読み込み
        with st.spinner("データを読み込んでいます..."):
            fs_data = pd.read_csv(fs_data_file, encoding="shift_jis")
            result_info = pd.read_csv(result_info_file, encoding="shift_jis")

        # データの前処理
        max_round = result_info["ラウンド"].max()
        current_round = max_round

        # 成功メッセージ
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"✅ データを正常に読み込みました")
        with col2:
            st.info(f"📅 現在のラウンド: {current_round}")
        with col3:
            st.info(f"👥 プレイヤー数: {result_info['Player'].nunique()}")

        # タブ作成
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📈 販売実績分析",
                "📢 広告効果分析",
                "🥧 市場シェア分析",
                "💰 財務分析",
                "🔮 需要予測",
            ]
        )

        with tab1:
            st.header("📈 販売実績分析")

            # プレイヤー別販売数量推移
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("S島販売数量推移（プレイヤー別）")
                fig_s = px.scatter(
                    result_info,
                    x="ラウンド",
                    y="S島販売数量",
                    color="Player",
                    title="S島販売数量推移",
                    labels={"S島販売数量": "S島販売数量", "ラウンド": "ラウンド"},
                    height=400,
                )
                fig_s.update_traces(marker=dict(size=12))
                fig_s.update_layout(
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor="lightgray"),
                    yaxis=dict(gridcolor="lightgray"),
                )
                st.plotly_chart(fig_s, use_container_width=True)

            with col2:
                st.subheader("H島販売数量推移（プレイヤー別）")
                fig_h = px.scatter(
                    result_info,
                    x="ラウンド",
                    y="H島販売数量",
                    color="Player",
                    title="H島販売数量推移",
                    labels={"H島販売数量": "H島販売数量", "ラウンド": "ラウンド"},
                    height=400,
                )
                fig_h.update_traces(marker=dict(size=12))
                fig_h.update_layout(
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor="lightgray"),
                    yaxis=dict(gridcolor="lightgray"),
                )
                st.plotly_chart(fig_h, use_container_width=True)

            # 総販売数量推移
            st.subheader("島別総販売数量推移")
            total_sales = (
                result_info.groupby("ラウンド")[["S島販売数量", "H島販売数量"]]
                .sum()
                .reset_index()
            )

            fig_total = go.Figure()
            fig_total.add_trace(
                go.Scatter(
                    x=total_sales["ラウンド"],
                    y=total_sales["S島販売数量"],
                    mode="lines+markers",
                    name="S島総販売数",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(size=10),
                )
            )
            fig_total.add_trace(
                go.Scatter(
                    x=total_sales["ラウンド"],
                    y=total_sales["H島販売数量"],
                    mode="lines+markers",
                    name="H島総販売数",
                    line=dict(color="#ff7f0e", width=3),
                    marker=dict(size=10),
                )
            )
            fig_total.update_layout(
                title="島別総販売数量推移",
                xaxis_title="ラウンド",
                yaxis_title="販売数量",
                height=400,
                plot_bgcolor="white",
                xaxis=dict(gridcolor="lightgray"),
                yaxis=dict(gridcolor="lightgray"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_total, use_container_width=True)

            # 統計情報
            st.subheader("📊 販売統計サマリー")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**S島販売統計**")
                s_stats = result_info.groupby("Player")["S島販売数量"].agg(
                    ["mean", "std", "sum"]
                )
                s_stats.columns = ["平均", "標準偏差", "合計"]
                st.dataframe(s_stats.round(1))
            with col2:
                st.markdown("**H島販売統計**")
                h_stats = result_info.groupby("Player")["H島販売数量"].agg(
                    ["mean", "std", "sum"]
                )
                h_stats.columns = ["平均", "標準偏差", "合計"]
                st.dataframe(h_stats.round(1))

        with tab2:
            st.header("📢 広告効果分析")

            # 広告費計算
            result_info["広告費合計"] = (
                result_info["ラジオ広告"]
                + result_info["S島タウン誌広告"]
                + result_info["H島タウン誌広告"]
            )
            result_info["販売数合計"] = (
                result_info["S島販売数量"] + result_info["H島販売数量"]
            )
            result_info["広告効率"] = result_info["販売数合計"] / (
                result_info["広告費合計"] + 1
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("ラジオ広告投資効果")
                fig_wil = px.scatter(
                    result_info,
                    x="ラジオ広告",
                    y="S島販売数量",
                    color="Player",
                    title="ラジオ広告投資額 vs S島販売数量",
                    trendline="ols",
                    height=400,
                )
                fig_wil.update_layout(
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor="lightgray"),
                    yaxis=dict(gridcolor="lightgray"),
                )
                st.plotly_chart(fig_wil, use_container_width=True)

            with col2:
                st.subheader("タウン誌広告投資効果")
                fig_tel = px.scatter(
                    result_info,
                    x="S島タウン誌広告",
                    y="S島販売数量",
                    color="Player",
                    title="S島タウン誌広告 vs S島販売数量",
                    trendline="ols",
                    height=400,
                )
                fig_tel.update_layout(
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor="lightgray"),
                    yaxis=dict(gridcolor="lightgray"),
                )
                st.plotly_chart(fig_tel, use_container_width=True)

            # 広告ROI分析
            st.subheader("広告投資効率（販売数/広告費）")
            fig_roi = px.line(
                result_info,
                x="ラウンド",
                y="広告効率",
                color="Player",
                title="広告投資効率推移",
                markers=True,
                height=400,
            )
            fig_roi.update_layout(
                plot_bgcolor="white",
                xaxis=dict(gridcolor="lightgray"),
                yaxis=dict(gridcolor="lightgray"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_roi, use_container_width=True)

            # 広告投資サマリー
            st.subheader("💡 広告投資サマリー")
            ad_summary = (
                result_info.groupby("Player")
                .agg(
                    {
                        "ラジオ広告": "mean",
                        "S島タウン誌広告": "mean",
                        "H島タウン誌広告": "mean",
                        "広告費合計": "mean",
                        "広告効率": "mean",
                    }
                )
                .round(1)
            )
            ad_summary.columns = [
                "ラジオ広告(平均)",
                "S島タウン誌(平均)",
                "H島タウン誌(平均)",
                "広告費合計(平均)",
                "広告効率(平均)",
            ]
            st.dataframe(ad_summary)

        with tab3:
            st.header("🥧 市場シェア分析")

            # 市場シェア計算
            s_share, h_share, market_share = calculate_market_share(result_info)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("S島市場シェア推移")
                fig_s_share = px.area(
                    s_share.T,
                    title="S島市場シェア（%）",
                    labels={"value": "シェア(%)", "index": "プレイヤー"},
                    height=400,
                )
                fig_s_share.update_layout(
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor="lightgray"),
                    yaxis=dict(gridcolor="lightgray", range=[0, 100]),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_s_share, use_container_width=True)

            with col2:
                st.subheader("H島市場シェア推移")
                fig_h_share = px.area(
                    h_share.T,
                    title="H島市場シェア（%）",
                    labels={"value": "シェア(%)", "index": "プレイヤー"},
                    height=400,
                )
                fig_h_share.update_layout(
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor="lightgray"),
                    yaxis=dict(gridcolor="lightgray", range=[0, 100]),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_h_share, use_container_width=True)

            # 最新ラウンドのシェア
            st.subheader(f"🎯 ラウンド{current_round}の市場シェア")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**S島市場シェア**")
                latest_s_share = pd.DataFrame(
                    {"シェア(%)": s_share.iloc[-1].round(1)}
                ).sort_values("シェア(%)", ascending=False)
                st.dataframe(latest_s_share)

            with col2:
                st.markdown("**H島市場シェア**")
                latest_h_share = pd.DataFrame(
                    {"シェア(%)": h_share.iloc[-1].round(1)}
                ).sort_values("シェア(%)", ascending=False)
                st.dataframe(latest_h_share)

            # シェア変動分析
            if len(s_share) > 1:
                st.subheader("📈 シェア変動分析（前ラウンドとの比較）")
                share_change = pd.DataFrame(
                    {
                        "S島シェア変化(%)": s_share.iloc[-1] - s_share.iloc[-2],
                        "H島シェア変化(%)": h_share.iloc[-1] - h_share.iloc[-2],
                    }
                ).round(1)

                # 色付けして表示
                def color_change(val):
                    color = "green" if val > 0 else "red" if val < 0 else "black"
                    return f"color: {color}"

                styled_change = share_change.style.applymap(color_change)
                st.dataframe(styled_change)

        with tab4:
            st.header("💰 財務分析")

            # 売上計算
            result_info["S島売上"] = result_info["S島販売数量"] * result_info["S島価格"]
            result_info["H島売上"] = result_info["H島販売数量"] * result_info["H島価格"]
            result_info["総売上"] = result_info["S島売上"] + result_info["H島売上"]

            # 売上推移
            st.subheader("売上高推移")
            fig_revenue = px.line(
                result_info,
                x="ラウンド",
                y="総売上",
                color="Player",
                title="プレイヤー別総売上高推移",
                markers=True,
                height=400,
            )
            fig_revenue.update_layout(
                plot_bgcolor="white",
                xaxis=dict(gridcolor="lightgray"),
                yaxis=dict(gridcolor="lightgray"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

            # 価格戦略分析
            st.subheader("価格戦略分析")
            col1, col2 = st.columns(2)

            with col1:
                fig_price_s = px.box(
                    result_info,
                    x="ラウンド",
                    y="S島価格",
                    title="S島価格分布",
                    height=400,
                )
                fig_price_s.update_layout(
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor="lightgray"),
                    yaxis=dict(gridcolor="lightgray"),
                )
                st.plotly_chart(fig_price_s, use_container_width=True)

            with col2:
                fig_price_h = px.box(
                    result_info,
                    x="ラウンド",
                    y="H島価格",
                    title="H島価格分布",
                    height=400,
                )
                fig_price_h.update_layout(
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor="lightgray"),
                    yaxis=dict(gridcolor="lightgray"),
                )
                st.plotly_chart(fig_price_h, use_container_width=True)

            # 価格と販売数の関係
            st.subheader("価格感応度分析")
            col1, col2 = st.columns(2)

            with col1:
                fig_price_vol_s = px.scatter(
                    result_info,
                    x="S島価格",
                    y="S島販売数量",
                    color="Player",
                    title="S島：価格vs販売数",
                    trendline="ols",
                    height=400,
                )
                st.plotly_chart(fig_price_vol_s, use_container_width=True)

            with col2:
                fig_price_vol_h = px.scatter(
                    result_info,
                    x="H島価格",
                    y="H島販売数量",
                    color="Player",
                    title="H島：価格vs販売数",
                    trendline="ols",
                    height=400,
                )
                st.plotly_chart(fig_price_vol_h, use_container_width=True)

            # 財務サマリー
            st.subheader("💵 財務サマリー")
            financial_summary = (
                result_info.groupby("Player")
                .agg({"総売上": ["mean", "sum"], "S島価格": "mean", "H島価格": "mean"})
                .round(0)
            )
            financial_summary.columns = [
                "平均売上",
                "累計売上",
                "S島平均価格",
                "H島平均価格",
            ]
            st.dataframe(financial_summary)

        with tab5:
            st.header("🔮 需要予測")

            # 総需要データの準備
            demand_data = (
                result_info.groupby("ラウンド")[["S島販売数量", "H島販売数量"]]
                .sum()
                .reset_index()
            )

            # 予測実行
            predictions = predict_demand(demand_data, current_round)

            st.subheader(f"📊 ラウンド{current_round + 1}の需要予測")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🏝️ S島総需要予測")
                predictions_s = pd.DataFrame(
                    {
                        "予測手法": [
                            "単純移動平均",
                            "加重移動平均",
                            "線形トレンド",
                            "指数平滑法",
                        ],
                        "予測値": [
                            int(predictions["ma3_s"]),
                            int(predictions["wma_s"]),
                            int(predictions["trend_s"]),
                            int(predictions["exp_s"]),
                        ],
                    }
                )
                st.dataframe(predictions_s, use_container_width=True)

                # 平均予測値
                avg_pred_s = int(predictions_s["予測値"].mean())
                st.metric(
                    "S島総需要予測（平均）",
                    f"{avg_pred_s}個",
                    f"{avg_pred_s - demand_data['S島販売数量'].iloc[-1]:.0f}",
                    delta_color="normal",
                )

            with col2:
                st.markdown("#### 🏝️ H島総需要予測")
                predictions_h = pd.DataFrame(
                    {
                        "予測手法": [
                            "単純移動平均",
                            "加重移動平均",
                            "線形トレンド",
                            "指数平滑法",
                        ],
                        "予測値": [
                            int(predictions["ma3_h"]),
                            int(predictions["wma_h"]),
                            int(predictions["trend_h"]),
                            int(predictions["exp_h"]),
                        ],
                    }
                )
                st.dataframe(predictions_h, use_container_width=True)

                # 平均予測値
                avg_pred_h = int(predictions_h["予測値"].mean())
                st.metric(
                    "H島総需要予測（平均）",
                    f"{avg_pred_h}個",
                    f"{avg_pred_h - demand_data['H島販売数量'].iloc[-1]:.0f}",
                    delta_color="normal",
                )

            # 予測根拠のグラフ
            st.subheader("📈 需要推移と予測")

            # 予測値を含むデータフレーム作成
            future_round = current_round + 1

            # グラフ作成
            fig_forecast = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("S島需要推移と予測", "H島需要推移と予測"),
                horizontal_spacing=0.15,
            )

            # S島
            fig_forecast.add_trace(
                go.Scatter(
                    x=demand_data["ラウンド"],
                    y=demand_data["S島販売数量"],
                    mode="lines+markers",
                    name="実績",
                    line=dict(color="#1f77b4", width=3),
                    marker=dict(size=10),
                ),
                row=1,
                col=1,
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=[current_round, future_round],
                    y=[demand_data["S島販売数量"].iloc[-1], avg_pred_s],
                    mode="lines+markers",
                    name="予測",
                    line=dict(color="#ff7f0e", dash="dash", width=3),
                    marker=dict(size=12, symbol="star"),
                ),
                row=1,
                col=1,
            )

            # H島
            fig_forecast.add_trace(
                go.Scatter(
                    x=demand_data["ラウンド"],
                    y=demand_data["H島販売数量"],
                    mode="lines+markers",
                    name="実績",
                    line=dict(color="#2ca02c", width=3),
                    marker=dict(size=10),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=[current_round, future_round],
                    y=[demand_data["H島販売数量"].iloc[-1], avg_pred_h],
                    mode="lines+markers",
                    name="予測",
                    line=dict(color="#d62728", dash="dash", width=3),
                    marker=dict(size=12, symbol="star"),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            fig_forecast.update_xaxes(
                title_text="ラウンド", row=1, col=1, gridcolor="lightgray"
            )
            fig_forecast.update_xaxes(
                title_text="ラウンド", row=1, col=2, gridcolor="lightgray"
            )
            fig_forecast.update_yaxes(
                title_text="販売数量", row=1, col=1, gridcolor="lightgray"
            )
            fig_forecast.update_yaxes(
                title_text="販売数量", row=1, col=2, gridcolor="lightgray"
            )
            fig_forecast.update_layout(
                height=400,
                plot_bgcolor="white",
                showlegend=True,
                legend=dict(x=0.45, y=1.15, orientation="h"),
            )

            st.plotly_chart(fig_forecast, use_container_width=True)

            # 最終予測結果
            st.success(
                f"""
            ### 🎯 ラウンド{future_round}の需要予測結果
            - **S島総需要予測: {avg_pred_s}個**
            - **H島総需要予測: {avg_pred_h}個**
            """
            )

            # 予測の信頼性評価
            st.subheader("📊 予測の信頼性評価")
            col1, col2 = st.columns(2)

            with col1:
                # S島の予測のばらつき
                s_std = predictions_s["予測値"].std()
                s_cv = (s_std / avg_pred_s * 100) if avg_pred_s > 0 else 0
                st.metric(
                    "S島予測の変動係数",
                    f"{s_cv:.1f}%",
                    help="値が小さいほど予測の信頼性が高い",
                )

            with col2:
                # H島の予測のばらつき
                h_std = predictions_h["予測値"].std()
                h_cv = (h_std / avg_pred_h * 100) if avg_pred_h > 0 else 0
                st.metric(
                    "H島予測の変動係数",
                    f"{h_cv:.1f}%",
                    help="値が小さいほど予測の信頼性が高い",
                )

            # 追加の分析情報
            with st.expander("📋 予測の詳細情報"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🏝️ S島の傾向分析**")
                    if len(demand_data) >= 2:
                        s_growth = (
                            (
                                demand_data["S島販売数量"].iloc[-1]
                                - demand_data["S島販売数量"].iloc[-2]
                            )
                            / demand_data["S島販売数量"].iloc[-2]
                            * 100
                        )
                        st.write(f"- 直近の成長率: {s_growth:.1f}%")
                        st.write(f"- トレンド係数: {predictions['z_s'][0]:.2f}")
                        st.write(f"- 過去3期平均: {predictions['ma3_s']:.0f}個")

                        # トレンド判定
                        if predictions["z_s"][0] > 5:
                            trend_msg = "📈 強い上昇トレンド"
                        elif predictions["z_s"][0] > 0:
                            trend_msg = "📈 緩やかな上昇トレンド"
                        elif predictions["z_s"][0] > -5:
                            trend_msg = "➡️ 横ばいトレンド"
                        else:
                            trend_msg = "📉 下降トレンド"
                        st.write(f"- トレンド判定: {trend_msg}")

                with col2:
                    st.markdown("**🏝️ H島の傾向分析**")
                    if len(demand_data) >= 2:
                        h_growth = (
                            (
                                demand_data["H島販売数量"].iloc[-1]
                                - demand_data["H島販売数量"].iloc[-2]
                            )
                            / demand_data["H島販売数量"].iloc[-2]
                            * 100
                        )
                        st.write(f"- 直近の成長率: {h_growth:.1f}%")
                        st.write(f"- トレンド係数: {predictions['z_h'][0]:.2f}")
                        st.write(f"- 過去3期平均: {predictions['ma3_h']:.0f}個")

                        # トレンド判定
                        if predictions["z_h"][0] > 5:
                            trend_msg = "📈 強い上昇トレンド"
                        elif predictions["z_h"][0] > 0:
                            trend_msg = "📈 緩やかな上昇トレンド"
                        elif predictions["z_h"][0] > -5:
                            trend_msg = "➡️ 横ばいトレンド"
                        else:
                            trend_msg = "📉 下降トレンド"
                        st.write(f"- トレンド判定: {trend_msg}")

                # 予測手法の説明
                st.markdown("---")
                st.markdown("### 📚 予測手法の説明")
                st.markdown(
                    """
                - **単純移動平均**: 過去3期の平均値による予測
                - **加重移動平均**: 直近のデータに重みをつけた予測（重み: 0.5, 0.3, 0.2）
                - **線形トレンド**: 線形回帰による長期的なトレンド予測
                - **指数平滑法**: 直近データを重視した予測（α=0.3）
                """
                )

    except Exception as e:
        st.error(f"⚠️ エラーが発生しました: {str(e)}")
        st.write(
            "ファイルの形式を確認してください。エンコーディングはshift_jisである必要があります。"
        )

        # デバッグ情報
        with st.expander("🔍 デバッグ情報"):
            st.write("エラーの詳細:", str(e))
            if "result_info" in locals():
                st.write("読み込まれたカラム:", list(result_info.columns))

else:
    # アップロード待機画面
    st.info("📁 左側のサイドバーから2つのCSVファイルをアップロードしてください。")

    # 使い方の説明
    st.markdown(
        """
    ### 📋 使い方
    
    1. **必要なファイル**をアップロード:
       - `bzgame-fs-data.csv`: 財務データ
       - `bzgame-result-info.csv`: ゲーム結果データ
    
    2. **分析結果**を確認:
       - 📈 販売実績分析: プレイヤー別の販売推移
       - 📢 広告効果分析: 広告投資のROI分析
       - 🥧 市場シェア分析: 各島でのシェア推移
       - 💰 財務分析: 売上・価格戦略の分析
       - 🔮 需要予測: 次ラウンドの需要予測
    
    ### 🎯 主な機能
    
    - **リアルタイム分析**: アップロードと同時に自動分析
    - **インタラクティブなグラフ**: ズーム・パン機能付き
    - **複数の予測手法**: 4つの異なる手法で需要予測
    - **プレイヤー比較**: 競合他社との比較分析
    
    ### ⚠️ 注意事項
    
    - CSVファイルは**shift_jis**エンコーディングで保存してください
    - ファイル名は正確に入力してください
    - データは正しい形式である必要があります
    """
    )

    # サンプルデータ形式
    with st.expander("📊 必要なデータ形式"):
        st.markdown(
            """
        **bzgame-result-info.csv の必須カラム:**
        - Player: プレイヤー名
        - ラウンド: ラウンド番号
        - S島販売数量: S島での販売数量
        - H島販売数量: H島での販売数量
        - S島価格: S島での販売価格
        - H島価格: H島での販売価格
        - ラジオ広告: ラジオ広告投資額
        - S島タウン誌広告: S島タウン誌広告投資額
        - H島タウン誌広告: H島タウン誌広告投資額
        """
        )

# フッター
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <small>Business Game 2025 Analysis Tool | Created with Streamlit</small>
</div>
""",
    unsafe_allow_html=True,
)
