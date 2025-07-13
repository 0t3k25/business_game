# Business Game 2025 分析・予測ツール

Alexander諸島でのビジネスゲームデータを分析し、需要予測を行うStreamlitアプリケーションです。

## 機能

### データ分析
1. **販売実績分析**
   - プレイヤー別S島・H島販売数量推移（散布図）
   - 島別総販売数量推移

2. **広告効果分析**
   - ラジオ広告（WIL）投資効果
   - タウン誌広告投資効果
   - 広告ROI分析

3. **市場シェア分析**
   - S島・H島の市場シェア推移
   - 最新ラウンドのシェア表示

4. **財務分析**
   - プレイヤー別売上高推移
   - 価格戦略分析

### 需要予測
次ラウンドの需要を以下の手法で予測：
- 単純移動平均
- 加重移動平均
- 線形トレンド
- 指数平滑法

## セットアップ

### ローカル環境での実行

1. リポジトリをクローン
```bash
git clone https://github.com/yourusername/business-game-2025-analysis.git
cd business-game-2025-analysis
```

2. 仮想環境を作成（推奨）
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 必要なパッケージをインストール
```bash
pip install -r requirements.txt
```

4. アプリケーションを起動
```bash
streamlit run app.py
```

### Streamlit Community Cloudへのデプロイ

1. GitHubにリポジトリを作成し、コードをプッシュ
2. [Streamlit Community Cloud](https://streamlit.io/cloud)にログイン
3. "New app"をクリック
4. GitHubリポジトリを選択
5. Main file pathに`app.py`を指定
6. "Deploy"をクリック

## 使い方

1. アプリケーションを起動
2. サイドバーから以下のCSVファイルをアップロード：
   - `bzgame-fs-data.csv`：財務データ
   - `bzgame-result-info.csv`：ゲーム結果データ
3. 各タブで分析結果を確認
4. 「需要予測」タブで次ラウンドの予測値を確認

## データファイルの形式

### bzgame-result-info.csv
必須カラム：
- Player: プレイヤー名
- ラウンド: ラウンド番号
- S島販売数: S島での販売数量
- H島販売数: H島での販売数量
- S島価格: S島での販売価格
- H島価格: H島での販売価格
- WIL: ラジオ広告投資額
- S^EL: S島タウン誌広告投資額
- H^EL: H島タウン誌広告投資額

### bzgame-fs-data.csv
財務データ（分析に応じて使用）

## 注意事項

- CSVファイルは`shift_jis`エンコーディングで保存されている必要があります
- データは正しい形式で入力されている必要があります

## ライセンス

MIT License

## 作成者

Business Game 2025 Analysis Tool
