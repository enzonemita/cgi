# I社 離職予測プロジェクト

## 分析の流れ

1. **人間が仮説を立てる**
   - 「給与が低く、未婚で、出張が多い人が辞めやすいのでは？」

2. **モデルを学習**
   - ロジスティック回帰 & LightGBM で ROC-AUC を算出

3. **特徴量重要度を確認**
   - LightGBMの Feature Importance で「AIがどの指標を重視したか」を可視化

4. **誤分類データを分析**
   - AIが予測を間違えた人たちの元データを統計的に比較（Cohen's d 効果量を使用）し、AIの盲点を特定

5. **特徴量を追加・削除**
   - 盲点を補う指標を追加し、ノイズとなっている指標は削除

**→ 3〜5を繰り返して精度を改善**


## 最新の結果

### 💡 LightGBM 特徴量重要度 Top 10

| 特徴量 | 重要度 |
| :--- | :--- |
| `MonthlyIncome` | 521 |
| `MonthlyAchievement` | 438 |
| `Income_per_Age` | 419 |
| `PerformanceIndex` | 370 |
| `DistanceFromHome` | 339 |
| `OverTime` | 297 |
| `Age` | 278 |
| `Income_per_Year` | 270 |
| `StressRating` | 249 |
| `NumCompaniesWorked` | 211 |

> 予測を間違えた **33人** のデータを `error_analysis.csv` に保存しました。

### 📊 モデル評価結果

**ベースラインモデル (ロジスティック回帰)**
- **Accuracy**: 0.88435
- **Precision**: 0.76923
- **Recall**: 0.41667
- **F1 Score**: 0.54054
- **ROC-AUC**: 0.84731

**改善モデル (LightGBM)**
- **Accuracy**: 0.88776
- **Precision**: 0.82609
- **Recall**: 0.39583
- **F1 Score**: 0.53521
- **ROC-AUC**: 0.82266