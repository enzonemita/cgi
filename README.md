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


最新の結果
--- 💡 LightGBM 特徴量重要度 Top 10 ---
                  Feature  Importance
       num__MonthlyIncome         730
  num__MonthlyAchievement         644
    num__PerformanceIndex         502
            num__OverTime         400
   num__TotalWorkingYears         363
    num__DistanceFromHome         337
        num__StressRating         269
  num__NumCompaniesWorked         261
           num__Incentive         241
num__YearsWithCurrManager         222

-> 予測を間違えた 34 人のデータを 'error_analysis.csv' に保存しました！
wandb: 
wandb: Run history:
wandb:  base_accuracy ▁
wandb:        base_f1 ▁
wandb: base_precision ▁
wandb:    base_recall ▁
wandb:   base_roc_auc ▁
wandb:  lgbm_accuracy ▁
wandb:        lgbm_f1 ▁
wandb: lgbm_precision ▁
wandb:    lgbm_recall ▁
wandb:   lgbm_roc_auc ▁
wandb: 
wandb: Run summary:
wandb:  **base_accuracy 0.86054**
wandb:        base_f1 0.38806
wandb: base_precision 0.68421
wandb:    base_recall 0.27083
wandb:   base_roc_auc 0.81513
wandb:  **lgbm_accuracy 0.88435**
wandb:        lgbm_f1 0.52778
wandb: lgbm_precision 0.79167
wandb:    lgbm_recall 0.39583
wandb:   lgbm_roc_auc 0.81631