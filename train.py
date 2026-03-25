import pandas as pd
import numpy as np
import random
import wandb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

# 警告を非表示（LightGBMのfeature_namesに関するものなど）
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 再現性のためのシード値の固定
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# 2. データの読み込み
# ==============================================================================
print("データの読み込み中...")
df = pd.read_csv("data.csv")
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
y = df["Attrition"]

# ==============================================================================
# 3. 実験設定（パターン）の定義
# ==============================================================================
# 過去の実験をまとめて記録できるように、特徴量の組み合わせをリストにしてループで回します
experiments = [
    {
        "name": "exp7_ultimate_insights",
        "numeric": [
            "MonthlyIncome", "JobLevel", "OverTime", "EnvironmentSatisfaction",
            "StockOptionLevel", "NumCompaniesWorked", "YearsSinceLastPromotion"
        ],
        "categorical": ["JobRole", "BusinessTravel"],
        "description": "婚姻状況を削除し、帰属意識や定着性(Stock, 転職回数など)を追加"
    }
]

# 各実験パターンを順番に実行して、WandB上に個別の記録として保存します
for exp in experiments:
    print(f"\n=========================================================")
    print(f"実行中: {exp['name']} ({exp['description']})")
    print(f"=========================================================")
    
    # WandBの初期化（実験パターンごとに新しいRunとして記録を開始）
    wandb.init(
        project="gci_attrition_prediction",
        name=exp["name"],
        config={
            "seed": SEED,
            "numeric_features": exp["numeric"],
            "categorical_features": exp["categorical"],
            "algorithms": ["LogisticRegression", "LightGBM"],
            "description": exp["description"]
        }
    )
    
    X = df[exp["numeric"] + exp["categorical"]]

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # データの前処理パイプラインの構築
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, exp["numeric"]),
        ("cat", categorical_transformer, exp["categorical"]),
    ])

    # === (1) ベースラインモデル (ロジスティック回帰) ===
    print("モデル学習(1/2): ロジスティック回帰...")
    baseline_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    y_proba_base = baseline_model.predict_proba(X_test)[:, 1]

    base_roc = roc_auc_score(y_test, y_proba_base)
    print(f"  -> 完了! ROC-AUC: {base_roc:.4f}")

    wandb.log({
        "base_accuracy": accuracy_score(y_test, y_pred_base),
        "base_precision": precision_score(y_test, y_pred_base, zero_division=0),
        "base_recall": recall_score(y_test, y_pred_base),
        "base_f1": f1_score(y_test, y_pred_base),
        "base_roc_auc": base_roc
    })

    # === (2) 改善モデル (LightGBM) ===
    print("モデル学習(2/2): LightGBM...")
    lgbm_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lgb.LGBMClassifier(random_state=SEED, n_estimators=200, learning_rate=0.05)),
    ])
    
    lgbm_model.fit(X_train, y_train)
    y_pred_lgb = lgbm_model.predict(X_test)
    y_proba_lgb = lgbm_model.predict_proba(X_test)[:, 1]

    lgb_roc = roc_auc_score(y_test, y_proba_lgb)
    print(f"  -> 完了! ROC-AUC: {lgb_roc:.4f}")

    wandb.log({
        "lgbm_accuracy": accuracy_score(y_test, y_pred_lgb),
        "lgbm_precision": precision_score(y_test, y_pred_lgb, zero_division=0),
        "lgbm_recall": recall_score(y_test, y_pred_lgb),
        "lgbm_f1": f1_score(y_test, y_pred_lgb),
        "lgbm_roc_auc": lgb_roc
    })

    # === (3) 分析機能の追加（特徴量重要度とエラーデータの抽出） ===
    print("\n---------------------------------------------------------")
    print("分析結果の抽出処理中...")
    
    # ① 特徴量重要度（Feature Importance）の取得と保存
    # パイプラインの前処理後につけられた「AIが理解できる特徴量名」を取得
    feature_names = lgbm_model.named_steps["preprocessor"].get_feature_names_out()
    # LightGBMが学習の中で「どれだけその指標を重視して分岐を作ったか」を取得
    importances = lgbm_model.named_steps["classifier"].feature_importances_
    
    # 見やすいように表にして、重要度が高い順に並び替え
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    # ターミナルにトップ10を発表
    print("\n--- 💡 LightGBM 特徴量重要度 Top 10 ---")
    print(fi_df.head(10).to_string(index=False))
    
    # 残りの全ランキングをCSVとして保存
    fi_df.to_csv("feature_importance.csv", index=False)
    
    # ② 誤分類データの抽出（Error Analysis）
    # AIが予測を間違えたテストデータだけを集めてCSVに保存する
    error_df = X_test.copy()            # テストデータ（答え合わせ用）
    error_df["Actual_Attrition"] = y_test   # 実際の正解
    error_df["Predicted_LGBM"] = y_pred_lgb # AIの予想
    
    # 実際の正解とAIの予想が一致しなかった（間違えた）人だけを取り出す
    wrong_predictions = error_df[error_df["Actual_Attrition"] != error_df["Predicted_LGBM"]]
    
    # 特徴量を探るためにCSVに保存して完了
    wrong_predictions.to_csv("error_analysis.csv", index=False)
    print(f"\n-> 予測を間違えた {len(wrong_predictions)} 人のデータを 'error_analysis.csv' に保存しました！")

    # 1つの実験パターンが終わったら、wandbのセッションを終了してサーバーに保存します
    wandb.finish()

print("\nすべての実験パターンとWandBへの記録が完了しました！")
