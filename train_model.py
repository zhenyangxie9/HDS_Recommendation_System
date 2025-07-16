import os
import pickle
import random
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import xgboost as xgb

print("--- Starting Final Recommendation Model Training (Optimized Version) ---")

# --- 0. 全局设定和目录创建 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('rules'):
    os.makedirs('rules')

NEGATIVE_SAMPLING_RATIO = 1

# --- 1. 数据加载和预处理 ---
print("Step 1: Loading and preprocessing data...")
# !!! 注意: Flask应用从 'data/HDS...' 加载，这里保持一致
EXCEL_PATH = os.path.join("data", "HDS Optional Enrolments Anonymised.xlsx")
df_all = pd.read_excel(EXCEL_PATH)

# 数据清洗 (与我们最终版本一致)
df_all = df_all[df_all['Course Code'].str.startswith('IIDS')].copy()
df_all = df_all[df_all['Long Course Title'] != 'Multi-omics for Healthcare']
for title, canonical in [('Decision Support Systems', None), ('Digital Transformation Project', 'IIDS61502')]:
    codes = df_all.loc[df_all['Long Course Title'] == title, 'Course Code'].unique().tolist()
    if codes:
        if canonical is None: canonical = sorted(codes)[0]
        df_all.loc[df_all['Long Course Title'] == title, 'Course Code'] = canonical
df_all = df_all.drop_duplicates(subset=['Student ID Pseudonymised', 'Course Code'])

all_courses = set(df_all['Course Code'].unique())
course_support = (df_all['Course Code'].value_counts() / len(df_all['Student ID Pseudonymised'].unique())).to_dict()

# --- 2. 生成并保存关联规则 ---
print("\nStep 2: Generating and saving association rules...")
best_params = {'support': 0.03, 'confidence': 0.3}
student_baskets = df_all.groupby('Student ID Pseudonymised')['Course Code'].apply(list).tolist()
df_transactions = pd.get_dummies(pd.Series(student_baskets).explode()).groupby(level=0).max()

frequent_itemsets_final = apriori(df_transactions, min_support=best_params['support'], use_colnames=True)
rules_df_final = association_rules(frequent_itemsets_final, metric="confidence", min_threshold=best_params['confidence'])
rules_df_final['antecedents'] = rules_df_final['antecedents'].apply(set)
rules_df_final['consequents'] = rules_df_final['consequents'].apply(set)

# 保存规则, Flask app会使用这个文件
RULES_SAVE_PATH = os.path.join("rules", "optimized_rules.pkl")
with open(RULES_SAVE_PATH, "wb") as f:
    pickle.dump(rules_df_final, f)
print(f"Optimized association rules saved to: {RULES_SAVE_PATH}")


# --- 3. 构建协同过滤模型组件 ---
print("\nStep 3: Building collaborative filtering model components...")
df_ui = df_all.assign(Enrolled=1).pivot_table(index='Student ID Pseudonymised', columns='Course Code', values='Enrolled', fill_value=0)
S_cf = cosine_similarity(df_ui.values.T)
idx_map = {c: i for i, c in enumerate(df_ui.columns)}
# (这些组件将在 get_features 中使用，无需单独保存)


# --- 4. 特征工程 & 为XGBRanker生成训练数据 ---
print("\nStep 4: Generating training data for the ranking model...")
def get_features(input_set, candidate_course):
    """ 使用我们最终优化的特征集 """
    cf_scores = [S_cf[idx_map[s], idx_map[candidate_course]] for s in input_set if s in idx_map]
    cf_score_sum = np.sum(cf_scores) if cf_scores else 0
    cf_score_avg = np.mean(cf_scores) if cf_scores else 0
    ar_score = 0
    for _, r in rules_df_final.iterrows():
        if r['antecedents'].issubset(input_set) and candidate_course in r['consequents']:
            ar_score = max(ar_score, r['confidence'])
    popularity = course_support.get(candidate_course, 0)
    return [cf_score_sum, cf_score_avg, ar_score, popularity]

feature_names = ["CF_Score_Sum", "CF_Score_Avg", "AR_Score", "Popularity"]

X_train, y_train, groups_for_cv = [], [], []
group_id_counter = 0

student_courses = df_all.groupby('Student ID Pseudonymised')['Course Code'].apply(list).to_dict()
evaluation_set = {sid: courses for sid, courses in student_courses.items() if len(courses) >= 3}

for sid, courses in tqdm(evaluation_set.items(), desc="Generating training samples for XGBRanker"):
    random.shuffle(courses)
    num_input_courses = random.randint(2, len(courses) - 1)
    input_set = courses[:num_input_courses]
    test_set = set(courses[num_input_courses:])
    if not test_set: continue

    current_user_samples = []
    for course in test_set:
        current_user_samples.append((get_features(input_set, course), 1))
        
    num_neg_samples = len(test_set) * NEGATIVE_SAMPLING_RATIO
    user_courses = set(courses)
    negative_pool = list(all_courses - user_courses)
    neg_samples = random.sample(negative_pool, min(num_neg_samples, len(negative_pool)))
    for course in neg_samples:
        current_user_samples.append((get_features(input_set, course), 0))
    
    if current_user_samples:
        features, labels = zip(*current_user_samples)
        X_train.extend(features)
        y_train.extend(labels)
        groups_for_cv.extend([group_id_counter] * len(current_user_samples))
        group_id_counter += 1

X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
groups_for_cv_np = np.array(groups_for_cv)


# --- 5. 手动进行超参数搜索并训练最终模型 ---
print("\nStep 5: Manually searching for best XGBRanker parameters...")

param_grid = {
    'n_estimators': [100, 200], 'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0]
}
params_list = list(ParameterGrid(param_grid))
group_kfold = GroupKFold(n_splits=3)
mean_scores = []

for params in tqdm(params_list, desc="Tuning Hyperparameters"):
    fold_scores = []
    for train_idx, test_idx in group_kfold.split(X_train_np, y_train_np, groups=groups_for_cv_np):
        X_train_fold, X_test_fold = X_train_np[train_idx], X_train_np[test_idx]
        y_train_fold, y_test_fold = y_train_np[train_idx], y_train_np[test_idx]
        groups_train_fold = groups_for_cv_np[train_idx]
        
        _, group_sizes_train_fold = np.unique(groups_train_fold, return_counts=True)
        
        ranker = xgb.XGBRanker(**params, random_state=SEED)
        ranker.fit(X_train_fold, y_train_fold, group=group_sizes_train_fold)
        
        y_pred_scores = ranker.predict(X_test_fold)
        fold_scores.append(average_precision_score(y_test_fold, y_pred_scores))
    
    mean_scores.append(np.mean(fold_scores))

best_params_idx = np.argmax(mean_scores)
best_params = params_list[best_params_idx]
print(f"\nBest parameters found: {best_params} with score: {mean_scores[best_params_idx]:.4f}")

print("Training final model with best parameters...")
_, group_sizes_full = np.unique(groups_for_cv_np, return_counts=True)
ltr_model = xgb.XGBRanker(**best_params, random_state=SEED)
ltr_model.fit(X_train_np, y_train_np, group=group_sizes_full)

# --- 6. 保存最终的 XGBRanker 模型 ---
# 注意：我们不再需要 StandardScaler，因为XGBoost树模型对特征缩放不敏感
MODEL_SAVE_PATH = os.path.join("models", "ltr_model.pkl")
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(ltr_model, f)
print(f"XGBoost Ranker model saved to: {MODEL_SAVE_PATH}")
# (旧的 scaler.pkl 文件可以删除了，因为它不再被需要)

print("\n--- Model training complete! You can now run app.py. ---")