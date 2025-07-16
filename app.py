import os
import pickle
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)

# --- 0. 全局设定 ---
print("Initializing application...")
# K_RECOMMEND 现在作为最大推荐数的默认值
K_RECOMMEND = 4 
CANDIDATE_POOL_SIZE = 50
NEW_COURSES = ["IIDS6XXXX", "IIDS67482"]

# --- 1. 加载所有预训练模型和静态数据 ---
print("Step 1: Loading all pre-trained models and data...")

# 加载课程元数据和热门度分数
try:
    EXCEL_PATH = os.path.join("data", "HDS Optional Enrolments Anonymised.xlsx")
    df_all = pd.read_excel(EXCEL_PATH)
    df_all = df_all[df_all['Course Code'].str.startswith('IIDS')].copy()
    df_all = df_all[df_all['Long Course Title'] != 'Multi-omics for Healthcare']
    for title, canonical in [('Decision Support Systems', None), ('Digital Transformation Project', 'IIDS61502')]:
        codes = df_all.loc[df_all['Long Course Title'] == title, 'Course Code'].unique().tolist()
        if codes:
            if canonical is None: canonical = sorted(codes)[0]
            df_all.loc[df_all['Long Course Title'] == title, 'Course Code'] = canonical
    df_all = df_all.drop_duplicates(subset=['Student ID Pseudonymised', 'Course Code'])

    course_meta_dict = {row['Course Code']: row['Long Course Title'] for _, row in df_all[['Course Code', 'Long Course Title']].drop_duplicates().iterrows()}
    course_support = (df_all['Course Code'].value_counts() / len(df_all['Student ID Pseudonymised'].unique())).to_dict()
    popular_courses_sorted = sorted(course_support.keys(), key=lambda c: course_support[c], reverse=True)
except Exception as e:
    print(f"Error loading course metadata: {e}")
    course_meta_dict, course_support, popular_courses_sorted = {}, {}, []

# 加载优化后的关联规则
try:
    RULES_PATH = os.path.join("rules", "optimized_rules.pkl")
    with open(RULES_PATH, "rb") as f:
        rules_df_final = pickle.load(f)
    if not rules_df_final.empty:
        rules_df_final['antecedents'] = rules_df_final['antecedents'].apply(lambda x: set(x) if isinstance(x, (list, frozenset)) else x)
        rules_df_final['consequents'] = rules_df_final['consequents'].apply(lambda x: set(x) if isinstance(x, (list, frozenset)) else x)
    print(f"Successfully loaded {len(rules_df_final)} optimized association rules.")
except Exception as e:
    print(f"Error loading association rules from {RULES_PATH}: {e}")
    rules_df_final = pd.DataFrame()

# 加载协同过滤组件
try:
    df_ui = df_all.assign(Enrolled=1).pivot_table(index='Student ID Pseudonymised', columns='Course Code', values='Enrolled', fill_value=0)
    S_cf = cosine_similarity(df_ui.values.T)
    idx_map = {c: i for i, c in enumerate(df_ui.columns)}
    print("Collaborative filtering components built.")
except Exception as e:
    print(f"Error building CF components: {e}")
    S_cf, idx_map = None, {}

# 加载最终的 XGBRanker 模型
try:
    MODEL_PATH = os.path.join("models", "ltr_model.pkl")
    with open(MODEL_PATH, 'rb') as f:
        ltr_model = pickle.load(f)
    print("XGBoost Ranker model loaded.")
except Exception as e:
    print(f"Error loading LTR model: {e}")
    ltr_model = None

# 其他静态数据
COURSE_SKILL_MAPPING = {
    'IIDS67682': {'skills': ['Machine Learning', 'Python', 'Data Analysis'], 'paths': ['Health Data Scientist', 'Machine Learning Engineer'], 'categories': ['programming', 'stats']},
    'IIDS67462': {'skills': ['Medical Imaging', 'Python', 'Mathematical Modeling'], 'paths': ['Medical Image Analyst', 'AI in Healthcare Specialist'], 'categories': ['imaging', 'programming', 'stats']},
    'IIDS67482': {'skills': ['Medical Imaging', 'AI', 'Deep Learning'], 'paths': ['Imaging Scientist', 'AI Researcher (Healthcare)'], 'categories': ['imaging', 'programming']},
    'IIDS67302': {'skills': ['Bioinformatics', 'Genomics'], 'paths': ['Clinical Bioinformatician', 'Genomic Data Scientist'], 'categories': ['biology']},
    'IIDS61402': {'skills': ['Decision Support', 'Clinical Informatics'], 'paths': ['Clinical Informatics Analyst', 'Healthcare IT Consultant'], 'categories': ['systems']},
    'IIDS60542': {'skills': ['Health Informatics', 'EHR'], 'paths': ['Health Informatics Specialist', 'Clinical Data Manager'], 'categories': ['systems', 'biology']},
    'IIDS68112': {'skills': ['RCTs', 'Statistics', 'R'], 'paths': ['Clinical Trial Statistician', 'Biostatistician'], 'categories': ['stats', 'systems']},
    'IIDS67612': {'skills': ['Advanced Statistics', 'Meta-Analysis'], 'paths': ['Biostatistician', 'Quantitative Researcher'], 'categories': ['stats']},
    'IIDS69052': {'skills': ['Digital Epidemiology', 'Public Health'], 'paths': ['Digital Epidemiologist', 'Public Health Analyst'], 'categories': ['biology', 'systems']},
    'IIDS67692': {'skills': ['Multi-modal Data', 'Omics', 'Data Fusion'], 'paths': ['Senior Health Data Scientist', 'Computational Biologist'], 'categories': ['programming', 'biology']},
    'IIDS61502': {'skills': ['Project Management', 'Health Informatics'], 'paths': ['Healthcare IT Consultant', 'Project Manager'], 'categories': ['systems']}
}
PREFERENCE_MAPPING = {
    'programming': ['IIDS67682', 'IIDS67462', 'IIDS67482', 'IIDS67692'],
    'stats': ['IIDS67612', 'IIDS68112', 'IIDS67682', 'IIDS67462'],
    'biology': ['IIDS67302', 'IIDS69052', 'IIDS60542', 'IIDS67692'],
    'imaging': ['IIDS67462', 'IIDS67482', 'IIDS67692'],
    'systems': ['IIDS60542', 'IIDS61402', 'IIDS69052', 'IIDS68112']
}
try:
    CSV_PATH = os.path.join("data", "Course and Student Grading Forms.csv")
    df_grading = pd.read_csv(CSV_PATH, dtype=str).fillna('')
    grading_forms = df_grading.to_dict(orient="records")
    
    # --- Start of new/modified logic ---
    df_all['Year'] = df_all['Term Code'].astype(str).str[1:3].astype(int) + 2000
    
    # Calculate total unique students per year
    yearly_student_totals = df_all.groupby('Year')['Student ID Pseudonymised'].nunique()
    
    # Calculate course enrollment counts per year
    yearly_course_counts = df_all.groupby(['Year', 'Course Code']).size().unstack(fill_value=0)
    
    # Calculate enrollment percentages
    year_course_percentages = (yearly_course_counts.div(yearly_student_totals, axis=0) * 100).fillna(0)
    
    all_chart_courses = sorted(list(year_course_percentages.columns))
    year_course_percentages_dict = year_course_percentages.to_dict(orient='index')
    years = sorted(list(year_course_percentages_dict.keys()))
    # --- End of new/modified logic ---

except Exception as e:
    print(f"Error loading frontend data: {e}")
    grading_forms, years, year_course_percentages_dict, all_chart_courses = [], [], {}, []


# --- 2. 推荐函数 (与最终的 train_model.py 对齐) ---
def get_features(input_set, candidate_course):
    cf_scores = [S_cf[idx_map[s], idx_map[candidate_course]] for s in input_set if s in idx_map and candidate_course in idx_map]
    cf_score_sum = np.sum(cf_scores) if cf_scores else 0
    cf_score_avg = np.mean(cf_scores) if cf_scores else 0
    ar_score = 0
    for _, r in rules_df_final.iterrows():
        if r['antecedents'].issubset(input_set) and candidate_course in r['consequents']:
            ar_score = max(ar_score, r['confidence'])
    popularity = course_support.get(candidate_course, 0)
    return [cf_score_sum, cf_score_avg, ar_score, popularity]

def recommend_ar_optimized(selected, num_rec=3):
    sel = set(selected)
    ar_raw = {}
    for _, r in rules_df_final.iterrows():
        if r['antecedents'].issubset(sel):
            for c in r['consequents']:
                if c not in sel: ar_raw[c] = max(ar_raw.get(c, 0), r['confidence'])
    return sorted(ar_raw.items(), key=lambda x: x[1], reverse=True)[:num_rec]

def recommend_cf(selected, num_rec=3):
    sel = set(selected)
    cf_raw = {}
    for c, j in idx_map.items():
        if c not in sel:
            raw_scores = [S_cf[idx_map[s], j] for s in selected if s in idx_map]
            avg_score = np.mean(raw_scores) if raw_scores else 0
            cf_raw[c] = avg_score
    return sorted(cf_raw.items(), key=lambda x: x[1], reverse=True)[:num_rec]

def recommend_ltr_hybrid(selected, num_rec=3):
    ar_candidates = [code for code, score in recommend_ar_optimized(selected, num_rec=CANDIDATE_POOL_SIZE)]
    cf_candidates = [code for code, score in recommend_cf(selected, num_rec=CANDIDATE_POOL_SIZE)]
    pop_candidates = popular_courses_sorted[:CANDIDATE_POOL_SIZE]
    candidate_pool = (set(ar_candidates) | set(cf_candidates) | set(pop_candidates)) - set(selected)
    if not candidate_pool: return []
    features_list = []
    candidate_list = list(candidate_pool)
    for course in candidate_list:
        features_list.append(get_features(selected, course))
    if not features_list: return []
    X_predict = np.array(features_list)
    scores = ltr_model.predict(X_predict)
    scored_candidates = sorted(zip(candidate_list, scores), key=lambda x: x[1], reverse=True)
    return scored_candidates[:num_rec]

def get_career_guidance(selected_codes):
    if not selected_codes: return None
    all_skills = [s for c in selected_codes if c in COURSE_SKILL_MAPPING for s in COURSE_SKILL_MAPPING[c]['skills']]
    all_paths = [p for c in selected_codes if c in COURSE_SKILL_MAPPING for p in COURSE_SKILL_MAPPING[c]['paths']]
    radar_categories = ['programming', 'stats', 'biology', 'imaging', 'systems']
    radar_counts = Counter()
    for code in selected_codes:
        if code in COURSE_SKILL_MAPPING:
            radar_counts.update(COURSE_SKILL_MAPPING[code].get('categories', []))
    radar_data = [radar_counts[cat] for cat in radar_categories]
    if not all_skills and not any(radar_data): return None
    return {
        "skills": [s for s, _ in Counter(all_skills).most_common(3)],
        "paths": [p for p, _ in Counter(all_paths).most_common(3)],
        "radarData": radar_data
    }

# --- 3. Flask 路由 ---
@app.route("/")
def index_route():
    return render_template(
        "index.html",
        grading_forms=grading_forms,
        course_meta_dict=course_meta_dict,
        new_courses=NEW_COURSES,
        logo_url=url_for('static', filename='img/logo.png'),
        years=years,
        year_course_percentages_json=json.dumps(year_course_percentages_dict),
        all_chart_courses_json=json.dumps(all_chart_courses)
    )

@app.route("/recommend", methods=["POST"])
def recommend_route():
    data = request.get_json()
    sel = data.get("selected", [])
    mode = data.get("mode", "hybrid")
    preferences = data.get("preferences", [])
    # !!! 核心修正: 使用前端发来的 num_rec，如果不存在则默认为 K_RECOMMEND
    num_rec = data.get("num_rec", K_RECOMMEND)

    recs_with_scores = []
    if preferences and not sel:
        candidate_courses = {c for pref in preferences for c in PREFERENCE_MAPPING.get(pref, []) if c in course_meta_dict}
        sorted_candidates = sorted(list(candidate_courses), key=lambda c: course_support.get(c, 0), reverse=True)
        recs_with_scores = [(code, course_support.get(code, 0)) for code in sorted_candidates[:num_rec]]
    elif mode == 'hybrid':
        recs_with_scores = recommend_ltr_hybrid(sel, num_rec=num_rec)
    elif mode == 'ar':
        recs_with_scores = recommend_ar_optimized(sel, num_rec=num_rec)
    elif mode == 'cf':
        recs_with_scores = recommend_cf(sel, num_rec=num_rec)
    else: # 'support'
        top_courses = sorted(course_support.items(), key=lambda item: item[1], reverse=True)
        recs_codes = [c for c, s in top_courses if c not in sel][:num_rec]
        recs_with_scores = [(code, course_support.get(code, 0)) for code in recs_codes]

    recs_details = []
    for code, score in recs_with_scores:
        features = get_features(sel, code)
        recs_details.append({
            "code": code,
            "title": course_meta_dict.get(code, ""),
            "support": features[3],
            "ar_score": features[2],
            "cf_score": features[1],
            "final_score": float(score)
        })

    guidance = get_career_guidance(sel)

    return jsonify({"recommendations": recs_details, "guidance": guidance})

# --- 4. 运行应用 ---
if __name__ == "__main__":
    print("\n--- Application is ready and can be accessed! ---")
    app.run(debug=False, host="0.0.0.0", port=5001)