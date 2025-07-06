import os
import pickle
import traceback
import json

from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------------------------------------------------------------------------
# 新课列表：2026 年新增课程
# -------------------------------------------------------------------------
NEW_COURSES = [
    "IIDS6XXXX",
    "IIDS67482",
    # …如有更多，请继续添加…
]

# -------------------------------------------------------------------------
# 1. 读取 “Course and Student Grading Forms” CSV
# -------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__),
                        "data",
                        "Course and Student Grading Forms.csv")
df_grading = pd.read_csv(CSV_PATH, dtype=str)
for col in ['Exam', 'Average Rating', 'Average Satisfaction']:
    if col in df_grading.columns:
        df_grading[col] = df_grading[col].fillna('')
grading_forms = df_grading.to_dict(orient="records")

# -------------------------------------------------------------------------
# 2. 读取选课原始数据并预处理
# -------------------------------------------------------------------------
EXCEL_PATH = os.path.join(os.path.dirname(__file__),
                          "data",
                          "HDS Optional Enrolments Anonymised.xlsx")
df_all = pd.read_excel(EXCEL_PATH)

# 2.1 只保留以 IIDS 开头
df_all = df_all[df_all['Course Code'].str.startswith('IIDS')].copy()
# 2.2 删除 Multi-omics for Healthcare
df_all = df_all[df_all['Long Course Title'] != 'Multi-omics for Healthcare']
# 2.3 合并同名课程
for title, canonical in [
    ('Decision Support Systems', None),
    ('Digital Transformation Project', 'IIDS61502'),
]:
    codes = df_all.loc[df_all['Long Course Title'] == title, 'Course Code'].unique().tolist()
    if not codes:
        continue
    if canonical is None:
        canonical = sorted(codes)[0]
    df_all.loc[df_all['Long Course Title'] == title, 'Course Code'] = canonical
# 2.4 去重（按学生+课程）
df_all = df_all.drop_duplicates(subset=['Student ID Pseudonymised', 'Course Code'])

# -------------------------------------------------------------------------
# 3. 统计支持度 & 热门课程
# -------------------------------------------------------------------------
course_meta_dict = {
    row['Course Code']: row['Long Course Title']
    for _, row in df_all[['Course Code', 'Long Course Title']].drop_duplicates().iterrows()
}
course_counts = df_all['Course Code'].value_counts()
total = course_counts.sum()
course_support = (course_counts / total).to_dict()
top4_courses = [(c, course_support[c]) for c in course_counts.index[:4]]

# -------------------------------------------------------------------------
# 4. 用户-物品矩阵 & CF 相似度
# -------------------------------------------------------------------------
df_ui = df_all.assign(Enrolled=1).pivot_table(
    index='Student ID Pseudonymised',
    columns='Course Code',
    values='Enrolled',
    aggfunc='max',
    fill_value=0
)
S = cosine_similarity(df_ui.values.T)
idx_map = {c: i for i, c in enumerate(df_ui.columns)}

# -------------------------------------------------------------------------
# 5. 读取并过滤关联规则
# -------------------------------------------------------------------------
RULES_PATH = os.path.join(os.path.dirname(__file__), "rules", "rules_ap.pkl")
with open(RULES_PATH, "rb") as f:
    rules_df = pickle.load(f)
rules_df['antecedents'] = rules_df['antecedents'].apply(set)
rules_df['consequents'] = rules_df['consequents'].apply(set)
allowed = set(course_meta_dict.keys())
rules_df = rules_df[
    rules_df['antecedents'].apply(lambda s: s.issubset(allowed)) &
    rules_df['consequents'].apply(lambda s: s.issubset(allowed))
].reset_index(drop=True)

# -------------------------------------------------------------------------
# 6. 推荐算法：AR、CF、Hybrid
# -------------------------------------------------------------------------
def cf_recommend(selected, S, idx_map, top_n=5):
    sel = set(selected)
    scores = {}
    for c, j in idx_map.items():
        if c in sel:
            continue
        scores[c] = sum(S[idx_map[s], j] for s in selected if s in idx_map)
    return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def recommend_hybrid(selected, rules_df, S, idx_map, num_rec=3, alpha=0.6):
    sel = set(selected)
    # AR raw
    ar_raw = {}
    for _, r in rules_df.iterrows():
        if r['antecedents'].issubset(sel):
            for c in r['consequents']:
                if c in sel:
                    continue
                ar_raw[c] = max(ar_raw.get(c, 0), r['confidence'])
    # CF raw
    cf_raw = {}
    for c, j in idx_map.items():
        if c in sel:
            continue
        cf_raw[c] = sum(S[idx_map[s], j] for s in selected if s in idx_map)
    # Min-max normalize
    def minmax(d):
        if not d:
            return {}
        lo, hi = min(d.values()), max(d.values())
        if lo == hi:
            return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}
    ar = minmax(ar_raw)
    cf = minmax(cf_raw)
    # Combine
    final = {c: alpha * ar.get(c, 0) + (1 - alpha) * cf.get(c, 0)
             for c in set(ar) | set(cf)}
    top = sorted(final.items(), key=lambda x: x[1], reverse=True)[:num_rec]
    res = []
    for c, sc in top:
        res.append({
            "code": c,
            "title": course_meta_dict.get(c, ""),
            "support": round(float(course_support.get(c, 0)), 4),
            "ar_score": round(ar.get(c, 0), 4) if c in ar else None,
            "cf_score": round(cf.get(c, 0), 4) if c in cf else None,
            "final_score": round(sc, 4)
        })
    return res

# -------------------------------------------------------------------------
# 7. 按年统计选课次数 (2022–2024)，Term Code 格式如 1241 → 年份取中间两位 + 2000
# -------------------------------------------------------------------------
df_all['Year'] = df_all['Term Code'].astype(str).str[1:3].astype(int) + 2000
yearly = df_all.groupby(['Year', 'Course Code']).size().unstack(fill_value=0)
year_course_counts = yearly.to_dict(orient='index')
years = sorted(year_course_counts.keys())

# -------------------------------------------------------------------------
# 8. 首页路由
# -------------------------------------------------------------------------
@app.route("/")
def index():
    logo_url = url_for('static', filename='img/logo.png')
    return render_template(
        "index.html",
        grading_forms=grading_forms,
        course_meta_dict=course_meta_dict,
        new_courses=NEW_COURSES,
        logo_url=logo_url,
        years=years,
        year_course_counts_json=json.dumps(year_course_counts)
    )

# -------------------------------------------------------------------------
# 9. AJAX 推荐接口
# -------------------------------------------------------------------------
@app.route("/recommend", methods=["POST"])
def ajax_recommend():
    try:
        data = request.get_json()
        sel = data.get("selected", [])
        num = data.get("num_rec", 3)
        mode = data.get("mode", "hybrid")

        # 热门 (support)
        if not sel or mode == "support":
            recs = []
            for c, s in top4_courses:
                if c in sel:
                    continue
                recs.append({
                    "code": c,
                    "title": course_meta_dict[c],
                    "support": round(float(s), 4),
                    "ar_score": None,
                    "cf_score": None,
                    "final_score": None
                })
            return jsonify({"recommendations": recs})

        if mode == "ar":
            return jsonify({"recommendations":
                recommend_hybrid(sel, rules_df, S, idx_map, num_rec=num, alpha=1.0)
            })

        if mode == "cf":
            codes = cf_recommend(sel, S, idx_map, top_n=num)
            recs = []
            for c in codes:
                raw = sum(S[idx_map[s], idx_map[c]] for s in sel if s in idx_map)
                recs.append({
                    "code": c,
                    "title": course_meta_dict[c],
                    "support": round(float(course_support.get(c, 0)), 4),
                    "ar_score": None,
                    "cf_score": round(raw, 4),
                    "final_score": None
                })
            return jsonify({"recommendations": recs})

        # hybrid
        return jsonify({"recommendations":
            recommend_hybrid(sel, rules_df, S, idx_map, num_rec=num, alpha=0.6)
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

# -------------------------------------------------------------------------
# 10. 启动
# -------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
