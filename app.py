import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client
from datetime import datetime, timedelta
import uuid
import extra_streamlit_components as stx
import os

# ===== 页面配置 =====
st.set_page_config(page_title="AI趋同度测试", layout="wide")
st.title("🧪 AI生成文案趋同度分析")
st.caption("现场实验：看看大模型是不是都在说一样的话")

# ===== 初始化Supabase客户端 =====
SUPABASE_URL = "https://znebmxrbjflnykotccma.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpuZWJteHJiamZsbnlrb3RjY21hIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg5MzcxNDYsImV4cCI6MjEwNDUxMzE0Nn0.3CYlfLv_WeliP48vFG108fCNLD-BIhwINj25nkMElqo"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== 设备唯一ID（Cookie 持久化，刷新不丢） =====
cookie_manager = stx.CookieManager(key="ai_sim_cookies")
DEVICE_COOKIE_NAME = "ai_sim_device_id"

def get_device_id():
    # 1) 优先从 cookie 读取（刷新/重开浏览器也能识别同一台设备）
    device_id = cookie_manager.get(DEVICE_COOKIE_NAME)
    if device_id:
        return device_id
    # 2) 本次会话内已生成过
    if st.session_state.get("__device_id"):
        return st.session_state["__device_id"]
    # 3) 新设备：生成 UUID 并写入 cookie + session_state
    new_id = str(uuid.uuid4())
    st.session_state["__device_id"] = new_id
    try:
        cookie_manager.set(
            DEVICE_COOKIE_NAME,
            new_id,
            expires_at=datetime.now() + timedelta(days=365),
            key="set_device_cookie",
        )
    except Exception:
        pass
    return new_id

device_id = get_device_id()

# ===== 数据读写 =====
def get_all_submissions():
    response = supabase.table("submissions").select("*").order("created_at").execute()
    return response.data

def add_submission(name, text, device_id):
    supabase.table("submissions").insert({
        "name": name,
        "text": text,
        "device_id": device_id,
    }).execute()

def name_exists(name):
    """检查该昵称是否已经提交过"""
    response = supabase.table("submissions").select("*").eq("name", name).execute()
    return len(response.data) > 0

def device_exists(dev_id):
    """检查该设备是否已经提交过"""
    if not dev_id:
        return False
    response = supabase.table("submissions").select("*").eq("device_id", dev_id).execute()
    return len(response.data) > 0

# ===== 左侧：提交区 =====
with st.sidebar:
    st.header("📝 提交你的文案")

    all_data = get_all_submissions()
    st.metric("📊 已提交文案数", len(all_data))

    # 双重检查：session_state 或 数据库里已有本设备记录
    already_submitted = st.session_state.get("submitted") or device_exists(device_id)

    if already_submitted:
        st.success("✅ 你已经提交过了，每台设备限提交一次～")
    else:
        with st.form("submit_form"):
            name = st.text_input("你的昵称", help="用于限制每人只提交一次，请勿与他人重复")
            text = st.text_area("粘贴AI生成的文案", height=150)
            submitted = st.form_submit_button("🚀 提交")

            if submitted:
                name = (name or "").strip()
                text = (text or "").strip()
                if not name:
                    st.error("❌ 请填写昵称（用于限制每人提交一次）")
                elif not text:
                    st.error("❌ 请粘贴你的文案")
                elif device_exists(device_id):
                    st.error("❌ 本设备已提交过，每台设备限提交一次")
                elif name_exists(name):
                    st.error(f"❌ 昵称「{name}」已被使用，请换一个")
                else:
                    add_submission(name, text, device_id)
                    st.session_state["submitted"] = True
                    st.rerun()

    st.divider()

    if st.button("🔄 刷新数据"):
        st.rerun()

# ===== 主区域：分析报告 =====
all_data = get_all_submissions()

if len(all_data) < 2:
    st.info("💡 等待更多文案提交中...（至少需要2份才能分析）")
    st.stop()

# 提取数据
texts = [item["text"] for item in all_data]
names = [item["name"] or f"匿名{i+1}" for i, item in enumerate(all_data)]

# ===== 分词 =====
def cut_text(text):
    return " ".join(jieba.cut(text))

tokenized = [cut_text(t) for t in texts]
n = len(tokenized)

# ===== 相似度算法：TF-IDF bigram 余弦 + Jaccard 词集加权平均 =====
# 1) TF-IDF：加入词级 bigram，压制高频虚词
vectorizer = TfidfVectorizer(
    tokenizer=cut_text,
    token_pattern=None,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
)
tfidf_matrix = vectorizer.fit_transform(texts)
cos_sim = cosine_similarity(tfidf_matrix)

# 2) Jaccard：按去重词集计算
def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0

jac_sim = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        s = jaccard(tokenized[i], tokenized[j])
        jac_sim[i, j] = jac_sim[j, i] = s
np.fill_diagonal(jac_sim, 1.0)

# 3) 加权平均：0.5 余弦 + 0.5 Jaccard（相似度回落到 60-70% 的自然区间）
sim_matrix = 0.5 * cos_sim + 0.5 * jac_sim

# 全部两两组合
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((i, j, float(sim_matrix[i][j])))
pairs.sort(key=lambda x: x[2], reverse=True)

# 每位参与者对全体的平均相似度
person_avg = (sim_matrix.sum(axis=1) - 1) / (n - 1)

# ===== Matplotlib中文字体设置 =====
import matplotlib.font_manager as fm
current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, 'simhei.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 展示Tab =====
tab1, tab2, tab3, tab4 = st.tabs(["📊 相似度总览", "🏆 趋同排行榜", "☁️ 词云分析", "🔍 文案对比"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("相似度分布")
        all_sims = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        fig, ax = plt.subplots()
        ax.hist(all_sims, bins=20, color="steelblue", edgecolor="white")
        ax.set_xlabel("相似度")
        ax.set_ylabel("频次")
        ax.axvline(np.mean(all_sims), color="red", linestyle="--", label=f"均值: {np.mean(all_sims):.1%}")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("相似度热力图")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(sim_matrix, cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        plt.colorbar(im, ax=ax, label="相似度")
        st.pyplot(fig)

    st.divider()

    st.subheader(f"👥 全部参与者（共 {n} 人）平均趋同度")
    st.caption("平均相似度 = 该同学与其余所有人的相似度均值，越高说明其文案越'模板化'")
    person_df = pd.DataFrame({
        "昵称": names,
        "平均相似度": person_avg,
        "文案字数": [len(t) for t in texts],
    }).sort_values("平均相似度", ascending=False).reset_index(drop=True)
    person_df.insert(0, "排名", range(1, len(person_df) + 1))
    st.dataframe(
        person_df.style.format({"平均相似度": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("🏆 全部文案两两趋同排行")
    st.caption(f"共 {len(pairs)} 对组合，按相似度从高到低排列")

    min_sim = st.slider("筛选：只显示相似度 ≥", 0.0, 1.0, 0.0, 0.05)
    shown_pairs = [(i, j, s) for (i, j, s) in pairs if s >= min_sim]
    st.write(f"当前展示 **{len(shown_pairs)}** 对")

    for idx, (i, j, score) in enumerate(shown_pairs):
        with st.expander(f"#{idx+1}  {names[i]} vs {names[j]} —— 相似度 {score:.1%}"):
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"**{names[i]}**")
                st.write(texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i])
            with col2:
                st.caption(f"**{names[j]}**")
                st.write(texts[j][:200] + "..." if len(texts[j]) > 200 else texts[j])

with tab3:
    st.subheader("☁️ 所有文案的词云（高频词即AI的'话术模板'）")
    all_text = " ".join(texts)
    words = jieba.cut(all_text)
    words = [w for w in words if len(w) > 1]
    filtered_text = " ".join(words)

    wc = WordCloud(
        font_path="simhei.ttf" if os.path.exists("simhei.ttf") else None,
        background_color="white",
        width=800,
        height=400,
        max_words=100
    ).generate(filtered_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    st.caption("💡 这些高频词就是大模型生成文案时的'舒适区'——大家都在用同样的词汇排列组合")

with tab4:
    st.subheader("🔍 任选两篇文案对比")
    if len(pairs) > 0:
        options = [
            f"{names[a]} vs {names[b]}（相似度 {score:.1%}）"
            for a, b, score in pairs
        ]
        selected = st.selectbox("选择要对比的文案对", options, index=0)
        i, j, score = pairs[options.index(selected)]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"📄 {names[i]}")
            st.write(texts[i])
        with col2:
            st.subheader(f"📄 {names[j]}")
            st.write(texts[j])

        st.metric("📈 相似度", f"{score:.1%}")
        st.caption("🔴 两篇文案在措辞、结构、语气上高度趋同——这就是大模型的'广度'带来的同质化陷阱")
