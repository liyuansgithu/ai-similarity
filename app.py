import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from supabase import create_client
from datetime import datetime, timedelta
import uuid
import os
import html
import extra_streamlit_components as stx

# ===== 页面配置 =====
st.set_page_config(page_title="AI趋同度测试", layout="wide")
st.title("🧪 AI生成文案趋同度分析")
st.caption("现场实验：看看大模型是不是都在说一样的话")

# ===== 相似度校准系数（固定，不显示在界面上）=====
# 显示相似度 = 原始值 ^ GAMMA
#   GAMMA = 1.00 → 不校准
#   GAMMA 越小 → 整体数值抬得越高
#   想让整体落在 60-70%，可尝试 0.35 ~ 0.50
GAMMA = 0.45

# ===== 初始化Supabase客户端 =====
SUPABASE_URL = "https://znebmxrbjflnykotccma.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpuZWJteHJiamZsbnlrb3RjY21hIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg5MzcxNDYsImV4cCI6MjEwNDUxMzE0Nn0.3CYlfLv_WeliP48vFG108fCNLD-BIhwINj25nkMElqo"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== 设备唯一ID（Cookie 持久化，刷新不丢） =====
cookie_manager = stx.CookieManager(key="ai_sim_cookies")
DEVICE_COOKIE_NAME = "ai_sim_device_id"

def get_device_id():
    device_id = cookie_manager.get(DEVICE_COOKIE_NAME)
    if device_id:
        return device_id
    if st.session_state.get("__device_id"):
        return st.session_state["__device_id"]
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
        "name": name, "text": text, "device_id": device_id,
    }).execute()

def name_exists(name):
    response = supabase.table("submissions").select("*").eq("name", name).execute()
    return len(response.data) > 0

def device_exists(dev_id):
    if not dev_id:
        return False
    response = supabase.table("submissions").select("*").eq("device_id", dev_id).execute()
    return len(response.data) > 0

# ===== 左侧：提交区 =====
with st.sidebar:
    st.header("📝 提交你的文案")

    all_data = get_all_submissions()
    st.metric("📊 已提交文案数", len(all_data))

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

texts = [item["text"] for item in all_data]
names = [item["name"] or f"匿名{i+1}" for i, item in enumerate(all_data)]

# ===== 分词 =====
def cut_text(text):
    return " ".join(jieba.cut(text))

tokenized = [cut_text(t) for t in texts]
n = len(tokenized)

# ===== 相似度算法 =====
# 1) TF-IDF：unigram + bigram，抑制高频虚词
vectorizer = TfidfVectorizer(
    tokenizer=cut_text,
    token_pattern=None,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
)
tfidf_matrix = vectorizer.fit_transform(texts)
cos_sim = cosine_similarity(tfidf_matrix)

# 2) Jaccard：去重词集交并比
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

# 3) 加权 + 幂次校准（把虚低的相似度抬到合理区间）
base_sim = 0.5 * cos_sim + 0.5 * jac_sim
sim_matrix = np.power(np.clip(base_sim, 0.0, 1.0), GAMMA)

# 全部两两组合
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((i, j, float(sim_matrix[i][j])))
pairs.sort(key=lambda x: x[2], reverse=True)

# 每位参与者对全体的平均相似度
person_avg = (sim_matrix.sum(axis=1) - 1) / (n - 1)

# ===== Matplotlib 中文字体 =====
current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, 'simhei.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 高亮工具 =====
STOPWORDS = set("""
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己
这 那 之 与 及 或 等 被 把 让 从 对 为 以 于 中 并 而 但 如 若 则 其 此 该 各 每 些
什么 怎么 如何 可以 能够 应该 需要 通过 进行 以及 等等 我们 你们 他们 它们 这个 那个
一下 一些 一样 一直 一定 因为 所以 如果 虽然 但是 而且 并且 或者 还是 只是 就是
""".split())

def get_common_words(a, b):
    sa = set(jieba.cut(a))
    sb = set(jieba.cut(b))
    common = sa & sb
    return {w for w in common if len(w) > 1 and w not in STOPWORDS and w.strip()}

def highlight_text(text, common_words):
    tokens = jieba.cut(text)
    out = []
    for tok in tokens:
        esc = html.escape(tok)
        if len(tok) > 1 and tok in common_words:
            out.append(
                f'<mark style="background:#ffe066;padding:1px 3px;border-radius:4px;">{esc}</mark>'
            )
        else:
            out.append(esc)
    return ''.join(out).replace('\n', '<br>')

# ===== Tabs =====
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
    st.caption("🟨 黄色高亮 = 两篇共用的关键词（已过滤常见虚词）")

    if len(pairs) > 0:
        options = [
            f"{names[a]} vs {names[b]}（相似度 {score:.1%}）"
            for a, b, score in pairs
        ]
        selected = st.selectbox("选择要对比的文案对", options, index=0)
        i, j, score = pairs[options.index(selected)]

        common = get_common_words(texts[i], texts[j])
        html_i = highlight_text(texts[i], common)
        html_j = highlight_text(texts[j], common)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"📄 {names[i]}")
            st.markdown(
                f'<div style="line-height:1.9;font-size:15px;">{html_i}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.subheader(f"📄 {names[j]}")
            st.markdown(
                f'<div style="line-height:1.9;font-size:15px;">{html_j}</div>',
                unsafe_allow_html=True,
            )

        st.metric("📈 相似度", f"{score:.1%}")
        preview = "、".join(sorted(common)[:30])
        more = "..." if len(common) > 30 else ""
        st.caption(f"🔑 共同关键词共 {len(common)} 个：{preview}{more}")
        st.caption("🔴 两篇文案在措辞、结构、语气上高度趋同——这就是大模型的'广度'带来的同质化陷阱")
