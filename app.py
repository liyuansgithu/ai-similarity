# app.py - 完整版智能体代码
import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# ===== 页面配置 =====
st.set_page_config(page_title="AI趋同度测试", layout="wide")
st.title("🧪 AI生成文案趋同度分析")
st.caption("现场实验：看看大模型是不是都在说一样的话")

# ===== 初始化数据存储 =====
if "submissions" not in st.session_state:
    st.session_state.submissions = []  # 存储 {"name": str, "text": str}

# ===== 左侧：提交区 =====
with st.sidebar:
    st.header("📝 提交你的文案")
    with st.form("submit_form"):
        name = st.text_input("你的昵称")
        text = st.text_area("粘贴AI生成的文案", height=150)
        submitted = st.form_submit_button("🚀 提交")
        if submitted and text:
            st.session_state.submissions.append({"name": name or "匿名", "text": text})
            st.success(f"✅ 已收录！当前共 {len(st.session_state.submissions)} 份文案")
            st.rerun()
    
    st.divider()
    st.metric("📊 已提交文案数", len(st.session_state.submissions))

# ===== 主区域：分析报告 =====
if len(st.session_state.submissions) < 2:
    st.info("💡 等待更多文案提交中...（至少需要2份才能分析）")
    st.stop()

# 提取数据
texts = [s["text"] for s in st.session_state.submissions]
names = [s["name"] or f"匿名{i+1}" for i, s in enumerate(st.session_state.submissions)]

# ---- 中文分词 ----
def cut_text(text):
    return " ".join(jieba.cut(text))

# ---- 计算相似度矩阵 ----
vectorizer = TfidfVectorizer(tokenizer=cut_text, token_pattern=None)
tfidf_matrix = vectorizer.fit_transform(texts)
sim_matrix = cosine_similarity(tfidf_matrix)

# 获取相似度最高的5对
pairs = []
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        pairs.append((i, j, sim_matrix[i][j]))
pairs.sort(key=lambda x: x[2], reverse=True)
top_pairs = pairs[:5]

# ===== Tab布局 =====
tab1, tab2, tab3, tab4 = st.tabs(["📊 相似度总览", "🏆 趋同排行榜", "☁️ 词云分析", "🔍 文案对比"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("相似度分布")
        # 提取所有非对角线相似度
        all_sims = [sim_matrix[i][j] for i in range(len(texts)) for j in range(i+1, len(texts))]
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
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        plt.colorbar(im, ax=ax, label="相似度")
        st.pyplot(fig)

with tab2:
    st.subheader("🏆 最相似的5对文案")
    for idx, (i, j, score) in enumerate(top_pairs):
        with st.expander(f"#{idx+1} 相似度: {score:.1%}"):
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
    # 分词并过滤停用词（可自定义）
    words = jieba.cut(all_text)
    # 简单过滤：长度>1的词
    words = [w for w in words if len(w) > 1]
    filtered_text = " ".join(words)
    
    wc = WordCloud(
        font_path="simhei.ttf",  # 确保有这个字体文件
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
    if len(top_pairs) > 0:
        # 默认选相似度最高的一对
        options = [f"第{i+1}对（{names[a]} vs {names[b]}，相似度{score:.1%}）" 
                   for i, (a, b, score) in enumerate(top_pairs)]
        selected = st.selectbox("选择要对比的文案对", options, index=0)
        idx = options.index(selected)
        i, j, score = top_pairs[idx]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"📄 {names[i]}")
            st.write(texts[i])
        with col2:
            st.subheader(f"📄 {names[j]}")
            st.write(texts[j])
        
        st.metric("📈 相似度", f"{score:.1%}")
        st.caption("🔴 两篇文案在措辞、结构、语气上高度趋同——这就是大模型的'广度'带来的同质化陷阱")