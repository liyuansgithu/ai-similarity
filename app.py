import streamlit as st
import streamlit_cookies_manager as stx
import pandas as pd
import numpy as np
import jieba
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from supabase import create_client, Client
import httpx

GAMMA = 0.45

# ===================== Supabase 硬编码（和你现在保持一致） =====================
SUPABASE_URL = "https://znebmxrbjflnykotccma.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpuzWJteHJiamZsbnlrb3RjY21hIiwiYXpwIjoiYW5vbiIsImlhdCI6MTczODg5M2cxNDY0NCwiZXhwIjoxOTUxMzE0NzY0fQ.3CYIfLv_WeliP48vFG108fCNLD-BlhwlNj25nkMElqo"
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
    http_client=httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0))
)

# ===================== Cookie 设备ID管理（沿用你写的逻辑） =====================
cookie_manager = stx.CookieManager(key="ai_sim_cookies")
DEVICE_COOKIE_NAME = "ai_sim_device_id"

def get_device_id():
    device_id = cookie_manager.get(DEVICE_COOKIE_NAME)
    if device_id:
        return device_id
    if st.session_state.get("_device_id"):
        return st.session_state["_device_id"]
    new_id = str(uuid.uuid4())
    cookie_manager[DEVICE_COOKIE_NAME] = new_id
    cookie_manager.save()
    st.session_state["_device_id"] = new_id
    return new_id

device_id = get_device_id()
cookie_manager.save()

# ===================== 工具函数 =====================
def calc_tfidf_cosine(text_list):
    def tokenize(text):
        return jieba.lcut(text)
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    tfidf_mat = vectorizer.fit_transform(text_list)
    sim_mat = cosine_similarity(tfidf_mat)
    return sim_mat

def text_diff_html(old_text, new_text):
    words_old = set(jieba.lcut(old_text))
    words_new = set(jieba.lcut(new_text))
    all_words = jieba.lcut(new_text)
    html_parts = []
    for w in all_words:
        if w in words_old:
            html_parts.append(w)
        else:
            html_parts.append(f"<span style='color:green;'>{w}</span>")
    old_words_list = jieba.lcut(old_text)
    del_parts = []
    for w in old_words_list:
        if w not in words_new:
            del_parts.append(f"<span style='color:red;text-decoration:line-through;'>{w}</span>")
        else:
            del_parts.append(w)
    return "".join(html_parts), "".join(del_parts)

# ===================== 加载数据 =====================
@st.cache_data(ttl=10)
def load_all_data():
    res = supabase.table("submissions").select("*").execute()
    df = pd.DataFrame(res.data)
    return df

df_raw = load_all_data()
this_dev_data = df_raw[df_raw["device_id"] == device_id]
submitted_round1 = len(this_dev_data[this_dev_data["round"] == 1]) > 0
submitted_round2 = len(this_dev_data[this_dev_data["round"] == 2]) > 0

# ===================== 侧边栏 =====================
st.sidebar.header("提交信息")
nickname = st.sidebar.text_input("填写昵称", value="")
st.sidebar.markdown(f"你的设备编号：`{device_id}`")
st.sidebar.info("规则：本设备第一次提交=AI原稿；提交原稿后，自动开启改写稿提交\n⚠️ 不要开无痕模式！")

# ===================== 标签页 =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊总览", "🏆趋同排行榜", "☁️词云", "📄两两对比", "🔄前后对比"])

st.header("AI文案趋同测试提交")
if not submitted_round1:
    st.subheader("第一轮：提交AI直接生成的原始文稿")
    content = st.text_area("粘贴AI原文", height=200)
    if st.button("提交第一轮文稿"):
        if not nickname or not content.strip():
            st.warning("昵称和文本不能为空！")
        else:
            supabase.table("submissions").insert({
                "device_id": device_id,
                "nickname": nickname,
                "content": content,
                "round": 1
            }).execute()
            st.cache_data.clear()
            st.success("✅第一轮提交成功！刷新页面，你现在可以提交改写后的第二轮文稿")
            st.rerun()
elif not submitted_round2:
    st.subheader("第二轮：提交你改写润色后的文稿")
    content = st.text_area("粘贴改写后的文稿", height=200)
    if st.button("提交第二轮改写稿"):
        if not nickname or not content.strip():
            st.warning("昵称和文本不能为空！")
        else:
            supabase.table("submissions").insert({
                "device_id": device_id,
                "nickname": nickname,
                "content": content,
                "round": 2
            }).execute()
            st.cache_data.clear()
            st.success("✅第二轮提交完成！本设备不再支持提交")
            st.rerun()
else:
    st.info("✅本设备已经完成两轮提交，无需重复提交，可以查看下方分析结果")

# tab1 总览
with tab1:
    st.subheader("全班文稿趋同总览（统计第一轮原稿）")
    df_round1 = df_raw[df_raw["round"] == 1].reset_index(drop=True)
    if len(df_round1) < 2:
        st.info("第一轮数据不足，至少2份文稿才可以计算相似度")
    else:
        text_list = df_round1["content"].tolist()
        sim_matrix = calc_tfidf_cosine(text_list)
        avg_sim = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
        st.metric("第一轮平均两两相似度", f"{avg_sim:.3f}")
        fig, ax = plt.subplots(figsize=(7,6))
        im = ax.imshow(sim_matrix, cmap="Blues")
        ax.set_xticks(range(len(df_round1)))
        ax.set_yticks(range(len(df_round1)))
        ax.set_xticklabels(df_round1["nickname"], rotation=45, ha="right")
        ax.set_yticklabels(df_round1["nickname"])
        fig.colorbar(im, ax=ax, label="余弦相似度")
        st.pyplot(fig)

# tab2 排行榜
with tab2:
    st.subheader("趋同排行榜（第一轮原稿）")
    df_round1 = df_raw[df_raw["round"] == 1].reset_index(drop=True)
    if len(df_round1) < 2:
        st.info("数据不足")
    else:
        text_list = df_round1["content"].tolist()
        sim_matrix = calc_tfidf_cosine(text_list)
        row_avg = np.sum(sim_matrix, axis=1)
        rank_df = pd.DataFrame({
            "nickname": df_round1["nickname"],
            "平均相似度": row_avg
        }).sort_values("平均相似度", ascending=False)
        st.dataframe(rank_df, use_container_width=True)

# tab3 词云
with tab3:
    st.subheader("第一轮文稿词云")
    df_round1 = df_raw[df_raw["round"] == 1]
    all_text = " ".join(df_round1["content"].astype(str))
    words = jieba.lcut(all_text)
    try:
        wc = WordCloud(font_path="simhei.ttf", background_color="white", width=800, height=400).generate(" ".join(words))
    except:
        wc = WordCloud(background_color="white", width=800, height=400).generate(" ".join(words))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

# tab4 两两对比
with tab4:
    st.subheader("第一轮文稿两两对比")
    df_round1 = df_raw[df_raw["round"] == 1].reset_index(drop=True)
    if len(df_round1) <2:
        st.info("数据不足")
    else:
        n1 = st.selectbox("选择文稿A", df_round1["nickname"])
        n2 = st.selectbox("选择文稿B", df_round1["nickname"], index=1)
        t1 = df_round1[df_round1["nickname"] == n1]["content"].iloc[0]
        t2 = df_round1[df_round1["nickname"] == n2]["content"].iloc[0]
        vec = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None)
        tf = vec.fit_transform([t1,t2])
        sim = cosine_similarity(tf[0:1], tf[1:2])[0][0]
        st.write(f"相似度：{sim:.3f}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{n1}**")
            st.write(t1)
        with col2:
            st.markdown(f"**{n2}**")
            st.write(t2)

# tab5 前后对比
with tab5:
    st.subheader("🔄 前测（原稿） vs 后测（改写稿）对比分析")
    df1 = df_raw[df_raw["round"] == 1].rename(columns={"content":"content_r1", "nickname":"nick_r1"})[["device_id", "nick_r1", "content_r1"]]
    df2 = df_raw[df_raw["round"] == 2].rename(columns={"content":"content_r2", "nickname":"nick_r2"})[["device_id", "nick_r2", "content_r2"]]
    pair_df = pd.merge(df1, df2, on="device_id", how="inner")
    if len(pair_df) == 0:
        st.info("还没有设备同时完成两轮提交，暂无对比数据")
    else:
        sim_list_r1 = []
        sim_list_r2 = []
        for idx,row in pair_df.iterrows():
            all_r1_texts = df_raw[df_raw["round"] ==1]["content"].tolist()
            s1 = calc_tfidf_cosine([row["content_r1"]] + all_r1_texts)[0,1:].mean()
            s2 = calc_tfidf_cosine([row["content_r2"]] + all_r1_texts)[0,1:].mean()
            sim_list_r1.append(s1)
            sim_list_r2.append(s2)
        pair_df["sim_r1"] = sim_list_r1
        pair_df["sim_r2"] = sim_list_r2
        pair_df["delta"] = pair_df["sim_r2"] - pair_df["sim_r1"]
        pair_df["improved"] = pair_df["delta"] < 0

        avg_r1 = np.mean(pair_df["sim_r1"])
        avg_r2 = np.mean(pair_df["sim_r2"])
        improve_cnt = pair_df["improved"].sum()
        st.metric("原稿平均趋同度", f"{avg_r1:.3f}")
        st.metric("改写后平均趋同度", f"{avg_r2:.3f}", delta=f"{avg_r2-avg_r1:.3f}")
        st.write(f"改写后趋同度下降的人数：{improve_cnt}/{len(pair_df)}")

        fig, ax = plt.subplots()
        ax.scatter(pair_df["sim_r1"], pair_df["sim_r2"])
        ax.plot([0,1], [0,1], "r--", label="基准线")
        ax.set_xlabel("原稿趋同度")
        ax.set_ylabel("改写后趋同度")
        ax.set_title("前后趋同度散点图（点在红线下方=改写有效）")
        ax.legend()
        st.pyplot(fig)

        dev_sel = st.selectbox("选择设备查看文本差异", pair_df["device_id"])
        sel = pair_df[pair_df["device_id"] == dev_sel].iloc[0]
        html_r2, html_r1_del = text_diff_html(sel["content_r1"], sel["content_r2"])
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"**原稿（{sel['nick_r1']}）**")
            st.markdown(html_r1_del, unsafe_allow_html=True)
        with c2:
            st.markdown(f"**改写稿（{sel['nick_r2']}）**")
            st.markdown(html_r2, unsafe_allow_html=True)
        st.write(f"原稿趋同度：{sel['sim_r1']:.3f} | 改写后趋同度：{sel['sim_r2']:.3f} | 变化：{sel['delta']:.3f}")

        buf = io.StringIO()
        pair_df.to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button("下载对比数据表CSV", buf.getvalue(), "compare_result.csv")
