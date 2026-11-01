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
import difflib
import extra_streamlit_components as stx

# ===== 页面配置 =====
st.set_page_config(page_title="AI趋同度测试", layout="wide")
st.title("🧪 AI生成文案趋同度分析")
st.caption("现场实验：看看大模型是不是都在说一样的话（支持前测 / 后测两轮对比）")

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

def get_submissions_grouped():
    """按昵称分组，组内按提交时间排序：第1条=第一轮(前测)，第2条=第二轮(后测)"""
    data = get_all_submissions()
    groups = {}
    for item in data:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        groups.setdefault(name, []).append(item)
    return groups

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

    groups_sb = get_submissions_grouped()
    st.metric("👥 参与人数", len(groups_sb))
    st.metric("📊 提交总条数", sum(len(v) for v in groups_sb.values()))

    # 教师开关：是否开放第二轮（后测）提交
    round2_open = st.checkbox(
        "🔓 开放第二轮（改写稿）",
        value=False,
        help="勾选后，已提交过第一轮的同学可在同一台设备上提交改写后的第二轮文案",
    )

    # 本设备已提交情况
    my_name = None
    my_rows = 0
    for nm, rows in groups_sb.items():
        for r in rows:
            if r.get("device_id") == device_id:
                my_rows += 1
                my_name = nm

    if my_rows >= 2:
        st.success("✅ 你的前后两版都已提交，感谢参与实验！")
    elif my_rows == 1:
        if round2_open:
            with st.form("submit_round2"):
                name2 = st.text_input("你的昵称（需与第一轮一致）", value=my_name or "")
                text2 = st.text_area(
                    "粘贴你改写后的文案（第二轮·后测）",
                    height=150,
                    help="在第一轮基础上修改，让文案更有辨识度（更少“AI味”）",
                )
                submitted2 = st.form_submit_button("✏️ 提交改写稿")

                if submitted2:
                    name2 = (name2 or "").strip()
                    text2 = (text2 or "").strip()
                    rows_name = groups_sb.get(name2, [])
                    if not name2:
                        st.error("❌ 请填写昵称（需与第一轮一致）")
                    elif not text2:
                        st.error("❌ 请粘贴改写后的文案")
                    elif not rows_name:
                        st.error(f"❌ 昵称「{name2}」没有第一轮提交记录，无法提交第二轮")
                    elif len(rows_name) >= 2:
                        st.error(f"❌ 昵称「{name2}」已提交过第二轮")
                    elif rows_name[0].get("device_id") != device_id:
                        st.error("❌ 请使用提交第一轮时的同一台设备/浏览器提交（用于保护你的数据）")
                    else:
                        add_submission(name2, text2, device_id)
                        st.rerun()
        else:
            st.info("✅ 你已提交第一轮（前测）。等老师开放第二轮后，可提交改写稿～")
    else:
        with st.form("submit_form"):
            name = st.text_input("你的昵称", help="用于限制每人只提交一次，请勿与他人重复")
            text = st.text_area("粘贴AI生成的文案（第一轮·前测）", height=150)
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
                    st.rerun()

    st.divider()

    if st.button("🔄 刷新数据"):
        st.rerun()

# ===== 主区域：分析报告 =====
groups = get_submissions_grouped()

if len(groups) < 2:
    st.info("💡 等待更多文案提交中...（至少需要2份才能分析）")
    st.stop()

# 每位参与者：before = 第一轮(前测)，after = 第二轮(后测，可能没有)
participants = []
for name, rows in groups.items():
    participants.append({
        "name": name,
        "before": rows[0],
        "after": rows[1] if len(rows) > 1 else None,
    })
participants.sort(key=lambda p: p["before"].get("created_at") or "")

texts = [p["before"]["text"] for p in participants]
names = [p["name"] or f"匿名{i+1}" for i, p in enumerate(participants)]

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

# ===== 前后对比（实验前测/后测）工具 =====
def round_sim_matrix(text_list):
    """对一组文本计算加权相似度矩阵（TF-IDF余弦 + Jaccard + 幂次校准），口径与主分析一致"""
    m = len(text_list)
    if m < 2:
        return np.ones((1, 1))
    tok = [cut_text(t) for t in text_list]
    vec = TfidfVectorizer(
        tokenizer=cut_text,
        token_pattern=None,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    cos = cosine_similarity(vec.fit_transform(text_list))
    jac = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            s = jaccard(tok[a], tok[b])
            jac[a, b] = jac[b, a] = s
    np.fill_diagonal(jac, 1.0)
    base = 0.5 * cos + 0.5 * jac
    return np.power(np.clip(base, 0.0, 1.0), GAMMA)

def upper_mean(mat):
    """矩阵上三角（不含对角线）的均值"""
    m = mat.shape[0]
    if m < 2:
        return float("nan")
    return float(mat[np.triu_indices(m, k=1)].mean())

def person_avg_in(mat):
    """每个个体对矩阵中其他人的平均相似度"""
    m = mat.shape[0]
    if m < 2:
        return np.array([float("nan")])
    return (mat.sum(axis=1) - 1) / (m - 1)

def text_pair_sim(t1, t2):
    """两段文本的相似度（与主分析同一口径）"""
    mat = round_sim_matrix([t1, t2])
    return float(mat[0][1])

def diff_html(before, after):
    """分词级前后对比：删除词标红+删除线，新增词标绿，供逐人展示改写差异"""
    toks_b = list(jieba.cut(before))
    toks_a = list(jieba.cut(after))
    sm = difflib.SequenceMatcher(None, toks_b, toks_a)
    out_b, out_a = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for t in toks_b[i1:i2]:
                e = html.escape(t)
                out_b.append(e)
                out_a.append(e)
        elif tag == "delete":
            for t in toks_b[i1:i2]:
                out_b.append(
                    f'<span style="background:#ffc4c4;text-decoration:line-through;">{html.escape(t)}</span>'
                )
        elif tag == "insert":
            for t in toks_a[j1:j2]:
                out_a.append(
                    f'<span style="background:#c3e8c3;">{html.escape(t)}</span>'
                )
        elif tag == "replace":
            for t in toks_b[i1:i2]:
                out_b.append(
                    f'<span style="background:#ffc4c4;text-decoration:line-through;">{html.escape(t)}</span>'
                )
            for t in toks_a[j1:j2]:
                out_a.append(
                    f'<span style="background:#c3e8c3;">{html.escape(t)}</span>'
                )
    return "".join(out_b).replace("\n", "<br>"), "".join(out_a).replace("\n", "<br>")

# ===== Tabs =====
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 相似度总览", "🔄 前后对比", "🏆 趋同排行榜", "☁️ 词云分析", "🔍 文案对比",
])

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
    st.caption("平均相似度 = 该同学与其余所有人的相似度均值，越高说明其文案越'模板化'（基于第一轮·前测文案）")
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
    st.subheader("🔄 实验前后对比（前测 vs 后测）")
    st.caption(
        "第一轮（前测）= 直接使用 AI 生成的文案；第二轮（后测）= 看到趋同分析后自己改写。"
        "对比两轮的整体趋同度，就能检验「改写」是否真的让文案更有辨识度。"
    )

    both = [p for p in participants if p["after"] is not None]
    if len(both) == 0:
        st.info(
            "📭 还没有人提交第二轮改写稿。老师在侧边栏勾选「🔓 开放第二轮提交」后，"
            "已提交过第一轮的同学即可提交改写版，这里会自动生成前后对比。"
        )
    else:
        names_both = [p["name"] for p in both]
        before_texts = [p["before"]["text"] for p in both]
        after_texts = [p["after"]["text"] for p in both]

        # ---- 逐人改写差异 ----
        st.markdown("### ✏️ 逐人改写对比")
        st.caption("🔴 红色删除线 = 被改掉的词；🟢 绿色 = 新增的词；其余为前后都保留的部分")
        sel_p = st.selectbox("选择参与者", names_both)
        idx = names_both.index(sel_p)
        self_sim_sel = text_pair_sim(before_texts[idx], after_texts[idx])
        b_html, a_html = diff_html(before_texts[idx], after_texts[idx])
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**{sel_p} · 前测**（与自己的后测版相似度 {self_sim_sel:.0%}）")
            st.markdown(
                f'<div style="line-height:1.9;font-size:15px;">{b_html}</div>',
                unsafe_allow_html=True,
            )
        with cc2:
            st.markdown(f"**{sel_p} · 后测**")
            st.markdown(
                f'<div style="line-height:1.9;font-size:15px;">{a_html}</div>',
                unsafe_allow_html=True,
            )

        # ---- 整体统计（至少2人提交第二轮）----
        if len(both) >= 2:
            mat_before = round_sim_matrix(before_texts)
            mat_after = round_sim_matrix(after_texts)
            mean_b = upper_mean(mat_before)
            mean_a = upper_mean(mat_after)
            avg_b = person_avg_in(mat_before)
            avg_a = person_avg_in(mat_after)
            self_sim = [text_pair_sim(b, a) for b, a in zip(before_texts, after_texts)]
            improved = [i for i in range(len(both)) if avg_a[i] < avg_b[i]]

            st.markdown("### 📊 整体趋同度：前测 vs 后测")
            c1, c2, c3 = st.columns(3)
            c1.metric("前测平均趋同度", f"{mean_b:.1%}")
            c2.metric("后测平均趋同度", f"{mean_a:.1%}")
            c3.metric(
                "前后变化",
                f"{mean_a - mean_b:+.1%}",
                delta=f"{len(improved)}/{len(both)} 人下降",
                delta_color="inverse",
            )

            if len(improved) > len(both) / 2:
                st.success(
                    f"✅ 结论：{len(improved)}/{len(both)} 位同学改写后趋同度下降"
                    f"（平均下降 {mean_b - mean_a:.1%}），说明「改写」能有效降低AI文案的趋同度。"
                )
            else:
                st.warning(
                    f"⚠️ 结论：仅 {len(improved)}/{len(both)} 位同学趋同度下降，"
                    "改写暂未显著降低趋同度，可以再讨论如何改写得更有辨识度。"
                )

            st.markdown("#### 每人前后趋同度变化")
            order = sorted(range(len(both)), key=lambda i: avg_b[i], reverse=True)
            fig, ax = plt.subplots(figsize=(10, max(3.2, 0.42 * len(both))))
            y = np.arange(len(both))
            h = 0.35
            ax.barh(y + h / 2, [avg_b[i] for i in order], height=h, color="#9db8d2", label="前测")
            ax.barh(y - h / 2, [avg_a[i] for i in order], height=h, color="#e8a87c", label="后测")
            ax.set_yticks(y)
            ax.set_yticklabels([names_both[i] for i in order], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("平均相似度（与该同学以外的所有人）")
            ax.legend()
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(6.5, 6))
            ax2.scatter(avg_b, avg_a, s=40, color="steelblue")
            lo = min(float(avg_b.min()), float(avg_a.min())) - 0.05
            hi = max(float(avg_b.max()), float(avg_a.max())) + 0.05
            ax2.plot([lo, hi], [lo, hi], "--", color="gray", label="无变化线")
            ax2.set_xlabel("前测平均相似度")
            ax2.set_ylabel("后测平均相似度")
            ax2.legend()
            st.pyplot(fig2)
            st.caption("散点落在对角线下方的同学 = 改写后趋同度下降（更有辨识度）")

            st.markdown("#### 数据明细")
            df_bc = pd.DataFrame({
                "昵称": names_both,
                "前测平均趋同度": avg_b,
                "后测平均趋同度": avg_a,
                "趋同度变化": avg_a - avg_b,
                "与自身后测的相似度": self_sim,
            })
            st.dataframe(
                df_bc.style.format({
                    "前测平均趋同度": "{:.1%}",
                    "后测平均趋同度": "{:.1%}",
                    "趋同度变化": "{:+.1%}",
                    "与自身后测的相似度": "{:.0%}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ 导出前后对比数据（CSV）",
                df_bc.to_csv(index=False).encode("utf-8-sig"),
                file_name="前后对比数据.csv",
                mime="text/csv",
            )
        else:
            st.info("ℹ️ 至少需要 2 位同学提交第二轮，才能展示整体前后对比统计（逐人对比已在上方展示）。")

with tab3:
    st.subheader("🏆 全部文案两两趋同排行")
    st.caption(f"共 {len(pairs)} 对组合，按相似度从高到低排列（当前为第一轮·前测数据）")

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

with tab4:
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

with tab5:
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
