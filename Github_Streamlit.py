import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from google import genai
import textwrap
import ast

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="CinemaPulse Dashboard",
    page_icon="🎬",
    layout="wide"
)

# 定义一个获取 Key 的函数
def get_api_key():
    # 1. 检测用户输入
    if 'user_api_key' in st.session_state and st.session_state.user_api_key:
        return st.session_state.user_api_key
    
    # 2.获取我内定的key
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    
    # 3. 都没有
    return None

# --- CSS 页面视觉设计 ---
st.markdown("""
    <style>
    /* 按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
    }
    
    /*Tab 字体*/
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.3rem; /* 字体加大 */
        font-weight: 700;  /* 加粗 */
        padding-top: 5px;
        padding-bottom: 5px;
    }
    
    /* 情感列表样式 (Tags风格) */
    .sentiment-list { margin: 0; padding-left: 0; list-style-type: none; }
    .sentiment-list li { 
        display: inline-block;
        background: rgba(255,255,255,0.6);
        padding: 4px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .positive-box {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        color: #064e3b !important;
        margin-bottom: 10px;
    }
    .negative-box {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        color: #7f1d1d !important;
        margin-bottom: 10px;
    }

    /* AI 报告样式 */
    .summary-box, .details-box {
        padding: 25px;
        border-radius: 12px;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 17px !important;
        line-height: 1.6 !important;
    }
    .summary-box h3, .details-box h3 {
        color: #ffffff !important;
        font-size: 1.4rem;
        font-weight: 700;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 10px;
        margin-bottom: 15px;
        margin-top: 0;
    }
    .summary-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .details-box { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); }
    
    /* 调整 Plotly 图表 */
    .js-plotly-plot .plotly .main-svg { background: rgba(0,0,0,0) !important; }
    </style>
""", unsafe_allow_html=True)

# --- 状态管理 ---
if 'df_result' not in st.session_state: st.session_state.df_result = None
if 'matched_sentiments' not in st.session_state: st.session_state.matched_sentiments = None
if 'llm_report_summary' not in st.session_state: st.session_state.llm_report_summary = None
if 'llm_report_details' not in st.session_state: st.session_state.llm_report_details = None
if 'current_keyword' not in st.session_state: st.session_state.current_keyword = ""

# 2. 侧边栏：API Key 配置
# ==========================================
with st.sidebar:
    st.header("🔑 Gemini Settings")
    st.markdown("Use your own Gemini API Key for privacy, or leave blank to use the hosted (Gemini 2.5 Flash) key.")
    
    # 用户输入 Key (存入 session_state)
    st.text_input(
        "Enter your Gemini API Key:", 
        type="password", 
        key="user_api_key",
        help="Get one from aistudio.google.com"
    )

# 获取最终使用的 Key
MY_API_KEY = get_api_key()

# 检查 Key 是否存在 (用于后续判断)
if not MY_API_KEY:
    st.warning("⚠️ No API Key found! AI features will be disabled. Please check Streamlit Secrets or enter a key.")

# ==========================================
# 2. 数据库连接
# ==========================================
@st.cache_resource
def init_connection():
    try:
        # 1. 从Streamlit Secrets 读取连接字符串
        if "MONGO_URI" in st.secrets:
            client = MongoClient(st.secrets[MONGO_URI])
            return client
        else:
            # 报错
            st.error("🚨 MongoDB URL not found in Secrets! Please check your configuration.")
            return None
            
    except Exception as e:
        st.error(f"MongoDB Connection Error: {e}")
        return None
    
@st.cache_data
def load_sentiment_data():
    try:
        return pd.read_csv("movie_sentiment_cleaned_simple.csv")
    except FileNotFoundError:
        return pd.DataFrame()

client = init_connection()
if client:
    db = client["movies_db"]
    coll_movies = db['movies']
    coll_keywords = db['keywords']
    coll_imdb_titles = db['imdb_titles']
    coll_imdb_ratings = db['imdb_ratings']

df_sentiment_source = load_sentiment_data()

# ==========================================
# 3. 辅助函数
# ==========================================
def parse_semicolon_list(text):
    if not text or pd.isna(text): return []
    text_str = str(text)
    return [t.strip() for t in text_str.split(';') if t.strip()]

# ==========================================
# 4. 核心逻辑
# ==========================================

def search_movies_logic(keyword):
    keyword_list = [k.strip() for k in keyword.split() if k.strip()]
    if not keyword_list: return pd.DataFrame()

    or_blocks = [{"$or": [{"genres": {"$regex": k, "$options": "i"}}, {"overview": {"$regex": k, "$options": "i"}}, {"title": {"$regex": k, "$options": "i"}}]} for k in keyword_list]
    df_movies = pd.DataFrame(list(coll_movies.find({"$and": or_blocks})))

    and_keyword_conditions = [{"keywords": {"$elemMatch": {"$regex": k, "$options": "i"}}} for k in keyword_list]
    keywords_list = list(coll_keywords.find({"$and": and_keyword_conditions}))
    
    if keywords_list:
        df_keywords = pd.DataFrame(keywords_list)
        if not df_movies.empty:
            df_from_keywords = df_movies[df_movies["movie_id"].isin(df_keywords["movie_id"])]
            df_movies = pd.concat([df_movies, df_from_keywords]).drop_duplicates("movie_id")
        else:
            extra_ids = df_keywords["movie_id"].unique().tolist()
            extra_movies = list(coll_movies.find({"movie_id": {"$in": extra_ids}}))
            if extra_movies: df_movies = pd.DataFrame(extra_movies)

    if df_movies.empty: return pd.DataFrame()

    df_imdb_titles = pd.DataFrame(list(coll_imdb_titles.find({}, {"tconst": 1, "primaryTitle": 1, "startYear": 1, "genres": 1})))
    df_movies = df_movies.merge(df_imdb_titles, left_on="imdb_id", right_on="tconst", how="left", suffixes=("", "_imdb"))
    df_imdb_ratings = pd.DataFrame(list(coll_imdb_ratings.find({}, {"tconst": 1, "averageRating": 1, "numVotes": 1})))
    df_movies = df_movies.merge(df_imdb_ratings, left_on="imdb_id", right_on="tconst", how="left")

    df_movies["release_date"] = pd.to_datetime(df_movies["release_date"], errors="coerce")
    df_movies["tmdb_rating"] = pd.to_numeric(df_movies["vote_average"], errors="coerce")
    df_movies["revenue"] = pd.to_numeric(df_movies["revenue"], errors="coerce")
    df_movies["year"] = df_movies["release_date"].dt.year
    
    return df_movies

def match_sentiments(df_movies, df_sentiment_source):
    if df_movies.empty or df_sentiment_source.empty: return pd.DataFrame()
    df_m = df_movies.copy()
    df_s = df_sentiment_source.copy()
    df_m["movie_id_int"] = pd.to_numeric(df_m["movie_id"], errors="coerce").astype("Int64")
    df_s["movie_id_int"] = pd.to_numeric(df_s["movie_id"], errors="coerce").astype("Int64")
    merged = df_m.merge(df_s[["movie_id_int", "movie_name", "top5pos", "top5neg"]], on="movie_id_int", how="inner")
    result = merged[["movie_id_int", "movie_name", "top5pos", "top5neg", "revenue", "tmdb_rating", "year"]].drop_duplicates(subset="movie_id_int").rename(columns={"movie_id_int": "movie_id"})
    return result

def generate_llm_report_refined(keyword, df_movies, sentiment_df, api_key):
    """ Prompt 3.0 (最终版) """
    rev_series = pd.to_numeric(df_movies['revenue'], errors='coerce')
    median_revenue = rev_series[rev_series > 0].median()
    if pd.isna(median_revenue): median_revenue = 0
    rating_series = pd.to_numeric(df_movies['tmdb_rating'], errors='coerce')
    avg_rating = rating_series[rating_series > 0].mean()
    if pd.isna(avg_rating): avg_rating = 0
    movie_count = len(df_movies)

    context = f"User Search Keyword: '{keyword}'\n"
    context += f"Market Overview: {movie_count} movies found.\n"
    context += f"Median Box Office: ${median_revenue:,.0f}\n"
    context += f"Average TMDB Rating: {avg_rating:.1f}/10\n\n"
    context += "=== DETAILED REVIEWS & SENTIMENT ===\n"
    
    if not sentiment_df.empty:
        for idx, row in sentiment_df.iterrows():
            pos_list = parse_semicolon_list(row.get('top5pos'))[:10]
            neg_list = parse_semicolon_list(row.get('top5neg'))[:10]
            context += f"\nMovie: {row['movie_name']}\n"
            context += f"  - Likes: {'; '.join(pos_list)}\n"
            context += f"  - Dislikes: {'; '.join(neg_list)}\n"
    else:
        context += "No review data available.\n"

    prompt = f"""
    {context}
    
    === INSTRUCTIONS ===
    You are an intelligent film data analyst.  The user searched for "{keyword}".
    
    **TASK:** Write a concise yet insightful report. You MUST follow the structure below exactly.
    
    **SECTION 1: SUMMARY**
    - Synthesize all your insights into a concise 2-3 sentence summary, and 2-3 actionable insights for movie producers or marketers interested in this genre/keyword.
    - Focus on the "So What?". Is this genre profitable? What's the key success factor?
    - Keep this section punchy and bold.
    
    <<<SPLIT>>>
    
    **SECTION 2: DEEP DIVE MARKET ANALYSIS**
    - **Performance**: Analyze commercial vs critical success. Use the median revenue and avg rating.
    - **Sentiment Drivers**: What specific elements (visuals, plot, pacing, acting) are driving positive/negative reviews?
    - **Gap Analysis**: Are there high-grossing movies with bad reviews (or vice versa)?
    - Total Output Length: under 300 Words
    
    **FORMATTING RULES:**
    - Do not use Markdown formatting. like" #, **, etc.
    - Do NOT output the string "<<<SPLIT>>>" anywhere else except to separate the two sections.
    - No introductory fluff.
    - Total Output Length: Keep it under 500 words.
    """

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ==========================================
# 5. Streamlit UI
# ==========================================

st.title("🎬 CinemaPulse: Movie Market Insight")

# 将输入框和按钮包裹在 form 中，这样在输入框按回车会自动触发
with st.form(key='search_form'):
    c1, c2 = st.columns([4, 1])
    with c1:

        keyword_input = st.text_input("Search Keyword:", placeholder="e.g. Love Story, Science Fiction", label_visibility="collapsed")
    with c2:

        search_clicked = st.form_submit_button("Search Movies 🔍", type="primary")

if search_clicked and keyword_input:
    st.session_state.current_keyword = keyword_input
    st.session_state.llm_report_summary = None 
    st.session_state.llm_report_details = None
    
    with st.spinner("Fetching data from MongoDB..."):
        res = search_movies_logic(keyword_input)
        st.session_state.df_result = res
        sent = match_sentiments(res, df_sentiment_source)
        st.session_state.matched_sentiments = sent

if st.session_state.df_result is not None and not st.session_state.df_result.empty:
    
    df_res = st.session_state.df_result
    df_sent = st.session_state.matched_sentiments
    curr_kw = st.session_state.current_keyword

    rev_series = pd.to_numeric(df_res['revenue'], errors='coerce')
    median_rev = rev_series[rev_series > 0].median()
    rating_series = pd.to_numeric(df_res['tmdb_rating'], errors='coerce')
    avg_rate = rating_series[rating_series > 0].mean()

    with st.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Movies Found", len(df_res))
        kpi2.metric("Median Box Office", f"${median_rev:,.0f}" if not pd.isna(median_rev) else "$0")
        kpi3.metric("Avg TMDB Rating", f"{avg_rate:.1f} / 10" if not pd.isna(avg_rate) else "0.0")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Market Charts", "💬 Sentiment Reviews", "🤖 AI Insight"])

    # --- TAB 1: Charts ---
    with tab1:
        st.subheader("Market Performance Analysis")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig1 = px.histogram(
                df_res.dropna(subset=['tmdb_rating']), 
                x="tmdb_rating", 
                nbins=20,
                title="<b>TMDB Rating Distribution</b>",
                color_discrete_sequence=['#3366cc'],
                labels={'tmdb_rating': 'Rating'}
            )
            fig1.update_layout(bargap=0.1, font=dict(size=14), title_font_size=18)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_chart2:
            df_rev_plot = df_res.groupby('year')['revenue'].mean().reset_index().dropna()
            df_rev_plot = df_rev_plot[df_rev_plot['revenue'] > 0]
            
            if not df_rev_plot.empty:
                fig2 = px.bar(
                    df_rev_plot, 
                    x='year', 
                    y='revenue',
                    title="<b>Average Revenue Trend (Yearly)</b>",
                    color='revenue',
                    color_continuous_scale='Viridis',
                    labels={'revenue': 'Avg Revenue ($)', 'year': 'Release Year'}
                )
                fig2.update_layout(font=dict(size=14), title_font_size=18)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Insufficient revenue data for trend chart.")

        with st.expander("📄 View Raw Data Table"):
            st.dataframe(df_res[['title', 'release_date', 'genres', 'tmdb_rating', 'revenue', 'overview']], use_container_width=True)

    # --- TAB 2: Sentiment ---
    with tab2:
        st.subheader(f"Audience Sentiment for '{curr_kw}'")
        
        if not df_sent.empty:
            for i, row in df_sent.iterrows():
                with st.expander(f"🎬 {row['movie_name']} ({row['year']}) | ⭐ {row['tmdb_rating']}", expanded=False):
                    
                    pos_list = parse_semicolon_list(row['top5pos'])
                    neg_list = parse_semicolon_list(row['top5neg'])

                    c_pos, c_neg = st.columns(2)
                    
                    with c_pos:
                        if pos_list:
                            list_html = '<ul class="sentiment-list">' + "".join([f"<li>{item}</li>" for item in pos_list[:5]]) + '</ul>'
                        else: list_html = "No positive highlights."
                        st.markdown(f"""<div class="positive-box"><h5 style="color:#155724; margin-top:0;">👍 What People Liked</h5>{list_html}</div>""", unsafe_allow_html=True)

                    with c_neg:
                        if neg_list:
                            list_html = '<ul class="sentiment-list">' + "".join([f"<li>{item}</li>" for item in neg_list[:5]]) + '</ul>'
                        else: list_html = "No negative complaints."
                        st.markdown(f"""<div class="negative-box"><h5 style="color:#721c24; margin-top:0;">👎 What People Disliked</h5>{list_html}</div>""", unsafe_allow_html=True)
        else:
            st.info("No detailed sentiment reviews matched.")

    # --- TAB 3: AI Summary ---
    with tab3:
        st.subheader("🤖 Executive Market Insight")
        
        if st.session_state.llm_report_summary is None:
            st.info(f"Generate concise insights for **{len(df_res)}** movies using Gemini 2.5 Flash.")
            
            if st.button("✨ Generate Executive Report", key="btn_gen_ai"):
                with st.spinner("Analyzing market data & sentiments..."):
                    
                    full_text = generate_llm_report_refined(curr_kw, df_res, df_sent, MY_API_KEY)
                    
                    if "<<<SPLIT>>>" in full_text:
                        parts = full_text.split("<<<SPLIT>>>")
                        summary_part = parts[0].strip()
                        details_part = parts[1].strip()
                    else:
                        summary_part = "Summary generation failed to split."
                        details_part = full_text

                    st.session_state.llm_report_summary = summary_part
                    st.session_state.llm_report_details = details_part
                    
                    st.rerun()

        else:
            st.markdown(f"""
            <div class="summary-box">
                <h3>🚀 Executive Takeaways</h3>
                {st.session_state.llm_report_summary.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="details-box">
                <h3>🔍 Deep Dive Analysis</h3>
                {st.session_state.llm_report_details.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Regenerate Report"):
                st.session_state.llm_report_summary = None
                st.session_state.llm_report_details = None
                st.rerun()

elif search_clicked:
    st.warning("No movies found for this keyword.")
else:
    st.info("👆 Please enter a keyword above to start analysis.")
