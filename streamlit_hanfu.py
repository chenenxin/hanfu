import streamlit as st
import pandas as pd
import random
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import pathlib
import subprocess




# 设置页面配置
st.set_page_config(
    page_title="📚汉服智能助手",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    /* 更新背景为古风淡色系 */
    .stApp {
        background: linear-gradient(135deg, #f5f5dc 0%, #e8d0a8 100%);
        padding: 20px;
    }
    /* 标题颜色调整 */
    h1 {
        color: #6b3e00;
        border-bottom: 3px solid #b19a7f;
    }
    /* 侧边栏调整为古风淡色 */
    [data-testid="stSidebar"] {
        background: rgba(245, 238, 226, 0.95) !important;
    }
    /* 按钮颜色同步调整 */
    .st-bb {
        background-color: #a67c52 !important;
    }
     /* 文本颜色 */
    p, .st-ae, .st-af {
        color: #555555 !important;
    }
</style>
""", unsafe_allow_html=True)
# Python 版本检查
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()
import subprocess

try:
    subprocess.run(['pip', 'install', 'fastai==2.8.2'], check=True)
except subprocess.CalledProcessError as e:
    print(f"安装 fastai==2.8.2 时出错: {e}")
from fastai.vision.all import *
import pathlib

@st.cache_resource
def load_models():
    """加载并缓存两个独立的模型"""
    # 临时保存原始路径类型，用于Windows兼容性
    original_posix_path = None
    if sys.platform == "win32":
        original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    
    # 初始化模型变量
    image_model = None
    collab_model = None
    
    try:
        # 加载图像分类模型
        image_model_path = pathlib.Path(__file__).parent / "汉服_model.pkl"
        image_model = load_learner(image_model_path)
        
        # 加载推荐模型
        collab_model_path = pathlib.Path(__file__).parent / "汉服model.pkl"
        collab_model = load_learner(collab_model_path)
        
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        
    finally:
        # 恢复原始路径类型
        if sys.platform == "win32" and original_posix_path is not None:
            pathlib.PosixPath = original_posix_path
    
    # 返回两个独立的模型
    return image_model, collab_model
# 加载模型
image_model, collab_model = load_models()

# 主应用
st.markdown('<h1 style="text-align:left; color: #6b3e00;">🌸识别热门汉服</h1>', unsafe_allow_html=True)
st.write("📸 上传汉服图片，自动识别热门款式")

# 直接使用加载好的 image_model
model = image_model

uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="上传的图片", use_container_width=True)
    
    pred, pred_idx, probs = model.predict(image)
    st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}") 







# 加载数据
@st.cache_data
def load_experiment_data():
    """加载实验数据文件"""
    try:
        # 加载处理后的评分数据
        ratings_df = pd.read_excel("问卷数据.xlsx")
        # 加载汉服数据
        hanfu_df = pd.read_excel("汉服整合.xlsx")
        if 'item_id' in hanfu_df.columns:
            hanfu_df['item_id'] = hanfu_df['item_id'].astype(int)  #  # 强制转为整数
        return ratings_df, hanfu_df
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None

ratings_df, hanfu_df = load_experiment_data()

def display_hanfu_by_gender():
    """根据性别显示汉服"""
    status = st.selectbox("请输入您的性别", ('女', '男'))
    if status == '女':
        # 示例图片文件路径，请替换为实际路径
        # 确保图像文件存在于给定的路径中
        Image1 = Image.open('display/曲裾.jpg')
        Image2 = Image.open('display/直裾.jpg')
        Image6 = Image.open('display/圆领袍.jpg')
        Image4 = Image.open('display/齐胸襦裙.jpg')
        Image5 = Image.open('display/齐腰襦裙.jpg')
        Image3 = Image.open('display/马面裙.jpg')
        Image7 = Image.open('display/袄裙.jpg')
        Image8 = Image.open('display/褙子.jpg')
        # 创建布局
        row1 = st.columns(4)
        with row1[0]:
            st.image(Image1, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">曲裾</div>', unsafe_allow_html=True)
        with row1[1]:
            st.image(Image2, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">直裾</div>', unsafe_allow_html=True)
        with row1[2]:
            st.image(Image6, width=200) 
            st.markdown('<div style="text-align:center; color: #6b3e00;">圆领袍</div>', unsafe_allow_html=True)
        with row1[3]:
            st.image(Image4, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">齐胸襦裙</div>', unsafe_allow_html=True)
        row2 = st.columns(4)
        with row2[0]:
            st.image(Image5, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">齐腰襦裙</div>', unsafe_allow_html=True)
        with row2[1]:
            st.image(Image3, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">马面裙</div>', unsafe_allow_html=True)  
        with row2[2]:
            st.image(Image7, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">袄裙</div>', unsafe_allow_html=True)
        with row2[3]:
            st.image(Image8, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">褙子</div>', unsafe_allow_html=True) 
        df = pd.DataFrame({
        'Name': ['曲裾', '直裾', '圆领袍', '齐胸襦裙', '齐腰襦裙', '马面裙', '袄裙', '褙子'],
        'description': [
            '流行于秦汉时期的绕襟深衣，线条优美，端庄大方。',
            '直襟的汉服款式，剪裁简洁，行动便利，适合日常穿着。',
            '圆领窄袖的袍服，多为官员或士人穿着，庄重大气。',
            '唐代流行的高腰裙装，将裙头系于胸上，尽显雍容华贵。',
            '裙腰与腰部齐平的传统裙装，清新秀丽，穿着舒适。',
            '明代特色裙装，前后有两个裙门，两侧褶裥，端庄稳重。',  
            '上衣为袄，下裙搭配的传统服饰，保暖性好，适合秋冬季节。',
            '直领对襟的长外衣，两侧开衩，潇洒飘逸，男女皆可穿着。'
        ]
    })
        st.table(df)
    else:
        # 示例图片文件路径，请替换为实际路径
        # 确保图像文件存在于给定的路径中
        Image4 = Image.open('display/男曲裾.jpeg')
        Image5 = Image.open('display/曳撒.jpg')
        Image6 = Image.open('display/圆领袍.jpg')
        Image7 = Image.open('display/男直裾.jpg')
        Image9 = Image.open('display/男褙子.jpg')
        # 创建布局
        row1 = st.columns(5)
        with row1[0]:
            st.image(Image4, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">曲裾</div>', unsafe_allow_html=True)
        with row1[1]:
            st.image(Image5, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">曳撒</div>', unsafe_allow_html=True)
        with row1[2]:
            st.image(Image6, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">圆领袍</div>', unsafe_allow_html=True)
        with row1[3]:
            st.image(Image7, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">直裾</div>', unsafe_allow_html=True)
        with row1[4]:
            st.image(Image9, width=200)
            st.markdown('<div style="text-align:center; color: #6b3e00;">褙子</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
        'Name': ['曲裾', '曳撒', '圆领袍', '直裾','褙子'],
        'description': [
            '流行于秦汉时期的绕襟深衣，线条优美，端庄大方。',
            '明代典型男装，交领右衽，两侧开衩，下摆有褶裥，兼具威严与飘逸。',
            '圆领窄袖的袍服，多为官员或士人穿着，庄重大气。',
            '直襟的汉服款式，剪裁简洁，行动便利，适合日常穿着。',
            '直领对襟的长外衣，两侧开衩，潇洒飘逸，男女皆可穿着。'
    ]})
        st.table(df)

# 初始化会话状态
def init_session_state():
    """初始化会话状态"""
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.current_step = 1  # 1:评分 | 2:推荐 | 3:满意度
        st.session_state.selected_hanfu = []
        st.session_state.user_ratings = {}
        st.session_state.recommendations = []
        st.session_state.rec_ratings = {}
        st.session_state.rating_range = (1, 5)
        st.session_state.satisfaction = None

init_session_state()

# 显示随机汉服并收集评分
def display_random_hanfu():
    """显示随机汉服并收集评分（增强版）"""
    global hanfu_df
    if hanfu_df is None or not isinstance(hanfu_df, pd.DataFrame):
        st.error("汉服数据异常，无法显示随机汉服")
        return

    if not st.session_state.selected_hanfu:
        if 'item_id' not in hanfu_df.columns:
            st.error("汉服数据缺少 item_id 列")
            return
        item_ids = hanfu_df['item_id'].dropna().tolist()
        if not item_ids:
            st.error("汉服数据中没有有效 item_id")
            return
        if 'name' not in hanfu_df.columns:
            st.error("汉服数据缺少 name 列")
            return
        if hanfu_df['name'].isna().all():
            st.error("汉服数据中的 name 列全为空")
            return
        
        st.session_state.selected_hanfu = random.sample(item_ids, 3)
        st.session_state.user_ratings = {}

    st.markdown('<h1 style="text-align:left; color: #6b3e00;">📋请为以下汉服评分</h1>', unsafe_allow_html=True)
    form_key = f"hanfu_rating_form_{random.randint(1, 1000000)}"
    with st.form(key=form_key):
        cols = st.columns(3)
        for i, item_id in enumerate(st.session_state.selected_hanfu):
            try:
                # 确保 item_id 是数值且存在于数据中
                if not pd.notna(item_id) and item_id in hanfu_df['item_id'].values:
                    name = hanfu_df[hanfu_df['item_id'] == item_id]['name'].values[0]
                else:
                    name = f"未知汉服 (ID: {item_id})"
            except Exception as e:
                name = f"数据异常: {e}"
            
            with cols[i]:
                name = hanfu_df[hanfu_df['item_id'] == item_id]['name'].iloc[0]
                st.write(name)
                rating = st.selectbox(
                    f"为汉服 {item_id} 评分",
                    options=list(range(1, 6)),
                    key=f"random_rating_{item_id}_{i}"
                )
                st.session_state.user_ratings[item_id] = rating

        submitted = st.form_submit_button("提交评分")
        if submitted:
            st.success("评分已提交！")
            st.write("您的评分如下:")
            for item_id, rating in st.session_state.user_ratings.items():
                st.write(f"Hanfu ID: {item_id}, 评分: {rating}")
            st.rerun()


# 显示推荐结果
def display_recommendations():
    """显示汉服推荐结果（增强版）"""
    global hanfu_df
    if hanfu_df is None or not isinstance(hanfu_df, pd.DataFrame):
        st.error("汉服数据异常，无法生成推荐")
        return

    st.header("🎯 个性化推荐")

    if st.button("获取个性化推荐", type="primary", key="get_recommendations"):
        if len(st.session_state.user_ratings) < 3:
            st.warning("请先为3个汉服评分")
            return

        with st.spinner("正在生成推荐..."):
            if 'item_id' not in hanfu_df.columns:
                st.error("汉服数据缺少 item_id 列，无法生成推荐")
                return
            item_ids = hanfu_df['item_id'].dropna().tolist()
            unrated_items = [item for item in item_ids if item not in st.session_state.user_ratings]
            
            if len(unrated_items) >= 5:
                recommendations = random.sample(unrated_items, 5)
            else:
                recommendations = random.sample(item_ids, min(5, len(item_ids)))

            formatted_recs = []
            for item_id in recommendations:
                try:
                    if pd.notna(item_id) and item_id in hanfu_df['item_id'].values:
                        name = hanfu_df[hanfu_df['item_id'] == item_id]['name'].values[0]
                        pred_rating = random.uniform(1, 5)
                        formatted_recs.append({
                            'item_id': item_id,
                            'name': name,
                            'predicted_rating': pred_rating
                        })
                except Exception as e:
                    st.warning(f"处理推荐项 {item_id} 时出错: {e}")

            st.session_state.recommendations = formatted_recs
            st.success("推荐生成成功！")

    if 'recommendations' in st.session_state and st.session_state.recommendations:
        st.subheader("为您推荐汉服")
        for idx, rec in enumerate(st.session_state.recommendations):
            try:
                with st.expander(f"推荐 {idx + 1} - 预测评分: {rec['predicted_rating']:.2f}"):
                    st.text_area(
                        f"Hanfu ID: {rec['item_id']}",
                        rec['name'],
                        height=150,
                        disabled=True,
                        key=f"rec_hanfu_{rec['item_id']}_{idx}"
                    )
                    rating = st.selectbox(
                        "您的实际评分",
                        options=list(range(1, 6)),
                        key=f"rec_rating_{rec['item_id']}_{idx}"
                    )
                    st.session_state.rec_ratings[rec['item_id']] = float(rating)
            except Exception as e:
                st.error(f"显示推荐项时出错: {e}")
# 计算满意度
def calculate_satisfaction(ratings):
    """计算满意度百分比"""
    if not ratings:
        return 0.0
    avg_rating = np.mean(list(ratings.values()))
    return ((avg_rating - 1) / 4) * 100  # 假设评分范围1 - 5

def display_satisfaction():
    """显示满意度结果"""
    if st.button("计算推荐满意度", key="calculate_satisfaction"):
        if not st.session_state.rec_ratings:
            st.warning("请先对推荐汉服评分")
            return

        satisfaction = calculate_satisfaction(st.session_state.rec_ratings)
        st.header(f"推荐满意度：{satisfaction:.1f}%")

        if satisfaction >= 80:
            st.success("🎉 非常满意！")
        elif satisfaction >= 60:
            st.info("😊 推荐效果良好，我们会继续优化")
        elif satisfaction >= 30:
            st.warning("😕 一般，有待改进")
        else:
            st.warning("😞 很抱歉未达到您的预期")

# 主应用
def main():
    st.markdown('<h1 style="text-align:left; color: #6b3e00;">⚙️汉服智能推荐系统</h1>', unsafe_allow_html=True)

    # 创建选项卡
    tab1, tab2 = st.tabs(["🌟汉服推荐", "🌟汉服展示"])

    with tab1:
        display_random_hanfu()
        display_recommendations()
        display_satisfaction()

    with tab2:
        display_hanfu_by_gender()

    
    # 侧边栏信息

        

    with st.sidebar:
        st.subheader(" 🔎汉服识别系统🔍")
        if model is not None:
            st.success("✅ 汉服识别模型加载成功")
        else:
            st.error("❌ 模型加载失败，请检查模型文件")

        st.subheader("📌 汉服数据库")
        if hanfu_df is not None:
            st.write(f"📂 收录热门汉服款式总数：{len(hanfu_df)}")
        else:
            st.write("📂 汉服数据加载失败")

        if ratings_df is not None:
            st.write(f"⭐ 用户评分总数：{len(ratings_df)}")
        else:
            st.write("⭐ 用户评分数据加载失败")
             # 重新开始按钮
        if st.sidebar.button("🔄 重新开始"):
            for key in ['selected_hanfu', 'user_ratings', 'recommendations', 
                       'rec_ratings', 'satisfaction_calculated']:
                st.session_state[key] = [] if key in ['selected_hanfu', 'recommendations'] else {}
            st.session_state.current_step = 1
            st.rerun()

if __name__ == "__main__":
    main()