import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit 설정 및 타이틀
st.set_page_config(layout="wide")
st.title("💪 피트니스 데이터 속성 간 상관관계 분석")
st.markdown("---")

# 파일 경로 (업로드된 파일 이름과 일치해야 함)
FILE_PATH = "fitness data (1).xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv"

@st.cache_data
def load_data(path):
    """CSV 파일을 로드하고 필요한 전처리를 수행합니다."""
    try:
        df = pd.read_csv(path, encoding='cp949') # 한글 인코딩 문제 해결을 위해 'cp949' 사용
        
        # 숫자형 데이터만 선택하고, NaN 값은 평균으로 채우기
        # '나이'와 '신장', '체중' 등의 주요 속성을 제외한 모든 컬럼에 대해 시도
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # 상관관계 행렬 계산
        corr_matrix = numeric_df.corr()
        return corr_matrix, numeric_df
    except FileNotFoundError:
        st.error(f"⚠️ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {path}")
        return None, None
    except Exception as e:
        st.error(f"⚠️ 데이터 로드 및 처리 중 오류 발생: {e}")
        return None, None

def find_extreme_correlations(corr_matrix, positive=True):
    """
    상관관계 행렬에서 가장 높거나(positive=True) 가장 낮은(positive=False) 
    상관관계를 가지는 속성 쌍을 찾습니다.
    """
    # 자기 자신과의 상관관계(1)를 제외
    np.fill_diagonal(corr_matrix.values, np.nan) 
    
    if positive:
        # 가장 높은 값 찾기
        max_corr = corr_matrix.max().max()
        if pd.isna(max_corr):
            return None, None, None
            
        # 해당 값을 가진 쌍의 인덱스 찾기
        (row, col) = np.where(corr_matrix == max_corr)
        
        # 첫 번째 쌍만 반환 (여러 쌍이 있을 수 있으나 대표값으로 1개만)
        feature1 = corr_matrix.index[row[0]]
        feature2 = corr_matrix.columns[col[0]]
        return feature1, feature2, max_corr
    else:
        # 가장 낮은(음의) 값 찾기
        min_corr = corr_matrix.min().min()
        if pd.isna(min_corr):
            return None, None, None
            
        # 해당 값을 가진 쌍의 인덱스 찾기
        (row, col) = np.where(corr_matrix == min_corr)
        
        # 첫 번째 쌍만 반환
        feature1 = corr_matrix.index[row[0]]
        feature2 = corr_matrix.columns[col[0]]
        return feature1, feature2, min_corr

# 데이터 로드 및 상관관계 계산
corr_matrix, numeric_df = load_data(FILE_PATH)

if corr_matrix is not None:
    
    col1, col2 = st.columns(2)
    
    with col1:
        # --- 양의 상관관계 버튼 ---
        if st.button("📈 가장 높은 양의 상관관계 속성 보기", type="primary"):
            st.subheader("✅ 가장 높은 양의 상관관계")
            feat1_pos, feat2_pos, corr_val_pos = find_extreme_correlations(corr_matrix.copy(), positive=True)
            
            if feat1_pos and feat2_pos:
                st.info(f"**속성 쌍:** **{feat1_pos}** & **{feat2_pos}**")
                st.success(f"**상관관계 값:** **{corr_val_pos:.4f}**")
                
                # 산점도 시각화
                st.markdown("#### 산점도")
                fig_pos = px.scatter(
                    numeric_df, 
                    x=feat1_pos, 
                    y=feat2_pos,
                    title=f"'{feat1_pos}' vs '{feat2_pos}' (r={corr_val_pos:.2f})",
                    template="plotly_white"
                )
                st.plotly_chart(fig_pos, use_container_width=True)
            else:
                st.warning("분석할 수 있는 숫자형 데이터가 충분하지 않습니다.")
    
    with col2:
        # --- 음의 상관관계 버튼 ---
        if st.button("📉 가장 높은 음의 상관관계 속성 보기", type="secondary"):
            st.subheader("❌ 가장 높은 음의 상관관계")
            feat1_neg, feat2_neg, corr_val_neg = find_extreme_correlations(corr_matrix.copy(), positive=False)
            
            if feat1_neg and feat2_neg:
                st.info(f"**속성 쌍:** **{feat1_neg}** & **{feat2_neg}**")
                st.error(f"**상관관계 값:** **{corr_val_neg:.4f}**")
                
                # 산점도 시각화
                st.markdown("#### 산점도")
                fig_neg = px.scatter(
                    numeric_df, 
                    x=feat1_neg, 
                    y=feat2_neg,
                    title=f"'{feat1_neg}' vs '{feat2_neg}' (r={corr_val_neg:.2f})",
                    template="plotly_white"
                )
                st.plotly_chart(fig_neg, use_container_width=True)
            else:
                st.warning("분석할 수 있는 숫자형 데이터가 충분하지 않습니다.")

    st.markdown("---")
    
    # 전체 상관관계 히트맵 (선택 사항)
    st.subheader("📊 전체 속성 간 상관관계 히트맵")
    
    # 히트맵에 표시할 컬럼 개수 제한 (컬럼이 너무 많으면 시각화가 어려워짐)
    if corr_matrix.shape[0] > 30:
        st.warning(f"경고: 컬럼 수가 {corr_matrix.shape[0]}개로 너무 많아 시각화가 어려울 수 있습니다. 상위 30개 컬럼만 표시합니다.")
        # '신장', '체중', '나이'를 포함하도록 상위 30개 컬럼 선택 (간소화)
        cols_to_plot = corr_matrix.index[:30] 
        corr_matrix_plot = corr_matrix.loc[cols_to_plot, cols_to_plot]
    else:
        corr_matrix_plot = corr_matrix
        
    fig_heatmap = px.imshow(
        corr_matrix_plot,
        text_auto=".2f",
        aspect="auto",
        title="상관관계 히트맵 (Heatmap)",
        color_continuous_scale=px.colors.diverging.RdBu,
        range_color=[-1, 1]
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
