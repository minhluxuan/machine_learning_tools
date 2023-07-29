import streamlit as st
import pandas as pd
from module import ModelLinearRegression, ModelRidgeRegression, ModelLassoRegression, ModelPolynomialRegression
from sklearn.datasets import load_diabetes, fetch_california_housing, load_wine
def regression():
    datasets = {
        "Diabetes": load_diabetes,
        "Wine": load_wine,
        "California Housing": fetch_california_housing
    }
    st.markdown("##### Trang chủ  »  Hồi quy")
    data_file = st.file_uploader("Tải lên tệp dữ liệu:")
    dataset = st.selectbox("Hoặc chọn tập dữ liệu có sẵn:", ["Diabetes", "Wine", "California Housing"])
    if data_file is not None:
        df = pd.read_excel(data_file)
        features = df.columns
    elif dataset is not None:
        data = datasets[dataset]()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        features = df.columns
    with st.expander("Tùy chỉnh"):
        model_type = st.selectbox("Mô hình:", ["Ordinary Least Square", "Ridge", "Lasso"])
        if not model_type == "Ordinary Least Square":
            is_auto_tunned = st.checkbox("Tự động điều chỉnh", value=True)
            if is_auto_tunned == True:
                alpha = None
            else:
                alpha = st.number_input("Hệ số alpha:", step=0.01, min_value=0.00, max_value=1000.00)
        degree = st.slider("Bậc: ", 1, 100)
        train_size = st.number_input("Kích thước tập huấn luyện:", step=0.01, min_value=0.00, max_value=1.00)
        if data_file is not None:
            feature_y = st.selectbox("Biến phụ thuộc:", features)
            y = df[feature_y]
        elif dataset is not None:
            feature_y = st.selectbox("Biến phụ thuộc:", ["Target"], disabled=True)
            y = data.target
        features_X = st.multiselect("Biến độc lập:", ["Tất cả"] + list(features))
        if "Tất cả" in features_X:
            features_X = features
        X = df[features_X]
        apply_button = st.button("Áp dụng")

    def show_result():
        if model_type == "Ordinary Least Square":
            if len(features_X) == 1:
                if degree == 1:
                    model = ModelLinearRegression(X, y, train_size)
                else:
                    model = ModelPolynomialRegression(X, y, "Ordinary Least Square", degree, train_size)
                figure = model.figure(features_X[0], feature_y)
                st.pyplot(figure)
            else:
                model = ModelLinearRegression(X, y, train_size)
        elif model_type == "Ridge":
            if len(features_X) == 1:
                if degree == 1:
                    model = ModelRidgeRegression(X, y, train_size, alpha)
                else:
                    model = ModelPolynomialRegression(X, y, "Ridge", degree, train_size, alpha=alpha)
                figure = model.figure(features_X[0], feature_y)
                st.pyplot(figure)
            else:
                model = ModelRidgeRegression(X, y, train_size, alpha)
        elif model_type == "Lasso":
            if len(features_X) == 1:
                if degree == 1:
                    model = ModelLassoRegression(X, y, train_size, alpha)
                else:
                    model = ModelPolynomialRegression(X, y, "Ridge", degree, train_size, alpha=alpha)
                figure = model.figure(features_X[0], feature_y)
                st.pyplot(figure)
            else:
                model = ModelLassoRegression(X, y, train_size, alpha)
        formula_text = model.formula
        st.write("Phương trình hồi quy:")
        st.latex(formula_text)
        st.write("Độ chính xác trên tập kiểm tra:")
        st.latex(f"{model.score}")
    if apply_button:
        show_result()         
def kmeans():
    pass

def main():
    pages = {
        "Hồi quy": regression,
        "K-Means": kmeans
    }
    st.title("Alpha Solutions Lab")
    st.sidebar.title("Machine Learning Tools")
    selected_page = st.sidebar.radio("", list(pages.keys()))
    pages[selected_page]()

if __name__ == "__main__":
    main()