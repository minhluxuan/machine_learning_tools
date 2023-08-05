import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from module import ModelLinearRegression, ModelRidgeRegression, ModelLassoRegression, ModelPolynomialRegression, kmeans_assign_labels
from sklearn.datasets import load_diabetes, fetch_california_housing, load_wine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
# from mnist_centroids import centroids
import pickle

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

def digits_recognition():
    # mnist = fetch_openml(name="mnist_784", version=1, parser='auto')
    # X, y = mnist["data"].astype('float32') / 255.0, mnist["target"].astype(int)

    model_selected = st.selectbox("Chọn mô hình", ["K-means", "Mạng nơ ron nhân tạo"])
    
    col1, col2 = st.columns(2)
    with open("trained_models/mnist/ann.pkl", "rb") as file:
        model = pickle.load(file)
    with open("trained_models/mnist/kmeans.pkl", "rb") as file:
        centroids = pickle.load(file)
    with col1:
        st.header("Chữ viết tay")
        canvas_result = st_canvas(
            fill_color="black",  # Màu nền khung vẽ
            stroke_width=30,  # Độ dày của nét vẽ
            stroke_color="white",  # Màu nét vẽ
            background_color="black",  # Màu nền trang web
            width=300,
            height=300,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None:
            image_array = np.array(canvas_result.image_data)
            image = Image.fromarray(image_array)
            new_size = (28, 28)
            image_resized = image.resize(new_size, Image.LANCZOS)
            image_resized_grayscaled_flatten_array = np.array(image_resized.convert("L")).flatten()
            # Normalize and reshape the image data
            input_data = image_resized_grayscaled_flatten_array.astype('float32') / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            # Predict the digit using the pre-trained model
            if model_selected == "Mạng nơ ron nhân tạo":
                prediction = model.predict(input_data)
                predicted_label = np.argmax(prediction[0])
            elif model_selected == "K-means":
                digits = [9,1,4,7,3,6,0,5,4,2]
                predicted_label = digits[kmeans_assign_labels([image_resized_grayscaled_flatten_array], centroids)[0]]
    with col2:
        st.header("Kết quả dự đoán")
        st.write(f'## {predicted_label}')
        # You can also display the confidence/probability of the prediction
        # st.write(f'Độ tin cậy: {prediction[0][predicted_label]:.2f}')

def spam_detection():
    df = pd.read_csv("./data/spam.csv", nrows=3000)
    msg = np.array(list(df['Message']))
    with open('trained_models/spam_detection/spam_detection.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    vectorizer = CountVectorizer()
    vectorizer.fit_transform(msg)
    vocabs = vectorizer.vocabulary_

    message = st.text_area("Nhập vào tin nhắn tiếng Anh:")
    check_button = st.button("Kiểm tra")

    if check_button:
        vectorizer = CountVectorizer(vocabulary=vocabs)
        transformed_message_array = vectorizer.fit_transform([message]).toarray()
        predicted_category = loaded_model.predict(transformed_message_array)
        status = "Thư rác" if predicted_category == 1 else "Không phải thư rác"
        st.write(status)

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans, MiniBatchKMeans

def test():
    pass


def main():
    pages = {
        "Hồi quy": regression,
        "K-Means": kmeans,
        "Nhận diện chữ số": digits_recognition,
        "Kiểm tra tin nhắn rác": spam_detection,
        "Test": test
    }
    st.title("Alpha Solutions Lab")
    st.sidebar.title("Machine Learning Tools")
    selected_page = st.sidebar.radio("", list(pages.keys()))
    pages[selected_page]()

if __name__ == "__main__":
    main()