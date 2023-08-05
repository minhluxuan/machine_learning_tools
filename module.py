import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

class ModelLinearRegression:
    def __init__(self, X, y, train_size):
        self.X = X
        self.y = y
        self.train_size = train_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=self.train_size, random_state=42)
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
        self.score = self.model.score(self.X_test, self.y_test)
        self.formula = self.formula()

    def f(self, X):
        return self.coef*X.T + self.intercept

    def predict(self, X):
        return self.model.predict(X)
    
    def figure(self, features_X, feature_y):
        fig, axes = plt.subplots()
        num = int((max(self.X[features_X]) - min(self.X[features_X])))*10
        X_model = np.linspace(min(self.X[features_X]), max(self.X[features_X]), num)
        y_model = self.f(X_model)
        axes.scatter(self.X[features_X], self.y, marker='o')
        axes.plot(X_model, y_model, color="red", label="OLS")
        axes.set_xlabel(features_X)
        axes.set_ylabel(feature_y)
        axes.set_title("Hồi quy tuyến tính")
        return fig
    
    def formula(self):
        formula_text = f"{self.intercept} "
        for i in range(0, self.X.shape[1]):
            index = f"{{{i+1}}}"
            formula_text += (f"+ {self.coef[i]} .x_{index}") if self.coef[i] > 0 else (f" {self.coef[i]} .x_{index}")
        return formula_text
    
class ModelRidgeRegression:
    def __init__(self, X, y, train_size, alpha = None):
        self.X = X
        self.y = y
        self.train_size = train_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=self.train_size, random_state=42)
        if alpha == None:
            self.model = RidgeCV(cv=10)
        else:
            self.model = Ridge(alpha=alpha)
        self.model.fit(self.X_train, self.y_train)
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
        self.score = self.model.score(self.X_test, self.y_test)
        self.formula = self.formula()

    def f(self, X):
        return self.coef*X.T + self.intercept

    def predict(self, X):
        return self.model.predict(X)
    
    def figure(self, features_X, feature_y):
        fig, axes = plt.subplots()
        num = int((max(self.X[features_X]) - min(self.X[features_X])))*10
        X_model = np.linspace(min(self.X[features_X]), max(self.X[features_X]), num)
        y_model = self.f(X_model)
        axes.scatter(self.X[features_X], self.y, marker='o')
        axes.plot(X_model, y_model, color="red", label="OLS")
        axes.set_xlabel(features_X)
        axes.set_ylabel(feature_y)
        axes.set_title("Hồi quy Ridge")
        return fig
    
    def formula(self):
        formula_text = f"{self.intercept} "
        for i in range(0, self.X.shape[1]):
            index = f"{{{i+1}}}"
            formula_text += (f"+ {self.coef[i]} .x_{index}") if self.coef[i] > 0 else (f" {self.coef[i]} .x_{index}")
        return formula_text

class ModelLassoRegression:
    def __init__(self, X, y, train_size, alpha = None):
        self.X = X
        self.y = y
        self.train_size = train_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=self.train_size, random_state=42)
        if alpha == None:
            self.model = LassoCV(cv=10)
        else:
            self.model = Lasso(alpha=alpha)
        self.model.fit(self.X_train, self.y_train)
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
        self.score = self.model.score(self.X_test, self.y_test)
        self.formula = self.formula()

    def f(self, X):
        return self.coef*X.T + self.intercept

    def predict(self, X):
        return self.model.predict(X)
    
    def figure(self, features_X, feature_y):
        fig, axes = plt.subplots()
        num = int((max(self.X[features_X]) - min(self.X[features_X])))*10
        X_model = np.linspace(min(self.X[features_X]), max(self.X[features_X]), num)
        y_model = self.f(X_model)
        axes.scatter(self.X[features_X], self.y, marker='o')
        axes.plot(X_model, y_model, color="red", label="OLS")
        axes.set_xlabel(features_X)
        axes.set_ylabel(feature_y)
        axes.set_title("Hồi quy Lasso")
        return fig
    
    def formula(self):
        formula_text = f"{self.intercept}"
        for i in range(0, self.X.shape[1]):
            index = f"{{{i+1}}}"
            formula_text += (f"+ {self.coef[i]} .x_{index}") if self.coef[i] > 0 else (f" {self.coef[i]} .x_{index}")
        return formula_text

class ModelPolynomialRegression:
    def __init__(self, X, y, model_type, degree, train_size, alpha=None):
        self.X = X
        self.y = y
        self.degree = degree
        self.train_size = train_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=self.train_size, random_state=42)
        self.poly = PolynomialFeatures(degree=self.degree)
        X_train_poly = self.poly.fit_transform(self.X_train)
        if model_type == "Ordinary Least Square":
            self.model = LinearRegression()
        elif model_type == "Ridge":
            if alpha == None:
                self.model = RidgeCV(cv=10)
            else:
                self.model = Ridge(alpha=alpha)
        elif model_type == "Lasso":
            if alpha == None:
                self.model = LassoCV(cv=10)
            else:
                self.model = Lasso(alpha=alpha)
        self.model.fit(X_train_poly, self.y_train)
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
        X_test_poly = self.poly.fit_transform(self.X_test)
        self.score = self.model.score(X_test_poly, self.y_test)
        self.formula = self.formula()

    def f(self, X):
        return self.coef @ X.T + self.intercept

    def figure(self, features_X, feature_y):
        fig, axes = plt.subplots()
        num = int((max(self.X[features_X]) - min(self.X[features_X])))*10
        X_model = np.linspace(min(self.X[features_X]), max(self.X[features_X]), num).reshape(-1, 1)
        X_model_poly = self.poly.fit_transform(X_model)
        y_model = self.f(X_model_poly)
        axes.scatter(self.X[features_X], self.y, marker='o')
        axes.plot(X_model, y_model, color="red", label="OLS")
        axes.set_xlabel(features_X)
        axes.set_ylabel(feature_y)
        axes.set_title("Hồi quy đa thức")
        return fig

    def formula(self):
        formula_text = f"{self.intercept} "
        for i in range(1, self.degree+1):
            index = f"{{{i+1}}}"
            formula_text += (f"+ {self.coef[i]} .x_{index}") if self.coef[i] > 0 else (f" {self.coef[i]} .x_{index}")
        return formula_text
    
def kmeans_assign_labels(X, centroids):
        dists = cdist(X, centroids)
        return np.argmin(dists, axis=1)