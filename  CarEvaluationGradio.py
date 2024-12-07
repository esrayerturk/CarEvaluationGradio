import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo

# Page configuration
st.set_page_config(
    page_title="Car Evaluation Analysis",
    page_icon="🚗",
    layout="wide"
)

# Title and introduction
st.title("🚗 Car Evaluation Analysis Dashboard")
st.markdown("""
This dashboard analyzes car evaluation data using different machine learning models.
The dataset includes various car attributes and their evaluation classifications.
""")


# Load and prepare data
@st.cache_data
def load_data():
    car_evaluation = fetch_ucirepo(id=19)
    X, y = car_evaluation.data.features, car_evaluation.data.targets
    df = pd.concat([X, y], axis=1)
    return df, X, y


df, X, y = load_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Model Training", "Model Comparison"])

# Data Overview Page
if page == "Data Overview":
    st.header("Dataset Overview")

    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(df):,}"
        )

    with col2:
        st.metric(
            label="Features",
            value=len(df.columns) - 1
        )

    with col3:
        st.metric(
            label="Target Classes",
            value=len(df['class'].unique())
        )

    with col4:
        st.metric(
            label="Missing Values",
            value=df.isnull().sum().sum()
        )

    st.write("")

    # Sample Data
    st.subheader("Sample Data")
    st.dataframe(
        df.head(),
        use_container_width=True,
        height=230
    )

    # Target Class Distribution
    st.subheader("Target Class Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='class', palette='viridis')
        plt.title('Distribution of Car Evaluations')
        st.pyplot(fig)

    with col2:
        st.write("")
        st.write("")
        class_distribution = df['class'].value_counts()
        for class_name, count in class_distribution.items():
            st.metric(
                label=class_name,
                value=count
            )

# Exploratory Analysis Page
elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    # Feature Distribution
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select Feature", df.columns[:-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=feature_to_plot, palette='coolwarm')
    plt.title(f'Distribution of {feature_to_plot}')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Feature vs Target
    st.subheader("Feature vs Target Class")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df, x=feature_to_plot, hue='class', palette='Set2')
    plt.title(f'{feature_to_plot} Distribution by Class')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    encoded_df = pd.get_dummies(df, drop_first=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(encoded_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap of Encoded Features')
    st.pyplot(fig)

# Model Training Page
elif page == "Model Training":
    st.header("Model Training and Evaluation")

    # Data preprocessing
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)
    y_encoded = y.values.ravel()

    # Train-test split
    test_size = st.slider("Select Test Size", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=42
    )

    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["Support Vector Machine", "Random Forest", "Logistic Regression"]
    )

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if model_choice == "Support Vector Machine":
                model = SVC(kernel='linear', random_state=42)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = LogisticRegression(max_iter=500, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Performance")
                accuracy = accuracy_score(y_test, y_pred)
                st.metric(label="Accuracy", value=f"{accuracy:.4f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

            with col2:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    confusion_matrix(y_test, y_pred),
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test)
                )
                plt.title(f'{model_choice} Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

            # Feature importance for Random Forest
            if model_choice == "Random Forest":
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': encoder.get_feature_names_out(),
                    'importance': model.feature_importances_
                })
                feature_importance = feature_importance.sort_values(
                    'importance', ascending=False
                ).head(10)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=feature_importance,
                    x='importance',
                    y='feature'
                )
                plt.title('Top 10 Most Important Features')
                st.pyplot(fig)

# Model Comparison Page
else:
    st.header("Model Comparison")

    if st.button("Compare All Models"):
        with st.spinner("Training all models..."):
            # Data preprocessing
            encoder = OneHotEncoder(sparse_output=False)
            X_encoded = encoder.fit_transform(X)
            y_encoded = y.values.ravel()

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y_encoded, test_size=0.2, random_state=42
            )

            # Train all models
            models = {
                "SVM": SVC(kernel='linear', random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'predictions': y_pred
                }

            # Display comparison results
            st.subheader("Accuracy Comparison")
            accuracy_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[model]['accuracy'] for model in results.keys()]
            })

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(accuracy_df)

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=accuracy_df,
                    x='Model',
                    y='Accuracy',
                    palette='viridis'
                )
                plt.title('Model Accuracy Comparison')
                plt.ylim(0, 1)
                st.pyplot(fig)

            # Detailed model comparison
            st.subheader("Detailed Model Performance")
            for name in results.keys():
                st.write(f"\n{name}:")
                st.text(classification_report(y_test, results[name]['predictions']))

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    confusion_matrix(y_test, results[name]['predictions']),
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=np.unique(y_test),
                    yticklabels=np.unique(y_test)
                )
                plt.title(f'{name} Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

# Footer
st.markdown("""
---
Created with ❤️ using Streamlit
""")