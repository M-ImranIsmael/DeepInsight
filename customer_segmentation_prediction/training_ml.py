#%% Libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#%% Step 1) Data Loading

CSV_PATH = os.path.join(os.getcwd(), "dataset", "train.csv")
df = pd.read_csv(CSV_PATH)

#%% Step 2) EDA

print(df.head())
print(df.info())
print(df.shape)
print(df.describe(include="all").T)

#%% Dropping ID COLUMN

# Dropping columns id
df.drop(labels="ID", axis=1, inplace=True)
print(df.info())

#%% Categorical VS continuous column
print(df.nunique())
columns = df.columns
categorical_columns = []
continuous_columns = []

for col in columns:
    if df[col].nunique() <= 9:
        categorical_columns.append(col)
    else:
        continuous_columns.append(col)

print(f"Categorical columns: {categorical_columns}")
print(f"Continuous columns: {continuous_columns}")

# Target: Segmentation/ Aggregation
df.groupby(["Segmentation", "Profession"]).agg({"Segmentation": "count"}).plot(
    kind="bar"
)
df.groupby(["Segmentation", "Gender"]).agg({"Segmentation": "count"}).plot(kind="bar")
df.groupby(["Segmentation", "Ever_Married"]).agg({"Segmentation": "count"}).plot(
    kind="bar"
)


#%%
# Visual EDA
for col in categorical_columns:
    plt.figure()
    sns.countplot(x=df[col])
    plt.show()

for col in continuous_columns:
    plt.figure()
    sns.distplot(df[col])
    plt.show()

#%%
for col in categorical_columns:
    print(f"{col}: {df[col].dtype}")

#%% Step 3) Data Cleaning

df.isna().sum()

#%%

if not os.path.exists("pickle_files"):
    os.makedirs("pickle_files")

le = LabelEncoder()

for cat in categorical_columns:
    if cat == "Family_Size":
        continue
    else:
        temp = df[cat]
        temp[df[cat].notnull()] = le.fit_transform(temp[df[cat].notnull()])
        df[cat] = pd.to_numeric(df[cat], errors="coerce")
        with open(os.path.join("pickle_files", col + "_le.pkl"), "wb") as f:
            pickle.dump(le, f)

#%%
print(df.info())
#%%
column_names = df.columns

ki = KNNImputer(n_neighbors=5)
df = ki.fit_transform(df)

df = pd.DataFrame(df, columns=column_names)
print(df.isna().sum())


#%% Step 4) Feature Selections

for con in continuous_columns:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con], axis=-1), df["Segmentation"])
    print(con)
    print(lr.score(np.expand_dims(df[con], axis=-1), df["Segmentation"]))


#%%


pca = PCA(n_components=2)
pca_X = pca.fit_transform(df.drop(labels=["Segmentation"], axis=1))
print(pca.explained_variance_ratio_)


PC_values = np.array(pca.n_components_) + 1
plt.figure()
plt.plot(pca.explained_variance_ratio_)


#%%
plt.figure()
plt.scatter(pca_X[:, 0], pca_X[:, 1])
plt.xlabel("PCA AXIS 1")
plt.ylabel("PCA AXIS 2")
plt.show()

#%%


kmeans = KMeans(n_clusters=4)  # For 4 segmentation (A, B, C, D)
kmeans.fit(pca_X)
y_pred_km = kmeans.predict(pca_X)

for i in np.unique(y_pred_km):
    filtered_y_pred = pca_X[y_pred_km == i]
    plt.scatter(filtered_y_pred[:, 0], filtered_y_pred[:, 1], label=i)
    plt.legend()


#%%

rf = RandomForestClassifier()
rf.fit(pca_X, df["Segmentation"])
y_pred_rf = rf.predict(pca_X)

print(classification_report(df["Segmentation"], y_pred_rf))


#%% Step 5) Data Preprocessing

X_train = df.drop(labels="Segmentation", axis=1)
y_train = df["Segmentation"]

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.3, random_state=123
)


#%% Step 6) Model Development


# Logistic Regression
pipeline_mms_lr = Pipeline(
    [("min_max_scaler", MinMaxScaler()), ("logistic_regression", LogisticRegression())]
)

pipeline_ss_lr = Pipeline(
    [
        ("standard_scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression()),
    ]
)

# Random Forest
pipeline_mms_rf = Pipeline(
    [("min_max_scaler", MinMaxScaler()), ("random_forest", RandomForestClassifier())]
)

pipeline_ss_rf = Pipeline(
    [("standard_scaler", StandardScaler()), ("random_forest", RandomForestClassifier())]
)

# Gradient Boost
pipeline_mms_gb = Pipeline(
    [
        ("min_max_scaler", MinMaxScaler()),
        ("gradient_boost", GradientBoostingClassifier()),
    ]
)

pipeline_ss_gb = Pipeline(
    [
        ("standard_scaler", StandardScaler()),
        ("gradient_boost", GradientBoostingClassifier()),
    ]
)


# SVC
pipeline_mms_svc = Pipeline([("min_max_scaler", MinMaxScaler()), ("svc", SVC())])

pipeline_ss_svc = Pipeline([("standard_scaler", StandardScaler()), ("svc", SVC())])


pipelines = [
    pipeline_mms_lr,
    pipeline_ss_lr,
    pipeline_mms_rf,
    pipeline_ss_rf,
    pipeline_mms_gb,
    pipeline_ss_gb,
    pipeline_mms_svc,
    pipeline_ss_svc,
]


pipe_dict = {}
best_score = 0
# Key for the dictionary
model_name = [
    "MMS + Logistic Regression",
    "SS + Logistic Regression",
    "MMS + Random Forest",
    "SS + Random Forest",
    "MMS + Gradient Boost",
    "SS + Gradient Boost",
    "MMS + SVC",
    "SS + SVC",
]


for pipe in pipelines:
    pipe.fit(X_train, y_train)
# Model Evaluation
for i, model in enumerate(pipelines):
    y_pred = model.predict(X_test)
    # key is mode_name, values are accuracy score and f1 score
    pipe_dict[model_name[i]] = [
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average="weighted"),
    ]

    # To get the best score from each model
    if model.score(X_test, y_test) > best_score:
        best_score = model.score(X_test, y_test)
        best_pipeline = model_name[i]


#%%
print(
    "The best model is {} with the accuracy score of {}".format(
        best_pipeline, best_score
    )
)
print(pipe_dict)
# Converting pipe into DataFrame
model_comparison_df = pd.DataFrame.from_dict(pipe_dict).T
model_comparison_df.columns = ["Accuracy_score", "f1_score"]
model_comparison_df.sort_values(
    ["Accuracy_score", "f1_score"], ascending=False, inplace=True
)
model_comparison_df.style.background_gradient(cmap="GnBu")
