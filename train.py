import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset/placement.csv")


df = df.drop(columns=["student_id", "salary_package_lpa"])


df["placement_status"] = df["placement_status"].map({
    "Placed": 1,
    "Not Placed": 0
})


df["total_skills"] = (
    df["coding_skill_score"] +
    df["logical_reasoning_score"] +
    df["aptitude_score"]
)

df["activity_score"] = (
    df["projects_count"] +
    df["internships_count"] +
    df["github_repos"]
)

target = "placement_status"

X = df.drop(columns=[target])
y = df[target]


X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_acc)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)
print("XGBoost Accuracy:", xgb_acc)

if xgb_acc > rf_acc:
    print("Best Model: XGBoost")
    best_model = xgb_model
else:
    print("Best Model: Random Forest")
    best_model = rf_model


y_pred = best_model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


importances =best_model.feature_importances_

for i, col in enumerate(X.columns):
    print(col, importances[i])


import joblib

joblib.dump(best_model, "model/best_model.pkl")
joblib.dump(X.columns, "model/model_columns.pkl")