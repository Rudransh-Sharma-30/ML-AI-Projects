import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# survival_rate = df.groupby("Sex")["Survived"].mean()
# plt.bar(survival_rate.index, survival_rate.values)
# plt.xlabel("Sex")
# plt.ylabel("Survival Probability")
# plt.title("Survival Rate by Sex")
# plt.show()

# survival_rate = df.groupby("Pclass")["Survived"].mean()

# plt.bar(survival_rate.index, survival_rate.values)
# plt.xlabel("Passenger Class")
# plt.ylabel("Survival Probability")
# plt.title("Survival Rate by Passenger Class")
# plt.show()


# plt.hist(df[df["Survived"] == 1]["Age"], bins=30, alpha=0.6, label="Survived")
# plt.hist(df[df["Survived"] == 0]["Age"], bins=30, alpha=0.6, label="Not Survived")

# plt.xlabel("Age")
# plt.ylabel("Count")
# plt.title("Age Distribution by Survival")
# plt.legend()
# plt.show()

# plt.hist(df[df["Survived"] == 1]["Fare"], bins=30, alpha=0.6, label="Survived")
# plt.hist(df[df["Survived"] == 0]["Fare"], bins=30, alpha=0.6, label="Not Survived")

# plt.xlabel("Fare")
# plt.ylabel("Count")
# plt.title("Fare Distribution by Survival")
# plt.legend()
# plt.show()

# df["HasCabin"] = df["Cabin"].notnull().astype(int)

# cabin_rate = df.groupby("HasCabin")["Survived"].mean()
# plt.bar(cabin_rate.index, cabin_rate.values)
# plt.xticks([0,1],["No Cabin", "Has Cabin"])
# plt.xlabel("Cabin Availability")
# plt.ylabel("Survival Probability")
# plt.title("Survival Rate by Cabin Availability")
# plt.show()

# sibsp_rate = df.groupby("SibSp")["Survived"].mean()
# plt.bar(sibsp_rate.index, sibsp_rate.values)
# plt.xlabel("Number of Siblings / Spouses")
# plt.ylabel("Survival Probability")
# plt.title("Survival Rate vs SibSp")
# plt.show()

# parch_rate = df.groupby("Parch")["Survived"].mean()
# plt.bar(parch_rate.index, parch_rate.values)
# plt.xlabel("Number of Parents / Children")
# plt.ylabel("Survival Probability")
# plt.title("Survival Rate vs Parch")
# plt.show()

####COMBINING THEM GIVES US BETTER RESULT
# df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
# family_rate = df.groupby("FamilySize")["Survived"].mean()
# plt.bar(family_rate.index, family_rate.values)
# plt.xlabel("Family Size")
# plt.ylabel("Survival Probability")
# plt.title("Survival Rate vs Family Size")
# plt.show()



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# model = XGBClassifier(
#     n_estimators=150,
#     max_depth=5,
#     learning_rate=0.1,
#     subsample=1.0,
#     colsample_bytree=1.0,
#     random_state=42,
#     eval_metric="logloss"
# )




model = RandomForestClassifier(
    n_estimators=400,
    max_depth=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)



for df in [train_df, test_df]:
    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

features = ["Sex", "Pclass", "Age", "Fare", "FamilySize", "HasCabin"]

X_train = train_df[features]
y_train = train_df["Survived"]
X_test = test_df[features]

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)


X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


# model = LogisticRegression(max_iter=1000, penalty = "l2", C = 1.0, solver = "lbfgs")
model.fit(X_train, y_train)


pred = model.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": pred.astype(int)
})

submission.to_csv("submission.csv", index=False)


print(submission)


######## FOR ACCURACY
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state= 1)
model.fit(X_tr,y_tr)
pred1 = model.predict(X_val)
acc = accuracy_score(y_val, pred1)

print("Validation Accuracy:", acc)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("CV Accuracy:", scores.mean())


