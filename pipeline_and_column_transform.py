from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import random
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import  confusion_matrix
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

n = 300
level = ["high_school", "bachlor", "master", "phd"]

data = {
    "age" : np.random.randint(26 , 55 , n),
    "experience" : np.random.randint(4 , 15 , n),
    "education_level" : np.random.choice(level , n ),
    "hours_per_week" : np.random.randint(36 , 48 ,n),
    "city_tier" : np.random.randint(1 , 4 , n),
    "skills_score" : np.random.randint(5 , 10 , n),
    "projects_complete" : np.random.randint(7 , 17 , n)

}

df = pd.DataFrame(data)

df["salary>50k"] =(
    (df["experience"] > 6 ) &
    (df["education_level"].isin(["master","phd"])) &
    (df["skills_score"]> 7)&
    (df["projects_complete"] > 10)
).astype(int)

x = df.drop("salary>50k", axis=1)
y = df["salary>50k"]

x_train , x_test , y_train , y_test = train_test_split(x ,y , test_size=0.2 , random_state=42)

numerical_column = ["age","experience","hours_per_week","city_tier","skills_score","projects_complete"]
categorical_column = ["education_level"]

numeric_transform = Pipeline(steps=[
    ("scaler" ,  StandardScaler())
]   )


categorical_transform = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))

])

preprocesing = ColumnTransformer(
    transformers= [
        ("num" , numeric_transform , numerical_column),
        ("cat" , categorical_transform , categorical_column)
    ]
)

clf = Pipeline(steps=[
    ("preprocessing" , preprocesing),
    ("model", LogisticRegression())
])

clf.fit(x_train , y_train)
pred = clf.predict(x_test)

print(clf.score(x_test , y_test)*100)
print(confusion_matrix(y_test , pred))
print(clf.named_steps["model"].coef_)
