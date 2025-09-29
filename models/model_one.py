import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import graphviz

df = pd.read_csv('data/adult.csv')
#print(df.info())

df['sex'] = pd.factorize(df['sex'])[0]
df['race'], cat_races = pd.factorize(df['race'])
df['relationship'], cat_relations = pd.factorize(df['relationship'])
df['education'], cat_ed = pd.factorize(df['education'])
df['occupation'], cat_ocup = pd.factorize(df['occupation'])
df['workclass'] = pd.factorize(df['workclass'])[0]
y , cat_inc = pd.factorize(df['income'])
df.drop(['native.country','marital.status', 'education.num', 'income'], axis=1, inplace=True)

#print(cat_races, cat_relations)
print(df.info())

X_train, X_hold, y_train, y_hold = train_test_split(df.values, y, test_size=0.2, random_state=12)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_hold)
print(accuracy_score(y_hold, knn_pred))

tree = DecisionTreeClassifier (max_depth=5, random_state=12)
tree.fit (X_train, y_train)
tree_pred = tree.predict(X_hold)
print(accuracy_score(y_hold, tree_pred))

#grid tree
tree_params = {'max_depth': range(1,9), "max_features": range(4,10)}
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(X_train, y_train)
print(tree_grid.best_params_)
print(accuracy_score(y_hold, tree_grid.predict(X_hold)))

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
knn_params = {'knn__n_neighbors': range(1,10)}
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)
knn_grid.fit(X_train,y_train)
print(knn_grid.best_params_)
print(accuracy_score(y_hold, knn_grid.predict(X_hold)))

dot_data = export_graphviz(
    tree_grid.best_estimator_,
    out_file=None,
    feature_names=df.columns,
    class_names=['Class 0', 'Class 1'],  # замените на ваши классы
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("best_decision_tree")  # сохраняет как best_decision_tree.pdf
graph.view()
