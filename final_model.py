from wine_functions import create_wine_dataframe, split_dataset
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

# fetch_wine_data()

wine = create_wine_dataframe()

train, train_labels, test, test_labels = split_dataset(wine)

cluster_attribs = list(range(8, 12))

cluster_pipeline = ColumnTransformer(
    [("kmeans", KMeans(n_clusters=2), cluster_attribs)],
    remainder="passthrough"
)

forest_clf = RandomForestClassifier(random_state=42, n_estimators=200)

final_classifier = Pipeline([
    ('std_scaler', StandardScaler()),
    ('col_cluster', cluster_pipeline),
    ('classifier', forest_clf),
])

final_classifier.fit(train, train_labels)

final_predictions = final_classifier.predict(test)

test_accuracy = accuracy_score(test_labels, final_predictions)
test_balanced_accuracy = balanced_accuracy_score(test_labels, final_predictions)
test_confusion_matrix = confusion_matrix(test_labels, final_predictions)

print("Final model accuracy score:\n", test_accuracy)
print("Final model balanced accuracy score:\n", test_balanced_accuracy)
print("Final model confusion matrix:\n", test_confusion_matrix)

# comparison to a classifier built on data without clustering:

comparison_classifier = Pipeline([
    ('std_scaler', StandardScaler()),
    ('classifier', forest_clf),
])

comparison_classifier.fit(train, train_labels)

comp_predictions = comparison_classifier.predict(test)

comp_accuracy = accuracy_score(test_labels, comp_predictions)
comp_balanced_accuracy = balanced_accuracy_score(test_labels, comp_predictions)
comp_confusion_matrix = confusion_matrix(test_labels, comp_predictions)

print("Comparison model accuracy score:\n", comp_accuracy)
print("Comparison model balanced accuracy score:\n", comp_balanced_accuracy)
print("Comparison model confusion matrix:\n", comp_confusion_matrix)

'''
The accuracy and balanced accuracy scores are slightly better for the classifier
that was trained using the clustered data compared to the classifier trained on data
without anything clustering, but the difference is minimal. Neither classifier
is very accurate, with their balanced accuracy scores being around 35%.
'''