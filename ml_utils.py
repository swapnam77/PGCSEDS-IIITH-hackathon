from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pandas import read_csv
import h2o
from h2o.automl import H2OAutoML
import subprocess

# define a Gaussain NB classifier
clf = RandomForestClassifier(max_depth=3, random_state=0)

# define the class encodings and reverse encodings
classes = {0: False, 1: True}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    # X, y = datasets.load_iris(return_X_y=True)
    df=read_csv("dataset/bug_pred.csv", sep=",")
    X = df.drop("defects", axis=1)
    y = df["defects"]


    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Model trained with accuracy: {round(acc, 3)}")
    subprocess.call(["jupyter","nbconvert","--to","notebook","--inplace","--execute","dataset/explainable_AI_starter.ipynb"])
    subprocess.call(["jupyter","nbconvert","dataset/explainable_AI_starter.ipynb","--no-input","--to","html"])
    print("Explainability file generated")

# def load_explainability():



# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

def explainability():
    h2o.init()

    # Import wine quality dataset
    f = "dataset/bug_pred.csv"
    df = h2o.import_file(f)

    # Reponse column
    y = "defects"

    # Split into train & test
    splits = df.split_frame(ratios = [0.8], seed = 1)
    train = splits[0]
    test = splits[1]

    # Run AutoML for 1 minute
    aml = H2OAutoML(max_runtime_secs=60, seed=1)
    aml.train(y=y, training_frame=train)

    # Explain leader model & compare with all AutoML models
    # exa = aml.explain(test)

    # Explain a single H2O model (e.g. leader model from AutoML)
    exm = aml.leader.explain(test)
    return exm

# function to retrain the model as part of the feedback loop
# def retrain(data):
#     # pull out the relevant X and y from the FeedbackIn object
#     X = [list(d.dict().values())[:-1] for d in data]
#     y = [r_classes[d.flower_class] for d in data]

#     # fit the classifier again based on the new data obtained
#     clf.fit(X, y)
