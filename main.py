import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# from simpletransformers.classification import ClassificationModel, ClassificationArgs

written_answers = pd.read_csv('mat-writtenAnswer.csv', sep=";")
system_resource_set = pd.read_csv('umimematikucz-system_resource_set.csv', sep=";")
system_kc = pd.read_csv('umimematikucz-system_kc.csv', sep=";")
word_levels = pd.read_csv('word_levels.csv', sep=";")


def parse_answer(x):
    data = json.loads(x)
    if len(data) <= 1:
        return None
    else:
        return data[1]


def parse_third(x):
    data = json.loads(x)
    if len(data) <= 2:
        return None
    else:
        return data[2]


written_answers["real_question"] = written_answers["question"].map(lambda x: json.loads(x)[0])
written_answers["answer"] = written_answers["question"].map(parse_answer)
written_answers["question_len"] = written_answers["real_question"].map(lambda x: len(str(x[1])))
written_answers["answer_len"] = written_answers["answer"].map(lambda x: len(str(x)) if x is not None else 0)
df_train, df_test = train_test_split(written_answers, test_size=0.2, random_state=42)


def question_len():
    X_train = df_train[["question_len"]]
    y_train = df_train["errorRate"]
    X_test = df_test[["question_len"]]
    y_test = df_test[["errorRate"]]

    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_test, y_test))

    x_points = list(df_train["question_len"])
    y_points = list(y_train)
    plt.plot(x_points, y_points, ".")

    x = [0, 250]
    y = [reg.intercept_, reg.intercept_ + reg.coef_[0]*250]
    plt.plot(x, y, "k-")
    plt.show()


def answer_len():
    X_train = df_train[["answer_len"]]
    y_train = df_train["errorRate"]
    X_test = df_test["answer_len"]
    y_test = df_test["errorRate"]

    reg = LinearRegression().fit(X_train, y_train)

    x_points = list(df_train["answer_len"])
    y_points = list(y_train)
    plt.plot(x_points, y_points, ".")

    x = [0, 250]
    y = [reg.intercept_, reg.intercept_ + reg.coef_[0] * 250]
    plt.plot(x, y, "k-")
    plt.show()


question_len()
"""# Enabling regression
# Setting optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs = 1
model_args.regression = True

# Create a ClassificationModel
model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=1,
    use_cuda=False,
    args=model_args
)

# Train the model
model.train_model(df_train)"""
