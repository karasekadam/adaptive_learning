import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as mp
# from simpletransformers.classification import ClassificationModel, ClassificationArgs
from scipy.stats import pearsonr
from statistics import mean
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

written_answers = pd.read_csv('mat-writtenAnswer.csv', sep=";")
system_resource_set = pd.read_csv('umimematikucz-system_resource_set.csv', sep=";")
system_kc = pd.read_csv('umimematikucz-system_kc.csv', sep=";")
word_levels = pd.read_csv('word_levels.csv', sep=";")

written_answers_merged = pd.merge(written_answers, system_resource_set, left_on="rs", right_on="id")
all_data = pd.merge(written_answers_merged, system_kc, left_on="parent", right_on="id", suffixes=("_resource", "_kc"))


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


def question_type():
    question_types = written_answers_merged.groupby("shortcut").count()
    # plot the question types
    plt.bar(question_types.index, question_types["resourceId"])
    print(sum(question_types["resourceId"]))
    plt.show()


zlomky_kc = [38, 55, 44, 43, 45, 172, 118, 111, 168, 41, 42, 329]
zlomky_data = all_data[all_data["id_kc"].isin(zlomky_kc)]


def filter_out_img(df: pd.DataFrame) -> pd.DataFrame:
    rows_to_drop = []
    for i, row in df.iterrows():
        question = json.loads(row["question"])
        for question_part in question:
            if "img" == question_part[0]:
                rows_to_drop.append(i)
    df.drop(rows_to_drop, inplace=True)
    return df


zlomky_data = filter_out_img(zlomky_data)
# question_type()


def parameters(df_data: pd.DataFrame):
    df_data["plus"] = df_data["question_text"].map(lambda x: x.count("+"))
    df_data["minus"] = df_data["question_text"].map(lambda x: x.count("-"))
    df_data["times"] = df_data["question_text"].map(lambda x: x.count("cdot"))
    df_data["div"] = df_data["question_text"].map(lambda x: x.count(":"))
    df_data["frac"] = df_data["question_text"].map(lambda x: x.count("frac"))
    df_data["question_len"] = df_data["question_text"].map(lambda x: len(x))
    df_data["answer_len"] = df_data["answer_text"].map(lambda x: len(x))
    df_data["answer_float"] = df_data["answer_text"].map(lambda x: "/" in x)


def merge_question(df_data: pd.DataFrame):
    question = ""
    for question_part in df_data["question"]:
        question += question_part[1]


def process_question_json(row):
    question_stream = row["question"]
    question = json.loads(question_stream)
    question_text = ""
    for question_part in question:
        if question_part[0] == "latex":
            question_text += " " + question_part[1]
    return question_text


def process_answer_json(row):
    answer_stream = row["question"]
    answer = json.loads(answer_stream)
    answer_text = ""
    for answer_part in answer:
        if answer_part[0] == "input":
            answer_text += " " + answer_part[1]["answer"][0]
    return answer_text


zlomky_data["question_text"] = zlomky_data.apply(process_question_json, axis=1)
zlomky_data["answer_text"] = zlomky_data.apply(process_answer_json, axis=1)

columns_to_keep = ["resourceId", "rs", "question", "question_text", "answer_text", "explanation_x", "errorRate", "responseTime"]
zlomky_data = zlomky_data[columns_to_keep]

parameters(zlomky_data)


def linear_regression(zlomky_data: pd.DataFrame, k_fold: int):
    zlomky_data = zlomky_data.sample(frac=1)
    sample_size = len(zlomky_data) // k_fold
    predicted = []
    real_values = []

    for i in range(k_fold):
        test_sample = zlomky_data.iloc[i*sample_size: (i+1)*sample_size]
        train_sample = zlomky_data.drop(test_sample.index)

        X_train = train_sample[model_parameters]
        y_train = train_sample["errorRate"]
        X_test = test_sample[model_parameters]
        y_test = test_sample["errorRate"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predicted.extend(predictions)
        real_values.extend(y_test)

    r2 = r2_score(real_values, predicted)
    print(f"Linear regression R2 score: {r2}")
    corr = pearsonr(real_values, predicted)
    print(f"Linear regression Pearson correlation: {corr[0]}")
    mse = mean_squared_error(real_values, predicted)
    print(f"Linear regression MSE: {mse}")

    evaluations = [[r2, corr[0], mse]]
    evaluations_names = ["R2", "Pearson correlation", "MSE"]
    df_evaluations = pd.DataFrame(evaluations, columns=evaluations_names)
    dataplot = sb.heatmap(df_evaluations, annot=True, linewidths=0.5)
    mp.title("Linear regression evaluation")
    mp.tight_layout()
    mp.show()

    lr_betas = np.insert(model.coef_, 0, model.intercept_, axis=0)
    df_features = pd.DataFrame(lr_betas, index=["intercept"] + model_parameters)
    dataplot = sb.heatmap(df_features, annot=True, linewidths=0.5)
    mp.title("Linear regression feature importance")
    mp.tight_layout()
    mp.show()


def regression_tree(zlomky_data: pd.DataFrame, k_fold: int):
    zlomky_data = zlomky_data.sample(frac=1)
    sample_size = len(zlomky_data) // k_fold
    predicted = []
    real_values = []

    for i in range(k_fold):
        test_sample = zlomky_data.iloc[i*sample_size: (i+1)*sample_size]
        train_sample = zlomky_data.drop(test_sample.index)

        X_train = train_sample[model_parameters]
        y_train = train_sample["errorRate"]
        X_test = test_sample[model_parameters]
        y_test = test_sample["errorRate"]

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predicted.extend(predictions)
        real_values.extend(y_test)

    r2 = r2_score(real_values, predicted)
    print(f"Regression tree R2 score: {r2}")
    corr = pearsonr(real_values, predicted)
    print(f"Regression tree Pearson correlation: {corr[0]}")
    mse = mean_squared_error(real_values, predicted)
    print(f"Linear regression MSE: {mse}")

    evaluations = [[r2, corr[0], mse]]
    evaluations_names = ["R2", "Pearson correlation", "MSE"]
    df_evaluations = pd.DataFrame(evaluations, columns=evaluations_names)
    dataplot = sb.heatmap(df_evaluations, annot=True, linewidths=0.5)
    mp.title("Regression tree evaluation")
    mp.tight_layout()
    mp.show()

    df_features = pd.DataFrame(model.feature_importances_, index=model_parameters)
    dataplot = sb.heatmap(df_features, annot=True, linewidths=0.5)
    mp.title("Regression tree feature importance")
    mp.tight_layout()
    mp.show()


data = pd.read_csv("mat-writtenAnswer.csv", sep=";")


model_parameters = ["plus", "minus", "times", "div", "frac", "question_len", "answer_len", "answer_float"]
for parameter in model_parameters:
    zlomky_data[parameter] = MinMaxScaler().fit_transform(zlomky_data[[parameter]])

linear_regression(zlomky_data, 195)
regression_tree(zlomky_data, 195)

correlation_data = zlomky_data[["plus", "minus", "times", "div", "frac", "question_len", "answer_len", "answer_float", "errorRate"]]
dataplot = sb.heatmap(correlation_data.corr(), cmap="vlag", annot=True, linewidths=0.5, center=0)
mp.title("Features correlation matrix")
mp.tight_layout()
mp.show()
