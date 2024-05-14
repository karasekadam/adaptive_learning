import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import re
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor


def filter_out_img(df: pd.DataFrame) -> pd.DataFrame:
    rows_to_drop = []
    for i, row in df.iterrows():
        question = json.loads(row["question"])
        for question_part in question:
            if "img" == question_part[0]:
                rows_to_drop.append(i)
    df.drop(rows_to_drop, inplace=True)
    return df


def find_biggest_integer(text):
    # Find all integers in the text
    integers = re.findall(r'\b\d+\b', text)

    # If no integers are found, return None
    if not integers:
        return None

    # Convert the integers from strings to integers
    integers = list(map(int, integers))

    # Find the maximum integer
    max_integer = max(integers) ** (1/2)

    return max_integer


def parameters(df_data: pd.DataFrame):
    df_data["plus"] = df_data["question_text"].map(lambda x: x.count("+"))
    df_data["minus"] = df_data["question_text"].map(lambda x: x.count("-"))
    df_data["times"] = df_data["question_text"].map(lambda x: x.count("cdot"))
    df_data["div"] = df_data["question_text"].map(lambda x: x.count(":"))
    df_data["frac"] = df_data["question_text"].map(lambda x: x.count("frac"))
    df_data["question_len"] = df_data["question_text"].map(lambda x: len(x))
    df_data["combined_frac"] = df_data["question_text"].map(lambda x: 1 if re.search(r'\d+\\frac', x) else 0)
    df_data["frac_to_float"] = df_data["plus"] + df_data["minus"] + df_data["times"] + df_data["div"] + df_data["combined_frac"] + df_data["frac"]
    df_data["frac_to_float"] = df_data["frac_to_float"].map(lambda x: 1 if x == 1 else 0)
    df_data["var"] = df_data["question_text"].map(lambda x: x.count("x"))
    df_data["equation"] = df_data["question_text"].map(lambda x: 1 if "=" in x else 0)
    df_data["answer_len"] = df_data["answer_text"].map(lambda x: len(x))
    df_data["answer_float"] = df_data["answer_text"].map(lambda x: 1 if "," in x else 0)
    df_data["answer_frac"] = df_data["answer_text"].map(lambda x: 1 if "/" in x else 0)
    df_data["highest_integer"] = df_data["question_text"].map(find_biggest_integer)


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


def linear_regression_training(zlomky_data: pd.DataFrame, k_fold: int, predicted_out: list, real_values_out: list):
    sample_size = len(zlomky_data) // k_fold

    for i in range(k_fold):
        test_sample = zlomky_data.iloc[i*sample_size: (i+1)*sample_size]
        train_sample = zlomky_data.drop(test_sample.index)

        model_parameters = list(zlomky_data.columns)
        model_parameters.remove("errorRate")
        X_train = train_sample[model_parameters]
        y_train = train_sample["errorRate"]
        X_test = test_sample[model_parameters]
        y_test = test_sample["errorRate"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predicted_out.extend(predictions)
        real_values_out.extend(y_test)

    return model


def linear_regression(zlomky_data: pd.DataFrame, k_fold: int):
    zlomky_data = zlomky_data.sample(frac=1)
    predicted = []
    real_values = []

    model = linear_regression_training(zlomky_data, k_fold, predicted, real_values)

    r2 = r2_score(real_values, predicted)
    print(f"Linear regression R2 score: {r2}")
    corr = pearsonr(real_values, predicted)
    print(f"Linear regression Pearson correlation: {corr[0]}")
    mse = mean_squared_error(real_values, predicted)
    print(f"Linear regression MSE: {mse}")

    evaluations = [[r2, corr[0], mse]]
    evaluations_names = ["R2", "Pearson correlation", "MSE"]
    df_evaluations = pd.DataFrame(evaluations, columns=evaluations_names)
    # plt.bar(df_evaluations.columns, df_evaluations.iloc[0])
    # plt.title("Linear regression evaluation")
    # plt.tight_layout()
    # plt.show()

    #lr_betas = np.insert(model.coef_, 0, model.intercept_, axis=0)
    #df_features = pd.DataFrame(lr_betas, index=["intercept"] + model_parameters)
    #dataplot = sb.heatmap(df_features, annot=True, linewidths=0.5, center=0)
    #plt.title("Linear regression feature importance")
    #plt.tight_layout()
    # plt.show()
    return df_evaluations["Pearson correlation"], df_evaluations["MSE"]


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

        model = DecisionTreeRegressor(max_depth=7, min_samples_split=5)
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

    df_features = pd.DataFrame(model.feature_importances_, index=model_parameters)
    plt.bar(df_features.index, df_features.iloc[:, 0])
    plt.title("Regression tree feature importance")
    plt.ylabel("Information gain")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("regression_tree_feature_importance.png", dpi=300)
    plt.show()
    # dataplot = sb.heatmap(df_features, annot=True, linewidths=0.5, center=0)
    # plt.title("Regression tree feature importance")
    # plt.tight_layout()
    # plt.show()

    """fig = plt.figure(figsize=(40, 12))
    _ = tree.plot_tree(model,
                       feature_names=model_parameters,
                       filled=True,
                       fontsize=24)
    plt.tight_layout()
    plt.savefig("regression_tree.png", dpi=800)
    plt.show()"""

    return df_evaluations["Pearson correlation"], df_evaluations["MSE"]


def linear_regression_feature_importance(zlomky_data: pd.DataFrame):
    parameter_results = {}
    for parameter in model_parameters:
        parameter_data = zlomky_data[["errorRate", parameter]]
        predicted = []
        real_values = []
        linear_regression_training(parameter_data, 195, predicted, real_values)
        parameter_results[parameter] = pearsonr(real_values, predicted)[0]

    plt.bar(list(parameter_results.keys()), list(parameter_results.values()))
    plt.title("Linear regression feature importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("linear_regression_feature_importance.png", dpi=300)
    plt.show()


def count_parameters_occurence(df: pd.DataFrame):
    parameters_occurence = {}
    parameter_occurence_df = df.copy()
    for parameter in ["plus", "minus", "times", "div", "combined_frac", "frac_to_float", "equation", "var", "answer_float", "answer_frac"]:
        parameter_occurence_df[parameter] = parameter_occurence_df[parameter].map(lambda x: 1 if x > 0 else 0)
        parameters_occurence[parameter] = parameter_occurence_df[parameter].sum()

    plt.bar(list(parameters_occurence.keys()), list(parameters_occurence.values()))
    plt.title("Parameters occurrence in the dataset")
    plt.ylabel("Occurrence count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("parameters_occurence.png", dpi=300)
    plt.show()


def correlation_matrix(zlomky_data: pd.DataFrame):
    correlation_data = zlomky_data[model_parameters + ["errorRate"]]
    plt.figure(figsize=(10, 8))
    dataplot = sb.heatmap(correlation_data.corr(), cmap="vlag", annot=True, linewidths=0.5, center=0, fmt=".2f")
    plt.title("Features correlation matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=300)
    plt.show()


class BaselineModel:
    def __init__(self):
        self.data = None
        self.plus_error = None
        self.minus_error = None
        self.times_error = None
        self.div_error = None
        self.combined_frac_error = None
        self.frac_to_float_error = None

    def fit(self, data):
        self.data = data
        self.plus_error = self.data[self.data["plus"] > 0]["errorRate"].mean()
        self.minus_error = self.data[self.data["minus"] > 0]["errorRate"].mean()
        self.times_error = self.data[self.data["times"] > 0]["errorRate"].mean()
        self.div_error = self.data[self.data["div"] > 0]["errorRate"].mean()
        self.combined_frac_error = self.data[self.data["combined_frac"] > 0]["errorRate"].mean()
        self.frac_to_float_error = self.data[self.data["frac_to_float"] > 0]["errorRate"].mean()

    def predict(self, predict_data: pd.DataFrame):
        list_of_predictions = []
        for i, row in predict_data.iterrows():
            if row["plus"] > 0:
                list_of_predictions.append(self.plus_error)
            elif row["minus"] > 0:
                list_of_predictions.append(self.minus_error)
            elif row["times"] > 0:
                list_of_predictions.append(self.times_error)
            elif row["div"] > 0:
                list_of_predictions.append(self.div_error)
            elif row["combined_frac"] > 0:
                list_of_predictions.append(self.combined_frac_error)
            elif row["frac_to_float"] > 0:
                list_of_predictions.append(self.frac_to_float_error)
            else:
                list_of_predictions.append(self.data["errorRate"].mean())
        return list_of_predictions


def baseline_model_training(zlomky_data: pd.DataFrame, k_fold: int):
    zlomky_data = zlomky_data.sample(frac=1)
    sample_size = len(zlomky_data) // k_fold
    predicted = []
    real_values = []

    for i in range(k_fold):
        test_sample = zlomky_data.iloc[i * sample_size: (i + 1) * sample_size]
        train_sample = zlomky_data.drop(test_sample.index)

        X_train = train_sample[model_parameters]
        y_train = train_sample["errorRate"]
        X_test = test_sample[model_parameters]
        y_test = test_sample["errorRate"]

        model = BaselineModel()
        model.fit(train_sample)
        predictions = model.predict(test_sample)
        predicted.extend(predictions)
        real_values.extend(y_test)

    r2 = r2_score(real_values, predicted)
    print(f"Baseline model R2 score: {r2}")
    corr = pearsonr(real_values, predicted)
    print(f"Baseline model Pearson correlation: {corr[0]}")
    mse = mean_squared_error(real_values, predicted)
    print(f"Baseline model MSE: {mse}")

    evaluations = [[r2, corr[0], mse]]
    evaluations_names = ["R2", "Pearson correlation", "MSE"]
    df_evaluations = pd.DataFrame(evaluations, columns=evaluations_names)
    # dataplot = sb.heatmap(df_evaluations, annot=True, linewidths=0.5, center=0)
    # plt.title("Baseline model evaluation")
    # plt.tight_layout()
    # plt.show()

    return df_evaluations["Pearson correlation"], df_evaluations["MSE"]


def variance_by_question_type(zlomky_data: pd.DataFrame):
    question_types = ["plus", "minus", "times", "div", "combined_frac", "frac_to_float", "equation", "answer_float"]
    results = {}
    for question_type in question_types:
        question_data = zlomky_data[zlomky_data[question_type] > 0]
        results[question_type] = question_data['errorRate'].var()

    plt.bar(list(results.keys()), list(results.values()))
    plt.title("ErrorRate variance by question type")
    plt.ylabel("Variance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("variance_by_question_type.png", dpi=300)
    plt.show()


written_answers = pd.read_csv('mat-writtenAnswer.csv', sep=";")
system_resource_set = pd.read_csv('umimematikucz-system_resource_set.csv', sep=";")
system_kc = pd.read_csv('umimematikucz-system_kc.csv', sep=";")
word_levels = pd.read_csv('word_levels.csv', sep=";")

written_answers_merged = pd.merge(written_answers, system_resource_set, left_on="rs", right_on="id")
all_data = pd.merge(written_answers_merged, system_kc, left_on="parent", right_on="id", suffixes=("_resource", "_kc"))

zlomky_kc = [38, 55, 44, 43, 45, 172, 118, 111, 168, 41, 42, 329, 48, 149, 150, 51]
zlomky_data = filter_out_img(all_data)
zlomky_data["question_text"] = zlomky_data.apply(process_question_json, axis=1)
zlomky_data["answer_text"] = zlomky_data.apply(process_answer_json, axis=1)
all_zlomky = zlomky_data[zlomky_data["question_text"].str.contains("frac")]
kc = all_zlomky["id_kc"].unique()
zlomky_data = zlomky_data[zlomky_data["id_kc"].isin(zlomky_kc)]

columns_to_keep = ["resourceId", "rs", "question", "question_text", "answer_text", "explanation_x", "errorRate",
                   "responseTime", "answers"]
zlomky_data = zlomky_data[columns_to_keep]

parameters(zlomky_data)
original_data = zlomky_data.copy()
model_parameters = ["plus", "minus", "times", "div", "frac", "question_len", "answer_len",
                    "answer_float", "combined_frac", "frac_to_float", "highest_integer", "equation", "var",
                    "answer_frac"]
zlomky_data = zlomky_data[model_parameters + ["errorRate"]]
for parameter in model_parameters:
    zlomky_data[parameter] = MinMaxScaler().fit_transform(zlomky_data[[parameter]])

lg_corr, lg_mse = linear_regression(zlomky_data, 195)
tree_corr, tree_mse = regression_tree(zlomky_data, 195)
baseline_corr, baseline_mse = baseline_model_training(zlomky_data, 195)
plt.bar(["Linear regression", "Regression tree", "Baseline model"], [lg_corr[0], tree_corr[0], baseline_corr[0]])
plt.title("Model comparison - Pearson correlation")
plt.ylabel("Pearson correlation")
plt.tight_layout()
plt.savefig("model_comparison_corr.png", dpi=300)
plt.show()
plt.bar(["Linear regression", "Regression tree", "Baseline model"], [lg_mse[0], tree_mse[0], baseline_mse[0]])
plt.title("Model comparison - MSE")
plt.ylabel("Mean squared error")
plt.tight_layout()
plt.savefig("model_comparison_mse.png", dpi=300)
plt.show()

correlation_matrix(zlomky_data)

# linear_regression_feature_importance(zlomky_data)
# count_parameters_occurence(original_data)

variance_by_question_type(zlomky_data)
pass
