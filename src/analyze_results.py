import csv
import json
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string

from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm


def _get_default_qword_dict():
    return {'who': 0, 'where': 0, 'what': 0, 'when': 0, 'how much': 0, 'how many': 0, 'unk': 0}


def _add_qword_freq(que, qword_dict):
    q = que.translate(str.maketrans('', '', string.punctuation))
    q = q.split()[:2]
    if q[0] in qword_dict.keys():
        qword_dict[q[0]] += 1
    elif " ".join(q) in qword_dict.keys():
        qword_dict[" ".join(q)] += 1
    else:
        # print("unk q: {}".format(q))
        qword_dict['unk'] += 1


def _get_qw_qword_dist(f_path):
    wellf_qword_dict = _get_default_qword_dict()
    malf_qword_dict = _get_default_qword_dict()
    f_path = Path(f_path)
    with open(f_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if float(row['score']) < 0.5:
                _add_qword_freq(row['question'].lower(), malf_qword_dict)
            else:
                _add_qword_freq(row['question'].lower(), wellf_qword_dict)
    return wellf_qword_dict, malf_qword_dict


def _get_paper_qword_dist(f_path):
    wellf_qword_dict = _get_default_qword_dict()
    malf_qword_dict = _get_default_qword_dict()
    f = open(f_path, 'r')
    data = json.loads(f.read())['data']
    for d in tqdm(data):
        for para in d['paragraphs']:
            for q_data in para['qas']:
                if q_data['is_wellformed']:
                    _add_qword_freq(q_data['question'].lower(), wellf_qword_dict)
                else:
                    _add_qword_freq(q_data['question'].lower(), malf_qword_dict)
    return wellf_qword_dict, malf_qword_dict


def _get_fever_qword_dist(f_path):
    wellf_qword_dict = _get_default_qword_dict()
    malf_qword_dict = _get_default_qword_dict()
    with jsonlines.open(f_path, mode='r') as r:
        for data in tqdm(r):
            if data["is_wellformed"]:
                _add_qword_freq(data['question_text'].lower(), wellf_qword_dict)
            else:
                _add_qword_freq(data['question_text'].lower(), malf_qword_dict)
    return wellf_qword_dict, malf_qword_dict


def get_qword_distribution(f_path, dataset):
    if dataset == 'qw':
        return _get_qw_qword_dist(f_path)
    elif dataset in ['squad', 'uqa']:
        return _get_paper_qword_dist(f_path)
    elif dataset == 'fever':
        return _get_fever_qword_dist(f_path)
    else:
        raise NotImplementedError(f"{dataset} is not a valid input")


def get_qw_lens(f_path):
    f_path = Path(f_path)
    lens, labels = [], []
    with open(f_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            lens.append(len(row['question'].split()))
            labels.append(0 if float(row['score']) < 0.5 else 1)

    assert len(lens) == len(labels)
    return lens, labels


def get_paper_lens(path):
    lens, labels = [], []
    f = open(path, 'r')
    data = json.loads(f.read())['data']
    for d in tqdm(data):
        for para in d['paragraphs']:
            for q_data in para['qas']:
                labels.append(q_data['is_wellformed'])
                lens.append(len(q_data['question'].split()))

    assert len(lens) == len(labels)
    return lens, labels


def get_fever_lens(path):
    lens, labels = [], []

    with jsonlines.open(path, mode='r') as r:
        for data in tqdm(r):
            labels.append(data["is_wellformed"])
            lens.append(len(data['question_text'].split()))

    assert len(lens) == len(labels)
    return lens, labels


def create_df(lens, labels, dataset_name):
    return pd.DataFrame(zip(lens, labels), columns=[f"{dataset_name}_wc", f"{dataset_name}_label"])


def get_hist(df, bins, dataset_name):
    return list(map(lambda label: np.histogram(df.query(f"{dataset_name}_label == {label}")[f"{dataset_name}_wc"], bins=bins)[0], [0, 1]))


def create_df_hist(lens, labels, dataset_name):
    bins = list(range(0, 45, 5))
    df = create_df(lens, labels, dataset_name)
    mal, well = get_hist(df, bins, DATASET)
    hist_df = pd.DataFrame(np.hstack([mal, well]), columns=["freq"])
    hist_df["dataset"] = DATASET
    hist_df["label"] = ["mal"] * (hist_df.shape[0] // 2) + ["well"] * (hist_df.shape[0] // 2)
    hist_df["bins"] = bins[1:] * 2
    hist_df["freq"] = (hist_df["freq"]-hist_df["freq"].min())/(hist_df["freq"].max()-hist_df["freq"].min())
    return hist_df


def plot_stacked_hist(dfs):
    fig, ax = plt.subplots()
    labels = dfs[0]['bins']
    width = 0.2
    x_labels = np.arange(len(labels)/2)

    mal_df = dfs[0].query("label == 'mal'")
    well_df = dfs[0].query("label == 'well'")
    qw_bar = ax.bar(x_labels - 1.5*width, well_df["freq"].values, width,
                    label=dfs[0]['dataset'].unique()[0]+"_wellformed", color='yellowgreen')
    qw_bar = ax.bar(x_labels - 1.5*width, mal_df["freq"].values, width, label=dfs[0]['dataset'].unique()[0]+"_malformed",
                    bottom=well_df["freq"].values, color='salmon')

    mal_df = dfs[1].query("label == 'mal'")
    well_df = dfs[1].query("label == 'well'")
    squad_bar = ax.bar(x_labels - 0.5*width, well_df["freq"].values, width,
                       label=dfs[1]['dataset'].unique()[0]+"_wellformed", color='olivedrab')
    squad_bar = ax.bar(x_labels - 0.5*width, mal_df["freq"].values, width, label=dfs[1]['dataset'].unique()[0]+"_malformed",
                       bottom=well_df["freq"].values, color='orangered')

    mal_df = dfs[2].query("label == 'mal'")
    well_df = dfs[2].query("label == 'well'")
    fever_bar = ax.bar(x_labels + 0.5*width, well_df["freq"].values, width,
                       label=dfs[2]['dataset'].unique()[0]+"_wellformed", color='darkolivegreen')
    fever_bar = ax.bar(x_labels + 0.5*width, mal_df["freq"].values, width, label=dfs[2]['dataset'].unique()[0]+"_malformed",
                       bottom=well_df["freq"].values, color='firebrick')
    #
    mal_df = dfs[3].query("label == 'mal'")
    well_df = dfs[3].query("label == 'well'")
    uqa_bar = ax.bar(x_labels + 1.5*width, well_df["freq"].values, width,
                     label=dfs[3]['dataset'].unique()[0]+"_wellformed", color='darkgreen')
    uqa_bar = ax.bar(x_labels + 1.5*width, mal_df["freq"].values, width, label=dfs[3]['dataset'].unique()[0]+"_malformed",
                     bottom=well_df["freq"].values, color='maroon')

    ax.set_ylabel('Relative Frequency of wellformed vs malformed questions')
    ax.set_xlabel('Word count')
    ax.set_xticks(x_labels, x_labels*5+5)
    ax.legend()
    plt.show()


# def merge_df(dfs):
#     merged = pd.merge(dfs[0], dfs[1], how="outer")
#     for i in range(2, len(dfs)):
#         merged = pd.merge(merged, dfs[i], how="outer")
#     return merged
#
#
# def plot_hist(df):
#     fig, ax = plt.subplots()
#     colors = ['b', 'o', 'g', 'r']
#     mal_df = df.query("label == 'mal'")
#     well_df = df.query("label == 'well'")
#     print(mal_df)
#     print(well_df)
#
#     # sns.barplot(data=combined_hist_df, x="bins", y="freq", hue="dataset", hue_order=['squad', 'fever', 'uqa', 'qw'], ci=None, ax=ax)
#     # sns.set_color_codes("muted")
#     sns.set_palette("tab10")
#
#     sns.barplot(data=well_df, x="bins", y="freq", hue="dataset", hue_order=['squad', 'fever', 'uqa', 'qw'],
#                 ci=None, ax=ax, alpha=0.8)
#     sns.barplot(data=mal_df, x="bins", y="freq", hue="dataset", hue_order=['squad', 'fever', 'uqa', 'qw'],
#                 ci=None, ax=ax, alpha=0.2)
#     ax.set_yscale("log")
#
#     plt.show()


if __name__ == "__main__":
    BASE_MODEL = "bert"
    SPLIT = 'test'
    IP_PATH = "/home/monk/Projects/query-wellformedness/data/results/{}/{}/{}.json"

    DATASET = 'qw'
    qw_lens, qw_labels = get_qw_lens("/home/monk/Projects/query-wellformedness/test.tsv")
    print(f"{DATASET} len-to-wellformedness corr: {pearsonr(qw_lens, qw_labels)}")
    qw_df = create_df_hist(qw_lens, qw_labels, DATASET)
    print("="*10 + DATASET +"="*10)
    print(get_qword_distribution("/home/monk/Projects/query-wellformedness/test.tsv", DATASET))

    DATASET = 'squad'
    squad_lens, squad_labels = get_paper_lens(IP_PATH.format(BASE_MODEL, DATASET, SPLIT))
    print(f"{DATASET} len-to-wellformedness corr: {pearsonr(squad_lens, squad_labels)}")
    squad_df = create_df_hist(squad_lens, squad_labels, DATASET)
    print("=" * 10 + DATASET + "=" * 10)
    print(get_qword_distribution(IP_PATH.format(BASE_MODEL, DATASET, SPLIT), DATASET))

    DATASET = "fever"
    fever_lens, fever_labels = get_fever_lens(IP_PATH.format(BASE_MODEL, DATASET, SPLIT))
    print(f"{DATASET} len-to-wellformedness corr: {pearsonr(fever_lens, fever_labels)}")
    fever_df = create_df_hist(fever_lens, fever_labels, DATASET)
    print("=" * 10 + DATASET + "=" * 10)
    print(get_qword_distribution(IP_PATH.format(BASE_MODEL, DATASET, SPLIT), DATASET))

    DATASET = "uqa"
    uqa_lens, uqa_labels = get_paper_lens(IP_PATH.format(BASE_MODEL, DATASET, SPLIT))
    print(f"{DATASET} len-to-wellformedness corr: {pearsonr(uqa_lens, uqa_labels)}")
    uqa_df = create_df_hist(uqa_lens, uqa_labels, DATASET)
    print("=" * 10 + DATASET + "=" * 10)
    print(get_qword_distribution(IP_PATH.format(BASE_MODEL, DATASET, SPLIT), DATASET))

    # print(qw_df.shape, squad_df.shape, fever_df.shape, uqa_df.shape)
    # print(qw_df, squad_df, fever_df, uqa_df)
    dfs = [qw_df, squad_df, fever_df, uqa_df]
    #
    plot_stacked_hist(dfs)


    # combined_hist_df = merge_df([qw_df, squad_df, fever_df, uqa_df])
    # print(combined_hist_df.shape)
    # plot_hist(combined_hist_df)

    # sns.barplot(data=combined_hist_df, x="bins", y="freq", hue="dataset", ci=None)

    # all_lens = [squad_lens, fever_lens, uqa_lens, qw_lens]
    # all_labels = [squad_labels, fever_labels, uqa_labels, qw_labels]

    # plt.hist(all_lens, histtype='bar', label=["SQuAD, Avg: {:.2f}".format(np.average(squad_lens)),
    #                                           "FEVER, Avg: {:.2f}".format(np.average(fever_lens)),
    #                                           "UQA,    Avg: {:.2f}".format(np.average(uqa_lens)),
    #                                           "QW,     Avg: {:.2f}".format(np.average(qw_lens))], log=True)
    # # plt.yscale('log')
    # plt.legend(prop={'size': 12})
    # plt.xlabel("Word Count")
    # plt.ylabel("#Questions")
    # plt.show()
