import json
import jsonlines

from collections import Counter
from pydash.collections import flat_map
from tqdm import tqdm
# from transformers import DistilBertTokenizerFast as Tokenizer, DistilBertForSequenceClassification as Classifier, TextClassificationPipeline
from transformers import BertTokenizerFast as Tokenizer, BertForSequenceClassification as Classifier, TextClassificationPipeline

TOKENIZER = '{}-base-uncased'
MODEL = 'qw_{}-base'


def get_fever_preds(ip_path, out_path, classifier, label_map):
    preds = []
    writer = jsonlines.open(out_path, mode='w')

    with jsonlines.open(ip_path, mode='r') as r:
        for data in tqdm(r):
            # q_encoding = tokenizer(data['question_text'], truncation=True, padding=True)
            res = classifier(data['question_text'])
            preds.extend(flat_map(res, label_map))
            data['is_wellformed'] = flat_map(res, label_map)[0]
            writer.write(data)
    writer.close()
    return preds


def get_paper_preds(ip_path, out_path, classifier, label_map):
    preds = []
    f = open(ip_path, 'r')
    s_data = json.loads(f.read())
    data = s_data['data']
    for d in tqdm(data):
        for para in d['paragraphs']:
            for q_data in para['qas']:
                res = classifier(q_data['question'])
                preds.extend(flat_map(res, label_map))
                q_data['is_wellformed'] = flat_map(res, label_map)[0]

    s_data['data'] = data
    with open(out_path, 'w') as writer:
        writer.write(json.dumps(s_data))
    return preds


if __name__ == "__main__":
    BASE_MODEL = "bert"
    SPLIT = 'test'
    IP_PATH = "data/{}/{}.json"
    OP_PATH = "data/results/{}/{}/{}.json"

    tokenizer = Tokenizer.from_pretrained(TOKENIZER.format(BASE_MODEL))
    model = Classifier.from_pretrained("model/" + MODEL.format(BASE_MODEL))

    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)
    label_map = lambda x: model.config.label2id[x['label']]

    DATASET = 'fever'
    fever_preds = get_fever_preds(IP_PATH.format(DATASET, SPLIT), OP_PATH.format(BASE_MODEL, DATASET, SPLIT), classifier, label_map)
    print("FEVER Preds: {}".format(Counter(fever_preds)))

    DATASET = 'uqa'
    uqa_preds = get_paper_preds(IP_PATH.format(DATASET, SPLIT), OP_PATH.format(BASE_MODEL, DATASET, SPLIT), classifier, label_map)
    print("UQA Preds: {}".format(Counter(uqa_preds)))

    DATASET = 'squad'
    squad_preds = get_paper_preds(IP_PATH.format(DATASET, SPLIT), OP_PATH.format(BASE_MODEL, DATASET, SPLIT), classifier, label_map)
    print("SQuAD Preds: {}".format(Counter(squad_preds)))
