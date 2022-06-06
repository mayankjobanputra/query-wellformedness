from pydash.arrays import chunk
from pydash.collections import flat_map
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
# from transformers import DistilBertTokenizerFast as Tokenizer, DistilBertForSequenceClassification as Classifier, TextClassificationPipeline
from transformers import BertTokenizerFast as Tokenizer, BertForSequenceClassification as Classifier, TextClassificationPipeline
from train_classifier import read_qw_data

TOKENIZER = '{}bert-base-uncased'
MODEL = 'qw_{}bert-base'


if __name__ == "__main__":
    BATCH_SIZE = 64
    tokenizer = Tokenizer.from_pretrained(TOKENIZER.format(''))

    tst_texts, tst_labels = read_qw_data('test.tsv')
    tst_encodings = tokenizer(tst_texts, truncation=True, padding=True)

    model = Classifier.from_pretrained("model/"+MODEL.format(''))
    label_map = lambda x: model.config.label2id[x['label']]
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    preds = []
    for text_chunk in tqdm(chunk(tst_texts, BATCH_SIZE)):
        res = classifier(text_chunk)
        preds.extend(flat_map(res, label_map))

    print(precision_recall_fscore_support(tst_labels, preds, average='binary'))
    print(confusion_matrix(tst_labels, preds))
    print(accuracy_score(tst_labels, preds))
