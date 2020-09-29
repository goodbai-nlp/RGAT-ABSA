# coding:utf-8
import json

tokstr = "We had some scallops as well as the cheese plate"
pos = "PRP VBD DT NNS RB RB IN DT NN NN"
head = [2, 0, 4, 2, 4, 5, 5, 10, 10, 4]
dep = ['nsubj', 'root', 'det', 'dobj', 'cc', 'mwe', 'mwe', 'det', 'compound', 'conj']

exp1 = {
        "token":tokstr.split(),
        "pos":pos.split(),
        "head": head,
        "deprel":dep,
        "aspects":[{"term":["scallops"], "from":3, "to":4, "polarity":"neutral"}]
    }


tokstr2 = "Sales of Apple overtake Banana in 2012"
pos2 = "NNS IN NNP VBD NNP IN CD"
head2 = [4, 3, 1, 0, 4, 7, 5]
dep2 = ['nsubj', 'case', 'nmod', 'root', 'dobj', 'case', 'nmod']


exp2 = {
        "token":tokstr2.split(),
        "pos":pos2.split(),
        "head":head2,
        "deprel":dep2,
        "aspects":[{"term":["Apple"], "from":2, "to":3, "polarity":"positive"}]
        }

tokstr3 = "Sales of banana overtake apple in 2012"
pos3 = "NNS IN NNP VBD NNP IN CD"
head3 = [4, 3, 1, 0, 4, 7, 5]
dep3 = ['nsubj', 'case', 'nmod', 'root', 'dobj', 'case', 'nmod']


exp3 = {
        "token":tokstr3.split(),
        "pos":pos3.split(),
        "head":head3,
        "deprel":dep3,
        "aspects":[{"term":["apple"], "from":4, "to":5, "polarity":"negative"}]
    }
res = [exp1, exp2, exp3]
res = [exp1, exp2, exp3, exp3]
res = [exp1, exp2, exp3]
res = [exp1]
with open('test-new.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4)
