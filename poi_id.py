#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import matplotlib.pyplot
from sklearn import tree
from sklearn.cross_validation import StratifiedShuffleSplit
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
dd = pd.DataFrame(data_dict).T
print dd.info()
print dd['poi'].value_counts()
### Select what features you'll use.

features_list1 = ["poi", "from_this_person_to_poi", "from_poi_to_this_person"]

features = ["salary", "bonus"]

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
### Remove outliers
data_dict.pop("TOTAL",0)
data_dict.pop("TRAVEL AGENCY IN THE PARK",0) 
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

data = featureFormat(data_dict, features_list1)

for point in data:
    from_poi = point[1]
    to_poi = point[2]
    if point[0] == 1:
        matplotlib.pyplot.scatter(from_poi, to_poi, color="r")
    else:
        matplotlib.pyplot.scatter(from_poi, to_poi, color="b")

matplotlib.pyplot.xlabel("from_this_person_to_poi")
matplotlib.pyplot.ylabel("from_poi_to_this_person")
matplotlib.pyplot.show()
### Create new feature(s) & Store to my_dataset for easy export below.
my_dataset = data_dict

def newfeature( poi_messages, messages ):
    fraction_list = []
    for i in my_dataset:
        if my_dataset[i][messages] == "NaN" or my_dataset[i][poi_messages] == "NaN":
            fraction_list.append(0.)
        else:
            fraction_list.append(float(my_dataset[i][poi_messages])/float(my_dataset[i][messages]))
    return fraction_list

fraction_from_poi_email=newfeature("from_poi_to_this_person","to_messages")
fraction_to_poi_email=newfeature("from_this_person_to_poi","from_messages")

### feature scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
fraction_from_poi_email=scaler.fit_transform(fraction_from_poi_email)
fraction_to_poi_email=scaler.fit_transform(fraction_to_poi_email)


n= 0
for i in my_dataset:
    my_dataset[i]["fraction_from_poi_email"]=fraction_from_poi_email[n]
    my_dataset[i]["fraction_to_poi_email"]=fraction_to_poi_email[n]
    n += 1
### final features_list

features_list2 = ["poi", "salary", "bonus", 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
data = featureFormat(my_dataset, features_list2)
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    if point[0] == 1:
        matplotlib.pyplot.scatter(from_poi, to_poi, color="r")
    else:
        matplotlib.pyplot.scatter(from_poi, to_poi, color="b")
matplotlib.pyplot.xlabel("fraction_from_poi_email")
matplotlib.pyplot.ylabel("fraction_to_poi_email")
matplotlib.pyplot.show()

### Extract features and labels from dataset for local testing
labels, features = targetFeatureSplit(data)

### Try a varity of classifiers
from sklearn import cross_validation
from time import time
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
t0=time()
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif,k=10)
selector.fit(features_train,labels_train)
features_train = selector.transform(features_train)
features_test = selector.transform(features_test)
clf = tree.DecisionTreeClassifier()
clf= clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("Accuracy: ",accuracy_score(labels_test, pred))
print("Precision: ",precision_score(labels_test, pred))
print("Recall: ",recall_score(labels_test, pred))
imp = clf.feature_importances_
fea_imp = dict(zip(features_list2, imp))
print {k:v for k,v in fea_imp.iteritems() if v>0.1}
print("Decision tree algorithm time:",round(time()-t0, 3),"s")


features_list3 = ["poi",'salary','loan_advances', 'from_this_person_to_poi', 'deferral_payments']
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"]
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

### Try a varity of classifiers
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
t0=time()
clf = tree.DecisionTreeClassifier()
clf= clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score
print("Accuracy: ",accuracy_score(labels_test, pred))
print("Precision: ",precision_score(labels_test, pred))
print("Recall: ",recall_score(labels_test, pred))
print("Decision tree algorithm 2.0 time:",round(time()-t0, 3),"s")

from sklearn.naive_bayes import GaussianNB
t0=time()
clf = GaussianNB()
clf= clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("Accuracy: ",accuracy_score(labels_test, pred))
print("Precision: ",precision_score(labels_test, pred))
print("Recall: ",recall_score(labels_test, pred))
print("Naive bayes time:",round(time()-t0, 3),"s")


from sklearn.linear_model import LogisticRegression
t0=time()
clf = LogisticRegression()
clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("Accuracy: ", accuracy_score(labels_test, pred))
print("Precision: ", precision_score(labels_test, pred))
print("Recall: ", recall_score(labels_test, pred))
print("Logistic Regression algorithm time:", round(time()-t0, 3), "s")

from sklearn.ensemble import RandomForestClassifier
t0=time()
clf = RandomForestClassifier()
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("Accuracy: ",accuracy_score(labels_test, pred))
print("Precision: ",precision_score(labels_test, pred))
print("Recall: ",recall_score(labels_test, pred))
print("RandomForest time:",round(time()-t0, 3),"s")

from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':[1,2,3,4,5,6,7,8,9,10],'criterion':('gini','entropy'),'splitter':('best','random')}
svr =tree.DecisionTreeClassifier()
clf = GridSearchCV(svr, parameters, scoring = "f1")
clf = clf.fit(features_train,labels_train)
print clf.best_params_
#from sklearn.model_selection import GridSearchCV
#parameters = {'max_depth':[1,2,3,4,5,6,7,8,9,10,11],'n_estimators':[1,2,3,4,5,6,7,8,9,10],'criterion':('gini','entropy'),'oob_score':('False','True')}
#svr = RandomForestClassifier()
#clf = GridSearchCV(svr, parameters)
#clf = clf.fit(features_train,labels_train)
#print clf.best_params_
data = featureFormat(my_dataset, features_list3)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
t0=time()
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=8,splitter='best')
#clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("Accuracy: ", accuracy_score(labels_test, pred))
print("Precision: ", precision_score(labels_test, pred))
print("Recall: ", recall_score(labels_test, pred))
print("Decision Tree algorithm 3.0 time:",round(time()-t0, 3),"s")
### Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
labels, features = targetFeatureSplit(data)
t0=time()
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=9,splitter='best')
clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("Accuracy: ", accuracy_score(labels_test, pred))
print("Precision: ", precision_score(labels_test, pred))
print("Recall: ", recall_score(labels_test, pred))
print("Decision Tree algorithm 3.0 time:",round(time()-t0, 3),"s")

dump_classifier_and_data(clf, my_dataset, features_list)