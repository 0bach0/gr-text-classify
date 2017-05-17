# -*- coding: UTF-8 -*-
from sklearn import metrics
from sklearn.pipeline import Pipeline
from pyvi.pyvi import ViTokenizer
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.externals import joblib


if __name__ == "__main__":
  newsgroups_train = load_files('/usr/src/app/fb_data',load_content=True,shuffle=True)
  newsgroups_test = load_files('/usr/src/app/fb_test',load_content=True,shuffle=True)
  stopwords={'bị','bởi','cả','các','cái','cần','càng','chỉ','chiếc','cho','chứ','chưa','chuyện','có','có_thể','cứ','của','cùng','cũng','đã','đang','đây','để','đến_nỗi','đều','điều','do','đó','được','dưới','gì','khi','không','là','lại','lên','lúc','mà','mỗi','một_cách','này','nên','nếu','ngay','nhiều','như','nhưng','những','nơi','nữa','phải','qua','ra','rằng','rằng','rất','rất','rồi','sau','sẽ','so','sự','tại','theo','thì','trên','trước','từ','từng','và','vẫn','vào','vậy','vì','việc','với','vừa'}
  # file fold, tend fold, susume: cross validation
  # clf = MultinomialNB().fit(X_train_tfidf, newsgroups_train.target)
	
  text_clf = Pipeline([('vect', CountVectorizer(stop_words=stopwords)),('tfidf', TfidfTransformer(use_idf=True)),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
  text_clf = text_clf.fit(newsgroups_train.data, newsgroups_train.target)
  docs_test = newsgroups_test.data
  predicted = text_clf.predict(docs_test)
  joblib.dump(text_clf, 'train.pkl') 

  print(np.mean(predicted == newsgroups_test.target))
  print(metrics.classification_report(newsgroups_test.target, predicted,target_names=newsgroups_test.target_names))
  print(metrics.confusion_matrix(newsgroups_test.target, predicted))



	# Get best parameters
	# parameters = {'vect__ngram_range': [(1, 1)],
	# 			'tfidf__use_idf': (True),
	# 			'clf__alpha': (1e-3,1e-4),
	# }
	# gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
	# gs_clf = gs_clf.fit(newsgroups_train.data, newsgroups_train.target)
	# for param_name in sorted(parameters.keys()):
	# 	print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
