# -*- coding: UTF-8 -*-
from sklearn.externals import joblib
from sklearn.datasets import load_files
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
from json import dumps
from pyvi.pyvi import ViTokenizer
import re

text_clf = joblib.load('train.pkl')
parser = reqparse.RequestParser()
parser.add_argument('data')

app = Flask(__name__)
api = Api(app)

def clean_sentence(sentence):
  	# if not sentence:
	# 	print('false')
	# 	print(sentence)
	
	sentence = remove_url(sentence)

  # Remove non-letters
	sentence = sentence.lower()
	tmp = u"[^a-zA-Z0-9/ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆẾỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪếễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵýỷỹ]"
	letters_only = re.sub(tmp," ", sentence)
	letters_only = re.sub(' +',' ',letters_only)

	return letters_only

def remove_url(sentence):
	result = re.sub(r"http\S+", "", sentence)
	return result

class text_classify(Resource):
    def post(self):
        args = parser.parse_args()
        data = args.data

        data=clean_sentence(data)
        data = ViTokenizer.tokenize(data)
        predicted = text_clf.predict([data])
        print('-----------------------------')
        print(data.encode('utf-8'))
        print(type[predicted[0]].encode('utf-8'))
        return {'data':data,'type': predicted[0],'classify':type[predicted[0]]}


api.add_resource(text_classify, '/text-classify')

if __name__ == "__main__":
  type = [u'Tin tức khác',u'Học phí',u'Học bổng',u'Tuyển sinh',u'Tuyển dụng',u'Đào tạo',u'Quy chế',u'Sinh viên']
  # docs_test = [u'thông_báo thông_báo kết_quả xét_duyệt hồ_sơ cao_học đợt 1 năm 2015 các thí_sinh đăng_ký dự thi thuộc đối_tượng phải học bổ_sung theo danh_sách kết_quả xét_duyệt hồ_sơ cần phải đến đăng_ký học bổ_sung tại viện đt sđh phòng 315 nhà_c1 và nộp học_phí tại phòng kế_hoạch tài_vụ phòng 204 nhà c3 4 trong khoảng thời_gian từ ngày 29/10/2014 đến ngày 31/10/2014 những thí_sinh thuộc đối_tượng phải học bổ_sung nếu không học bổ_sung sẽ không đủ điều_kiện dự thi thông_tin chi_tiết']
  # predicted = text_clf.predict(docs_test)
  # print(predicted)

  app.run(port=3000,host='0.0.0.0')
  
  