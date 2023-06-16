import pickle
import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import KFold 
from surprise import accuracy

def run():
    # 사전훈련 데이터 로딩
    ratings = pd.read_csv('pre_training_rating_data.csv')
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['user_id', 'novel_id', 'rating']], reader)

    # 잠재요인 기반 협업 필터링
    svd = SVD()

    # k-fold 교차 검증 설정
    kf = KFold(n_splits=5)

    # k-fold 교차 검증을 사용하여 모델 학습/평가
    for trainset, testset in kf.split(data):
        svd.fit(trainset)
        predictions = svd.test(testset)
        accuracy.rmse(predictions, verbose=True)

    # 실제 리뷰 데이터 로딩
    ratings_new = pd.read_csv('application_rating_data.csv')
    reader_new = Reader(rating_scale=(ratings_new['rating'].min(), ratings_new['rating'].max()))
    data_new = Dataset.load_from_df(ratings_new[['user_id', 'novel_id', 'rating']], reader_new)

    # k-fold 교차 검증을 사용하여 추가 학습/평가
    for trainset_new, testset_new in kf.split(data_new):
        svd.fit(trainset_new)
        predictions_new = svd.test(testset_new)
        accuracy.rmse(predictions_new, verbose=True)

    # 모델 저장
    with open('svd_model_transferred.pkl', 'wb') as file:
        pickle.dump(svd, file)
