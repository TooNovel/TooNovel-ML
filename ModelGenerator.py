import pickle
import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import KFold 

def run():
    # 데이터 로딩
    ratings = pd.read_csv('rating.csv')
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['user_id', 'novel_id', 'rating']], reader)

    # 잠재요인 기반 협업 필터링
    svd = SVD()

    # k-fold 교차 검증 설정
    kf = KFold(n_splits=5)

    from surprise import accuracy

    # k-fold 교차 검증을 사용하여 모델 평가
    for trainset, testset in kf.split(data):
        svd.fit(trainset)
        predictions = svd.test(testset)
        accuracy.rmse(predictions, verbose=True)

    with open('svd_model.pkl', 'wb') as file:
        pickle.dump(svd, file)