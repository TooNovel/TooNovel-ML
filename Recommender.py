import pickle
import random
from collections import defaultdict
import pandas as pd
from surprise import Dataset, Reader
import json

def run(user_id):
    print(user_id)
    with open('svd_model_transferred.pkl', 'rb') as file:
        model = pickle.load(file)

    def get_top_n_recommendations(predictions, n=5):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    # 데이터 로딩
    ratings = pd.read_csv('application_rating_data.csv')
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['user_id', 'novel_id', 'rating']], reader)

    # 데이터셋 생성 - 항목 ID 리스트 생성용
    trainset = data.build_full_trainset()

    # 예측할 사용자 ID와 해당 사용자가 평가하지 않은 모든 항목 ID 리스트 생성
    user_inner_id = trainset.to_inner_uid(user_id)
    items_rated_by_user = set(trainset.ur[user_inner_id])
    items_rated_by_user = {trainset.to_raw_iid(inner_id) for (inner_id, _) in items_rated_by_user}
    items_to_predict = set(trainset.all_items()) - items_rated_by_user

    # 예측 수행
    predictions = [model.predict(user_id, item_id) for item_id in items_to_predict]
    top_n_recommendations = get_top_n_recommendations(predictions, n=100)

    # 사용자별 추천 작품 출력
    user_recommendations = top_n_recommendations[user_id]

    # 랜덤한 결과를 위해 배열을 섞어서 100개중 10개만 출력
    random.shuffle(user_recommendations)
    user_recommendations = user_recommendations[0:10]

    # print(f"User {user_id}에게 추천하는 작품:")
    # for i, (novel_id, rating) in enumerate(user_recommendations):
    #     print(f"{i+1}. Novel ID: {novel_id}, 예측 평점: {rating:.2f}")

    # # int64는 처리 못하므로 int로 변경
    # user_recommendations = {int(novel_id): round(float(rating), 4) for novel_id, rating in user_recommendations}

    # # JSON 형태로 변환
    # result = json.dumps(user_recommendations)
    # print(result)

    # 예상 점수는 필요 없을 것 같아 novel_id만 반환하도록 변경
    result = [int(novel_id) for novel_id, _ in user_recommendations]
    return result