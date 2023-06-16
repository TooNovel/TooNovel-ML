import csv
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_DBNAME = os.getenv('DB_DBNAME')

def run():
    con = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_DBNAME, charset='utf8')
    cursor = con.cursor()

    sql = "SELECT user_id, novel_id, review_grade rating FROM review"
    cursor.execute(sql)

    rows = cursor.fetchall()

    column_names = [column[0] for column in cursor.description]

    with open('rating_review.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        writer.writerows(rows)

    print('리뷰 데이터 저장 완료')

    con.close()