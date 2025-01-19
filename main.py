"""
Steps:
1. List up all the files in the diary folder : "/diary"
2. Read each file and extract following information:
    - Date (YYYY-MM-DD)
    - Frontmatter
    - Content(RawContent - Frontmatter - Index): 昨日のエントリから始まる行を削除
    Then, store the information in a database
3. Read the database and generate following information:
    - Workout Menu
    - Date created
    Then, store the information in a database 
4. Read the database to extract Diary Extractd, then do sentiment analysis, then store the information in a database

"""

from datetime import datetime
import os
import duckdb
import json
import re
import yaml
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

duckdb_file_name = "output/diary.duckdb"

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def list_files(path):
    return os.listdir(path)

def extract_raw_infomation(file_path):
    with open(file_path, "r") as f:
        raw_content = f.read()
        split_content = raw_content.split("---")
        frontmatter = split_content[1]
        raw_content = split_content[2].split("昨日のエントリ")[0]
        return frontmatter, raw_content


def store_raw_contents(raw_contents):
    with open("raw_contents.json", "w") as f:
        json.dump(raw_contents, f)
    con = duckdb.connect(duckdb_file_name)
    con.sql("CREATE OR REPLACE TABLE raw_contents AS SELECT * FROM read_json('raw_contents.json')")
    con.sql("select * from raw_contents").write_csv("output/raw_contents.csv")
    con.close()

def sentiment_analysis(text):
    if text is None:
        return [{"label": "Neutral", "score": 0.5}]
    inputs = tokenizer(text=text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return [{"label": sentiment_map[p], "score": probabilities[0][p].item()} for p in torch.argmax(probabilities, dim=-1).tolist()]


def extract_information_from_frontmatter(frontmatter_str):
    """フロントマターからジムメニューと作成日時を抽出する
    
    Returns:
        tuple: (gym_menu_list, created_at)
            - gym_menu_list: ジムメニューの配列
            - created_at: 作成日時（YYYY-MM-DD HH:MM形式）
    """
    
    try:
        # YAMLとしてパース
        frontmatter_data = yaml.safe_load(frontmatter_str)
        
        # ジムメニューを取得（存在しない場合は空配列）
        gym_menu_list = frontmatter_data.get('ジムメニュー', [])
        
        # 作成日時を取得（存在しない場合は空文字列）
        created_at = frontmatter_data.get('作成日時', '')
        return gym_menu_list, created_at
        
    except Exception as e:
        print(f"Error parsing frontmatter: {e}")
        return [], ''

def transform_raw_content(raw_content):
    # 利用するトークン量を減らすため、不要な文字を削除していく
    raw_content = raw_content.replace("\n", " ")
    return raw_content

def get_ymd_from_file_name(file_name):
    return file_name.split(".")[0]

def main():
    path = "/mnt/c/Obsidian/Valut111/Vault111/diary/Daily"
    files = list_files(path)
    contents = []
    for file in files:
        frontmatter, raw_content = extract_raw_infomation(f"{path}/{file}")
        gym_menu_list, created_at = extract_information_from_frontmatter(frontmatter)
        content = transform_raw_content(raw_content)
        sentiment_result = sentiment_analysis(content)
        ymd = get_ymd_from_file_name(file)
        contents.append({
            "file_name": file,
            "ymd": ymd,
            "raw_content": raw_content,
            "gym_menu_list": gym_menu_list,
            "created_at": created_at,
            "sentiment_label": sentiment_result[0]['label'],
            "sentiment_score": sentiment_result[0]['score'],
        })
        print("file_name: ", file, "sentiment_label: ", sentiment_result[0]['label'], "sentiment_score: ", sentiment_result[0]['score'])
    store_raw_contents(contents)
        

if __name__ == "__main__":
    main()
    # res = sentiment_analysis("I love programming.")
    # print(res)

