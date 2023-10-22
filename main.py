from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import re


# 크롤링 함수
def crawling():
    # 주소
    url = "https://www.rottentomatoes.com/browse/movies_at_home/sort:popular"

    response = requests.get(url)

    # bs4를 사용하여 HTML 파싱
    soup = BeautifulSoup(response.text, "html.parser")

    # score 추출
    score = soup.select('score-pairs-deprecated')
    # title 추출
    title = soup.select('[data-qa="discovery-media-list-item-title"]')
    # date 추출
    date = soup.select('[data-qa="discovery-media-list-item-start-date"]')

    data_list = []

    for idx in range(len(score)):
        # score (토마토)
        score_value = score[idx]['criticsscore']
        # audience score (팝콘)
        audience_score_value = score[idx]['audiencescore']
        title_value = title[idx].get_text(strip=True)
        date_value = date[idx].get_text(strip=True)

        data_object = {
            'score': score_value,
            'audience_score': audience_score_value,
            'title': title_value,
            'date': date_value
        }

        data_list.append(data_object)

    return data_list


# Tokenization 및 Cleaning 함수
def tokenization_and_cleaning(dataset):
    # stopwords 로드
    nltk.download('stopwords')
    # punkt 로드
    nltk.download('punkt')

    stop_words = stopwords.words('english')
    cleaned_data = []
    tokenization_title = []
    for data in dataset:
        title = data['title']
        # 소문자로 변환
        title = title.lower()
        # Tokenization (단어 토큰화),
        words = word_tokenize(title)
        # 토큰화 한 words를 따로 리스트에 담기 (결과값 출력용)
        tokenization_title.append(words)
        # Stopwords(불용어) 제거 + 알파벳 문자만 남김
        words = [word for word in words if word.isalpha() and word not in stop_words]
        # 다시 공백으로 연결하여 문자열로 변환
        cleaned_title = ' '.join(words)
        data['title'] = cleaned_title
        cleaned_data.append(data)
    # Tokenization 및 Cleaning한 리스트, 토큰화한 title
    return cleaned_data, tokenization_title


# 품사 태깅 함수
def tagging_title(title):
    title_list = []
    for data in title:
        title_list.append(pos_tag(data))
    return title_list


# Lemmatization (표제어 추출) 함수
def lemmatization(dataset):
    lemmatizer = WordNetLemmatizer()

    lemmatized_data = []
    for data in dataset:
        item = [lemmatizer.lemmatize(word) for word in data]
        lemmatized_data.append(item)
    return lemmatized_data

# score (토마토) 에 따른 태깅
def score_tagging(dataset):
    for data in dataset:
        score = int(data['score'])
        if score >= 60:
            data['tag'] = "Fresh"
        else:
            data['tag'] = 'Rotten'
    return dataset

if __name__ == "__main__":
    # 크롤링 data
    data = crawling()

    # 반환 값 =  전체 list, 토큰화 한 Title
    tokenization_list, tokenization_title = tokenization_and_cleaning(data)
    # 토큰화 한 Title 출력
    print("기존 Data = ", data)
    print("토큰화 Title List = ", tokenization_title)
    tagged_title = tagging_title(tokenization_title)
    print("품사 태깅 Title List = ", tagged_title)
    lemmatized_title = lemmatization(tokenization_title)
    print("표제어 추출 Title List = ", lemmatized_title)
    score_tagged_title = score_tagging(tokenization_list)
    print("점수에 따른 Tagging = ", score_tagged_title)

    # 정규 표현식 활용 - 제목의 첫 글자 대문자로 변환
    for data in tokenization_list:
        data['title'] = data['title'].title()
    print("제목의 첫 문자를 대문자로 변환 = ", tokenization_list)

    # 정규 표현식 활용 - score(토마토)와 audience_score(팝콘)의 데이터 유형 변환
    for data in tokenization_list:
        data['score'] = int(data['score'])
        data['audience_score'] = int(data['audience_score'])
    print("데이터 유형 변환 ", tokenization_list)

    # 정규 표현식 활용 - date의 변환
    for data in tokenization_list:
        date_match = re.search(r'(\w{3}) (\d{1,2}), (\d{4})', data['date'])
        if date_match:
            month = date_match.group(1)
            day = int(date_match.group(2))
            year = int(date_match.group(3))
            month_number = {'Jan': 1, 'Feb': 2, 'Mar': 3,
                            'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9,
                            'Oct': 10, 'Nov': 11, 'Dec': 12}
            month_num = month_number.get(month, "null")  # 월을 숫자로 매핑
            if month_num != "null":
                data['date'] = f"{year}.{month_num}.{day}"
    print("날짜 변환 = ", tokenization_list)

