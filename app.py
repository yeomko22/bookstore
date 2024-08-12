import json

import streamlit as st
from google.cloud import translate
from google.oauth2.service_account import Credentials
from openai import OpenAI
from pinecone import Pinecone

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("bookstore")
google_secret = st.secrets["GOOGLE_SECRET"]
credentials = Credentials.from_service_account_info(google_secret)
google_translate_client = translate.TranslationServiceClient(credentials=credentials)


def get_translation(query):
    parent = f"projects/{google_secret['project_id']}/locations/global"
    response = google_translate_client.translate_text(
        request={
            "parent": parent,
            "contents": [query],
            "mime_type": "text/plain",
            "source_language_code": "ko",
            "target_language_code": "en-US",
        }
    )
    return response.translations[0].translated_text


def extract_embedding(text_list):
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-large",
    )
    embedding_list = [x.embedding for x in response.data]
    return embedding_list


def search(query_embedding):
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    return results


def parse_search_results(results):
    matches = results["matches"]
    metadata_list = [x["metadata"] for x in matches]
    item_list = [{
        "제목": x["title"],
        "저자": x["authors"],
        "요약": x["summary"]
    } for x in metadata_list]
    return item_list


def generate_prompt(query, items):
    prompt = f"""
유저가 읽고 싶은 책에 대한 묘사와 이에 대한 추천 결과가 주어집니다.
유저의 입력과 각 추천 결과 책의 제목, 저자, 소개 등을 참고하여 추천사를 작성하세요.
당신에 대한 소개를 먼저 하고, 친절한 말투로 작성해주세요.
중간 중간 이모지를 적절히 사용해주세요.
```
query: {query}
items: {items}
```
    """
    return prompt


def request_chat_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 책을 추천해주는 AI 책방지기 웜즈입니다."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    return response


def print_streaming_response(response):
    container = st.empty()
    content = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            container.markdown(content)


st.title("웜즈의 책방")

# 1. 쿼리를 번역
# 2. 임베딩 추출
# 3. 프롬프트 생성
# 4. 텍스트 생성 요청
# 5. 결과를 화면에 출력
with st.form("form"):
    query = st.text_input("읽고 싶은 책을 묘사해주세요")
    submit_button = st.form_submit_button(label="추천받기")
if submit_button:
    with st.spinner("관련 책을 검색 중입니다..."):
        translated_query = get_translation(query)
        query_embedding = extract_embedding([
            translated_query
        ])
        results = search(query_embedding[0])
        item_list = parse_search_results(results)
        for item in item_list:
            with st.expander(item["제목"]):
                st.markdown(f"**저자:** {item['저자']}")
                st.markdown(f"**줄거리:** {item['요약']}")
    with st.spinner("추천사를 작성 중입니다..."):
        prompt = generate_prompt(
            query=query,
            items=json.dumps(item_list, indent=2, ensure_ascii=False)
        )
        response = request_chat_completion(prompt)
    print_streaming_response(response)
