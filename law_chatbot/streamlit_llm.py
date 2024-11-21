# streamlit_llm.py

import streamlit as st
from unsloth import FastLanguageModel
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

# Streamlit App 제목
st.title("Legal AI Chatbot")

# 모델 경로
model_path = "../llama/Llama-3-legal_fintuning_epoch_1000"

# 모델 및 토크나이저 초기화 (캐시된 모델 로딩)
@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

# StoppingCriteria 설정
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_token_id in input_ids[0]

stop_token = "<|end_of_text|>"
stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)[0]
stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])

# TextStreamer 설정
class StreamlitTextStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.response = ""

    def write(self, text):
        self.response += text
        st.write(text)

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if user_input := st.chat_input("Type your question here:"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 입력 텍스트 포맷팅
    alpaca_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_input}

### Response:
"""
    inputs = tokenizer(
        alpaca_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")

    # TextStreamer를 사용한 실시간 응답 표시
    with st.chat_message("assistant"):
        streamer = StreamlitTextStreamer(tokenizer)
        _ = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria,
            do_sample=True,
            top_k=50,
            temperature=0.1,
        )
        response = streamer.response
        st.markdown(response)

    # 응답 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
