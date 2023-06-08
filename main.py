import os
import sys
import traceback

import torch
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from transformers import AutoModel, AutoTokenizer


TOP_K = 5

DOCUMENT_NAME = 'document.txt'
VECTOR_STORE_PATH = 'vector_store'

# Could be cuda or cpu or other torch supported devices
LLM_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_DEVICE = 'cuda'

# The embedding model name could be one of the following:
#   ghuyong/ernie-3.0-nano-zh
#   nghuyong/ernie-3.0-base-zh
#   shibing624/text2vec-base-chinese
#   GanymedeNil/text2vec-large-chinese
EMBEDDING_MODEL_NAME = 'GanymedeNil/text2vec-large-chinese'

# The LLM model name could be one of the following or local path:
#   THUDM/chatglm-6b-int4-qe
#   THUDM/chatglm-6b-int4
#   THUDM/chatglm-6b-int8
#   THUDM/chatglm-6b
LLM_MODEL = 'THUDM/chatglm-6b-int8'

PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}
"""


# ----------------------------------------------------------------------------------------------------------------------

class DocumentTextSplitter(CharacterTextSplitter):
    def __init__(self, threshold: int, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def split_text(self, text: str) -> [str]:
        # 定义成对符号
        pairs = {'"': '"', "'": "'", "‘": "’", "“": "”", "（": "）", "(": ")",
                 "[": "]", "【": "】", "{": "}", "<": ">", "《": "》", "〈": "〉"}
        stack = []
        result = []
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            if len(paragraph) <= self.threshold:
                result.append(paragraph)
            else:
                start = 0
                for i, char in enumerate(paragraph):
                    if char in pairs:
                        if stack and stack[-1] == pairs[char]:
                            stack.pop()
                        else:
                            stack.append(char)
                    elif not stack and char in ".!?。！？":
                        if i + 1 - start > self.threshold:
                            avg_split = (i + 1 - start) // self.threshold
                            for j in range(avg_split):
                                result.append(paragraph[start + j * self.threshold:start + (j + 1) * self.threshold])
                            start += avg_split * self.threshold
                        else:
                            result.append(paragraph[start:i + 1].strip())
                            start = i + 1
                if start < len(paragraph):
                    if len(paragraph) - start > self.threshold:
                        avg_split = (len(paragraph) - start) // self.threshold
                        for j in range(avg_split):
                            result.append(paragraph[start + j * self.threshold:start + (j + 1) * self.threshold])
                    else:
                        result.append(paragraph[start:].strip())
        return result


def file_to_document(file_name) -> list:
    loader = TextLoader(file_name, autodetect_encoding=True)
    docs = loader.load_and_split(DocumentTextSplitter(100))
    return docs


def setup_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': EMBEDDING_DEVICE})
    return embeddings


def check_create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def build_vector_store(document_file: str, vector_store_path: str, embedding):
    try:
        check_create_directory(vector_store_path)
        docs = file_to_document(document_file)
        vector_store = FAISS.from_documents(documents=docs, embedding=embedding)
        vector_store.save_local(vector_store_path)
        return vector_store
    except Exception as e:
        print(e)
        return None
    finally:
        pass


def load_vector_store(vector_store_path: str, embedding):
    try:
        vector_store = FAISS.load_local(folder_path=vector_store_path, embeddings=embedding)
        return vector_store
    except Exception as e:
        print(e)
        return None
    finally:
        pass


def setup_llm():
    try:
        device = torch.device(LLM_DEVICE)
        model = AutoModel.from_pretrained(LLM_MODEL, trust_remote_code=True).half().to(device)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        return model, tokenizer
    except Exception as e:
        print(e)
        return None, None
    finally:
        pass


def prob_related_documents_and_score(related_docs_with_score):
    for doc, score in related_docs_with_score:
        print('%s [%s]: %s' % (doc.metadata['source'], score, doc.page_content))


def generate_llm_prompts(question: str, related_docs_with_score) -> str:
    context = '\n'.join([doc.page_content for doc, score in related_docs_with_score])
    return PROMPT_TEMPLATE.replace('{question}', question).replace('{context}', context)


def main():
    embedding = setup_embedding()

    vector_store = load_vector_store(VECTOR_STORE_PATH, embedding) or \
                   build_vector_store(DOCUMENT_NAME, VECTOR_STORE_PATH, embedding)

    if vector_store is None:
        exit(-1)

    # -----------------------------------------------------------------------------

    model, tokenizer = setup_llm()
    if model is None or tokenizer is None:
        exit(-2)
    model = model.eval()

    # -----------------------------------------------------------------------------

    history = []

    while True:
        inputs = input("Input your question：")

        related_docs_with_score = vector_store.similarity_search_with_score(inputs, k=TOP_K)
        prob_related_documents_and_score(related_docs_with_score)

        prompts = generate_llm_prompts(question=inputs, related_docs_with_score=related_docs_with_score)
        print(prompts)

        response, _ = model.chat(
            tokenizer,
            prompts,
            history=history,
            max_length=10000,
            temperature=5.0
        )

        # history += [[prompts, response]]

        print(response)


# ----------------------------------------------------------------------------------------------------------------------

def exception_hook(_type, value, tback):
    # log the exception here
    print('Exception hook triggered.')
    print(_type)
    print(value)
    print(tback)
    # then call the default handler
    sys.__excepthook__(_type, value, tback)


if __name__ == "__main__":
    sys.excepthook = exception_hook
    try:
        main()
    except Exception as e:
        print('Error =>', e)
        print('Error =>', traceback.format_exc())
        exit()
    finally:
        pass
