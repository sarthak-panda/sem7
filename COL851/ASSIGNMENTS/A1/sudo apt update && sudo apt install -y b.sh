sudo apt update && sudo apt install -y build-essential git cmake python3 python3-venv python3-pip curl

mkdir -p ~/llm-local && cd ~/llm-local

git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp

mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF

cmake --build . -j$(nproc)

cd ../.. && python3 -m venv venv && source venv/bin/activate && python -m pip install --upgrade pip && pip install llama-cpp-python tqdm datasets

curl -L -o ~/llm-local/model.gguf "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf"

./build/bin/llama-cli -m ~/llm-local/model.gguf -t $(nproc) -c 2048 -n 256 --instruct

cd ..

export CUDA_VISIBLE_DEVICES=""

printf "Summarize: The quick brown fox jumps over the lazy dog.\nTranslate to French: Hello world.\nExplain recursion in one sentence.\n" > ~/llm-local/prompts.txt

cat > ~/llm-local/batch_infer.py <<'PY'
from llama_cpp import Llama
import time, json, os
model_path = os.path.expanduser('~/llm-local/model.gguf')
llm = Llama(model_path=model_path)
with open(os.path.expanduser('~/llm-local/prompts.txt')) as f, open(os.path.expanduser('~/llm-local/outputs.jsonl'), 'w') as out:
    for i, p in enumerate(f):
        p = p.strip()
        if not p: continue
        t0 = time.time()
        res = llm.create_completion(prompt=p, max_tokens=128)
        t1 = time.time()
        out.write(json.dumps({'id': i, 'latency': t1 - t0, 'output': res['choices'][0]['text']}) + '\n')
PY
python3 ~/llm-local/batch_infer.py

cat outputs.jsonl