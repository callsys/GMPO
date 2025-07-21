conda create -n gmpo python==3.10
conda activate gmpo
pip install vllm==0.8.4 && pip install oat-llm==0.1.3.post1
cd understand_r1_zero_main
pip install -e .

# download the base model first: Qwen2.5-Math-7B, Qwen2.5-Math-1.5B
bash qwen2.5-math-7b-gmpo.sh


