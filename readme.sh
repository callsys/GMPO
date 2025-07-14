conda create -n ur1z python==3.10
pip install vllm==0.7.2 && pip install oat-llm==0.0.9
git clone https://github.com/sail-sg/understand-r1-zero && cd understand-r1-zero
pip install -e .
cp ../train_zero_math_gmpo.py .
cp ../qwen2.5-math-7b-gmpo.sh examples
bash examples/qwen2.5-math-7b-gmpo.sh


