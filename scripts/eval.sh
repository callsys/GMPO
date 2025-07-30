
model="oat-output/gmpo_7B/saved_models/step_00500/"
python utils/evaluation/evaluate_model.py --model_name $model
# python utils/evaluation/evaluate_model.py --model_name $model --top_p 0.95 --temperature 0.6 --n_samples 16
