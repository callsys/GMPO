
model="/root/GMPO/oat-output/gmpo_noclip/saved_models/step_00600"
# python utils/evaluation/evaluate_model.py --model_name $model
python utils/evaluation/evaluate_model.py --model_name $model --top_p 0.95 --temperature 0.6 --n_samples 16
