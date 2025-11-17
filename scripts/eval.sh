export LIBHOME=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export NVHOME=${LIBHOME}/python3.10/site-packages/nvidia;
cp -rp ${NVHOME}/cuda_runtime/lib/libcudart.so.12 ${NVHOME}/cuda_runtime/lib/libcudart.so.11.0;
cp -rp ${NVHOME}/cublas/lib/libcublasLt.so.12 ${NVHOME}/cublas/lib/libcublasLt.so.11;
cp -rp ${NVHOME}/cublas/lib/libcublas.so.12 ${NVHOME}/cublas/lib/libcublas.so.11;
cp -rp ${NVHOME}/cufft/lib/libcufft.so.11 ${NVHOME}/cufft/lib/libcufft.so.10;
cp -rp ${NVHOME}/cusparse/lib/libcusparse.so.12 ${NVHOME}/cusparse/lib/libcusparse.so.11;
cp -rp ${NVHOME}/cudnn/lib/libcudnn.so.9 ${NVHOME}/cudnn/lib/libcudnn.so.8;
export LD_LIBRARY_PATH=${LIBHOME}:${NVHOME}/cuda_runtime/lib:${NVHOME}/cublas/lib:${NVHOME}/cufft/lib:${NVHOME}/cusparse/lib:${NVHOME}/cudnn/lib:$LD_LIBRARY_PATH;
export WANDB_MODE=offline;

# model="oat-output/gmpo_7B/saved_models/step_00500/"
# python utils/evaluation/evaluate_model.py --model_name $model
# python utils/evaluation/evaluate_model.py --model_name $model --top_p 0.95 --temperature 0.6 --n_samples 16

python scripts/eval.py

