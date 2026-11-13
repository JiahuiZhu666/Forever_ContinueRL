# Runtime environment for the NEVO cumulative-decay reproduction on the local RTX 4090.
# Mirrors REPRODUCE.md section 3; paths adapted to this machine.
export BUNDLE_ROOT=/home/jerry/nevo_cumulative_stable_20261016/nevo_cumulative_stable_20261016
export NEVO_ENV=/home/jerry/nevo_env
export PATH="$NEVO_ENV/bin:$PATH"

export MUJOCO_PY_MUJOCO_PATH=/home/jerry/.mujoco/mujoco200
export MUJOCO_PY_MJKEY_PATH=/home/jerry/.mujoco/mjkey.txt
# Deviation from REPRODUCE.md: conda-forge mesalib 26 no longer ships OSMesa, so mujoco-py is
# built with its EGL (LinuxGPUExtensionBuilder) backend instead (FORCE_CPU unset, /usr/lib/nvidia
# present). Rendering is never used by train_cumulative.py; physics is identical.
unset MUJOCO_PY_FORCE_CPU
export MUJOCO_PY_FORCE_GPU=/usr/lib/x86_64-linux-gnu   # consumed by the local builder.py patch
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export CPATH="$NEVO_ENV/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NEVO_ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$NEVO_ENV/lib:$MUJOCO_PY_FORCE_GPU${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# XLA needs libdevice and ptxas. The conda cudatoolkit has libdevice but no ptxas, so the
# system CUDA 12.4 toolkit directory is used (ptxas 12.4 targets sm_89; runtime libs stay 11.8).
export CUDA_HOME="${NEVO_CUDA_DATA_DIR:-/usr/local/cuda-12.4}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
