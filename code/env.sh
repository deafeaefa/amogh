# GCQ environment — source this in every shell/job
export PYT=/share/pkg.8/academic-ml/spring-2026/install/spring-2026-pyt/bin/python
export GCQ_STORE=/projectnb/rise-tower/eric1/GCQ
export HF_HOME=$GCQ_STORE/hf_cache
export GCQ_DATA=$GCQ_STORE/data
export GCQ_RUNS=$GCQ_STORE/runs
export TOKENIZERS_PARALLELISM=false
# job allocation = physical GPUs 4,5,6 (torch devices 0,1,2); leave CUDA_VISIBLE_DEVICES as SGE set it
