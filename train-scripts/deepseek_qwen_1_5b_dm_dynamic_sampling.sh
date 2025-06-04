set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

GPU_NUM=8
train_files="['./data/dapo_math/train.parquet', 'data/lighteval-math/train.parquet']"
test_files="['./data/math500/test.parquet', './data/amc/test.parquet', './data/aime2024/test.parquet', 'data/gaokao/test.parquet', 'data/minervamath/test.parquet', 'data/olympiadbench/test.parquet']"
project_name='greso'
mkdir -p data-log/$project_name
experiment_name='deepseek_qwen_1_5b_dm_dynamic_sampling'

python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=192 \
    +data.real_train_batch_size=128 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    +ttis.type=dynamic_sampling \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=50 \
    +trainer.val_before_train=False \
    +trainer.save_dir=data-task/$project_name/$experiment_name \
    +trainer.max_steps=1001 \
    trainer.total_epochs=2000 \
    | tee data-log/$project_name/$experiment_name.log