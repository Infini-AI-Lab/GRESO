import torch

def select_batch_slice(batch, num):
    selected_indices = torch.arange(num)
    selected_batch = batch.select_via_index(selected_indices)

    return selected_batch

# descending order: higher score better
def select_batch(batch, num, scores, config):
    selected_indices = torch.argsort(scores, descending=True)[:num]
    unselected_indices = torch.argsort(scores, descending=True)[num:]

    print(selected_indices)
    print(unselected_indices)
    selected_indices = selected_indices.unsqueeze(1) * config.actor_rollout_ref.rollout.n + torch.arange(config.actor_rollout_ref.rollout.n).unsqueeze(0)
    selected_batch = batch.select_via_index(selected_indices.view(-1))

    unselected_indices = unselected_indices.unsqueeze(1) * config.actor_rollout_ref.rollout.n + torch.arange(config.actor_rollout_ref.rollout.n).unsqueeze(0)
    unselected_batch = batch.select_via_index(unselected_indices.view(-1))

    return selected_batch, unselected_batch

def filter_zero_advantage(batch, config):
    reward = batch.batch['reward']
    reward = reward.reshape(-1, config.actor_rollout_ref.rollout.n)
    zero_advantage_flag = (reward.max(dim=1)[0] == reward.min(dim=1)[0])
    # select batch without zero advantage
    selected_indices = torch.arange(reward.shape[0])[~zero_advantage_flag]
    print('****************** Q-Scaling ******************')
    print('TTIS: Filter Zero Advantage')
    print(f'Original batch size: {len(batch)}')
    print('zero_advantage_flag:')
    print(zero_advantage_flag)
    # print(selected_indices)
    selected_indices = selected_indices.unsqueeze(1) * config.actor_rollout_ref.rollout.n + torch.arange(config.actor_rollout_ref.rollout.n).unsqueeze(0)
    batch = batch.select_via_index(selected_indices.view(-1))
    print(f'New batch size: {len(batch)}')
    return batch

