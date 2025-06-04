# create a class DataProfiler
# each data is identified by a unique string id
# each data has a reward trace which is a list of tuples (epoch, reward)
# the class should have a method to add reward tuple. If the data is not in the list, it should be added
# the class should have a method to get the reward trace for a given data id
import torch

class DataProfiler:
    def __init__(self):
        self.data = {}

    def add_reward(self, epoch: int, data_id: str, reward: float):
        if data_id not in self.data:
            self.data[data_id] = []
        self.data[data_id].append((epoch, reward))

    # the input is a list of datasource (str) and a list of index (int);
    # return a list of unique data_id which is combined from the datasource and index
    def get_data_id_list(self, data_sources: list, indices: list):
        data_ids = []
        for data_source, index in zip(data_sources, indices):
            data_ids.append(f"{data_source}_{index}")
        return data_ids

    # Calculate the keep probability for given a list of data_ids
    # if there is at list one 1 in reward trace, the keep probability is 0, else it is 1
    def get_skip_easy_probabiliy(self, data_ids: list):
        keep_probability_list = []
        for data_id in data_ids:
            reward_trace = self.data.get(data_id, [])
            if 1 in reward_trace:
                keep_probability_list.append(0.0)
            else:
                keep_probability_list.append(1.0)
        return keep_probability_list

    # return True if there is at least one 1 in the reward trace in the recent n epochs
    def easy_example_skip(self, current_epoch, data_id, n=-1):
        if data_id not in self.data:
            return False
        reward_trace = self.data[data_id]
        if len(reward_trace) == 0:
            return False
        if n == -1: # search all the epochs
            n = 10e7
        for i in range(0, len(reward_trace)):
            if reward_trace[i][0] + n >= current_epoch and reward_trace[i][1] == 1:
                return True
        return False

    def easy_linear_backoff_skip(self, current_epoch, data_id, k):
        if data_id not in self.data:
            return False
        reward_trace = self.data[data_id]
        if len(reward_trace) == 0:
            return False

        # from end to start, find how many consercutive rewards
        count = 0
        for i in range(len(reward_trace)-1, -1, -1):
            if reward_trace[i][1] < 0.98: # contain at least one 1 wrong response
                break
            count +=1

        if count == 0: return False

        max_epoch = max([reward[0] for reward in reward_trace])

        backoff_epoch = torch.randint(0, k * count, (1,)).item()
        if max_epoch + backoff_epoch >= current_epoch:
            return True
        else:
            return False

    def hard_linear_backoff_skip(self, current_epoch, data_id, k):
        if data_id not in self.data:
            return False
        reward_trace = self.data[data_id]
        if len(reward_trace) == 0:
            return False

        if reward_trace[-1][1] > 0.11: # solvable example
            return False

        # from end to start, find how many consercutive rewards are less than 0.1
        count = 0
        for i in range(len(reward_trace)-1, -1, -1):
            if reward_trace[i][1] > 0.11:
                break
            count +=1

        if count == 0: return False

        max_epoch = max([reward[0] for reward in reward_trace])
        # randomly sample an integer between 0 and 2*count
        backoff_epoch = torch.randint(0, k * count, (1,)).item()
        if max_epoch + backoff_epoch >= current_epoch:
            return True
        else:
            return False

    def easy_probabilistic_skip(self, current_epoch, data_id, p=0.75):
        if data_id not in self.data:
            return False
        reward_trace = self.data[data_id]
        if len(reward_trace) == 0:
            return False

        # from end to start, find how many consercutive rewards
        count = 0
        for i in range(len(reward_trace)-1, -1, -1):
            if reward_trace[i][1] < 0.98: # contain at least one 1 wrong response
                break
            count +=1

        #calculate the probability of skipping based on the number of rewards
        skip_probability = 1 - max(p ** count, 0.01)
        #determine if the example should be skipped based on the probability
        if torch.rand(1).item() < skip_probability:
            return True
        else:
            return False

    def hard_probabilistic_skip(self, current_epoch, data_id, p=0.5):
        if data_id not in self.data:
            return False
        reward_trace = self.data[data_id]
        if len(reward_trace) == 0:
            return False

        if reward_trace[-1][1] > 0.11: # solvable example
            return False

        # from end to start, find how many consercutive rewards are less than 0.1
        count = 0
        for i in range(len(reward_trace)-1, -1, -1):
            if reward_trace[i][1] > 0.11:
                break
            count +=1

        #calculate the probability of skipping based on the number of rewards
        skip_probability = 1 - max(p ** count, 0.01)
        #determine if the example should be skipped based on the probability
        if torch.rand(1).item() < skip_probability:
            return True
        else:
            return False

    # filter the batch by the easy_example_skip function
    # the input is a batch of data, and the output is a batch of data
    def filter_examples_easy(self, current_epoch, batch, n=-1):
        # import ipdb; ipdb.set_trace()
        selected_index = []
        data_index = batch.non_tensor_batch['index'].astype(int).tolist()
        data_source = batch.non_tensor_batch['data_source'].tolist()
        data_id_list = self.get_data_id_list(data_source, data_index)
        easy_examples_count = 0
        for index, data_id in enumerate(data_id_list):
            if self.easy_example_skip(current_epoch, data_id, n):
                easy_examples_count += 1
                continue
            selected_index.append(index)
        batch = batch.select_via_index(selected_index)
        print(f"easy examples count: {easy_examples_count}")
        # print(f"selected examples count: {len(selected_index)}")
        return batch

    # use easy_example_skip and hard_linear_backoff_skip to filter the batch
    def filter_examples_linear_backoff(self, current_epoch, batch, n=-1):
        selected_index = []
        data_index = batch.non_tensor_batch['index'].astype(int).tolist()
        data_source = batch.non_tensor_batch['data_source'].tolist()
        data_id_list = self.get_data_id_list(data_source, data_index)
        easy_examples_count = 0
        hard_examples_count = 0
        for index, data_id in enumerate(data_id_list):
            if self.easy_example_skip(current_epoch, data_id, n):
                easy_examples_count += 1
                continue
            if self.hard_linear_backoff_skip(current_epoch, data_id):
                hard_examples_count += 1
                continue
            selected_index.append(index)
        batch = batch.select_via_index(selected_index)
        print(f"easy examples count: {easy_examples_count}, hard examples count: {hard_examples_count}")
        # print(f"selected examples count: {len(selected_index)}")
        return batch

    # use easy_example_skip and hard_probabilistic_skip to filter the batch
    def filter_examples_probabilistic(self, current_epoch, batch, n=-1, p=0.5):
        selected_index = []
        data_index = batch.non_tensor_batch['index'].astype(int).tolist()
        data_source = batch.non_tensor_batch['data_source'].tolist()
        data_id_list = self.get_data_id_list(data_source, data_index)
        easy_examples_count = 0
        hard_examples_count = 0
        for index, data_id in enumerate(data_id_list):
            if self.easy_example_skip(current_epoch, data_id, n=n):
                easy_examples_count += 1
                continue
            if self.hard_probabilistic_skip(current_epoch, data_id, p=p):
                hard_examples_count += 1
                continue
            selected_index.append(index)
        batch = batch.select_via_index(selected_index)
        print(f"easy examples count: {easy_examples_count}, hard examples count: {hard_examples_count}")
        # print(f"selected examples count: {len(selected_index)}")
        return batch

    def filter_examples_all_linear_backoff(self, current_epoch, batch, k_easy, k_hard):
        selected_index = []
        data_index = batch.non_tensor_batch['index'].astype(int).tolist()
        data_source = batch.non_tensor_batch['data_source'].tolist()
        data_id_list = self.get_data_id_list(data_source, data_index)
        easy_examples_count = 0
        hard_examples_count = 0
        for index, data_id in enumerate(data_id_list):
            if self.easy_linear_backoff_skip(current_epoch, data_id, k_easy):
                easy_examples_count += 1
                continue
            if self.hard_linear_backoff_skip(current_epoch, data_id, k_hard):
                hard_examples_count += 1
                continue
            selected_index.append(index)
        batch = batch.select_via_index(selected_index)
        print(f"AL; easy examples count: {easy_examples_count}, hard examples count: {hard_examples_count}")
        # print(f"selected examples count: {len(selected_index)}")
        return batch

    def filter_examples_all_probabilistic(self, current_epoch, batch, p_easy, p_hard, return_log=False):
        selected_index = []
        data_index = batch.non_tensor_batch['index'].astype(int).tolist()
        data_source = batch.non_tensor_batch['data_source'].tolist()
        data_id_list = self.get_data_id_list(data_source, data_index)
        easy_examples_count = 0
        hard_examples_count = 0
        skip_log = []
        for index, data_id in enumerate(data_id_list):
            if self.easy_probabilistic_skip(current_epoch, data_id, p=p_easy):
                easy_examples_count += 1
                skip_log.append([data_id, 0])
                continue
            if self.hard_probabilistic_skip(current_epoch, data_id, p=p_hard):
                hard_examples_count += 1
                skip_log.append([data_id, 1])
                continue
            selected_index.append(index)
            skip_log.append([data_id, 2])

        batch = batch.select_via_index(selected_index)
        print(f"AP; easy examples count: {easy_examples_count}, hard examples count: {hard_examples_count}")
        # print(f"selected examples count: {len(selected_index)}")
        if return_log:
            return batch, skip_log
        return batch

    # get a list of id, and a list of reward float; add them to the profiler
    def add_reward_list(self, epoch, data_ids: list, rewards: list):
        for data_id, reward in zip(data_ids, rewards):
            self.add_reward(epoch, data_id, reward)

    def get_reward_trace(self, data_id: str):
        return self.data.get(data_id, [])

    def get_all_data_ids(self):
        return list(self.data.keys())

    def save(self, path: str):
        torch.save(self.data, path)

    def load(self, path: str):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)


