# experiment_config.py

from RLAgent import RLAgent
from Cache import Cache
from DataLoader import DataLoaderPintos

def get_env_and_agent():
    data_loader = DataLoaderPintos("pda-159-dataset/filesys/extended/syn-rw.csv", boot=False)

    env = Cache(
        requests=data_loader,
        cache_size=get_cache_size(),
        feature_selection=('Base',),
        reward_params=dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3)
    )

    agent = RLAgent(
        n_actions=env.n_actions,
        n_features=len(env.reset()['features']),
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy_min=(0.05, 0.1),
        e_greedy_max=(0.2, 0.3),
        e_greedy_init=(0.1, 0.2),
        e_greedy_increment=(0.01, 0.01),
        e_greedy_decrement=(0.005, 0.005),
        reward_threshold=5,
        explore_mentor='LRU',
        replace_target_iter=200,
        memory_size=10000,
        batch_size=32,
        output_graph=False,
        verbose=1
    )

    return env, agent
















    def _get_cache_size():

      return 100
