from jaxrl.datasets.dataset import Batch
from jaxrl.datasets.replay_buffer import ReplayBuffer


def make_env_and_dataset(*args, **kwargs):
    # Offline datasets are not required for online Continual World experiments.
    from jaxrl.datasets.dataset_utils import make_env_and_dataset as make_dataset
    return make_dataset(*args, **kwargs)
