from dataclasses import dataclass


@dataclass
class Config:
    random_reset = True
    step_per_collect = 2048
    training_num = 64
    single_env_step = step_per_collect // training_num

