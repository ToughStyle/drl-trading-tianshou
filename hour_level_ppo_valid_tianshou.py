import pandas as pd
import numpy as np
# from processor.preprocessors import data_split
from trading_env.env_cryptotrading_tianshou import CryptoTradingEnv
from config import (
	TRAIN_START_DATE,
	VALID_2_END_DATE,
	TEST_START_DATE,
	TEST_END_DATE,
)
import argparse
import datetime
import os
import pprint

import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from tianshou.env import SubprocVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import warnings
warnings.filterwarnings("ignore")
from config_template import Config
# from stable_baselines3.common.logger import configure

def data_split(df, start, end, target_date_col="timestamp"):
	"""
    split the dataset into training or testing using timestamp
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
	data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
	data = data.sort_values([target_date_col, "tic"], ignore_index=True)
	data.index = data[target_date_col].factorize()[0]
	return data


processed_dts10_add_tech = pd.read_csv("dataset/hour-level/crypto_TI_t10_2023-04-01.csv")
state_interval = 5
TEST_START_DATE = pd.to_datetime(TEST_START_DATE)
TEST_START_DATE = TEST_START_DATE - pd.Timedelta(hours=state_interval - 1)
TEST_START_DATE = TEST_START_DATE.strftime('%Y-%m-%d %H:%M:%S')
train = data_split(processed_dts10_add_tech, TRAIN_START_DATE, VALID_2_END_DATE)
valid = data_split(processed_dts10_add_tech, TEST_START_DATE, TEST_END_DATE)

print(f'{len(train) = }')
print(f'{len(valid) = }')

crypto_tic_dim = len(train.tic.unique())
# state_space_dim = 1 + 2*crypto_tic_dim + len(INDICATORS)*crypto_tic_dim
state_space_dim = 1 + (state_interval * 5 + 1) * crypto_tic_dim
print(f"Stock Dimension: {crypto_tic_dim}, State Space: {state_space_dim}")

buy_cost_list = sell_cost_list = [0.001] * crypto_tic_dim
num_stock_shares = [0] * crypto_tic_dim

env_kwargs_train = {
	"cash": 100000,
	"action_scaling": 1 / 10,
	"num_crypto_shares": num_stock_shares,
	"buy_cost_pct": buy_cost_list,
	"sell_cost_pct": sell_cost_list,
	"state_space_dim": state_space_dim,
	"crypto_dim": crypto_tic_dim,
	"action_space_dim": crypto_tic_dim,
	"print_verbosity": 5,
	"eval_time_interval": 30 * 24,
	"is_debug": True,
	"risk_control": False,
	"model_name": "ppo_hour_level",
	"data_granularity": 24,
	"state_interval": state_interval,
	"is_training": True,
	"cfg": Config()
}
env_kwargs_test = env_kwargs_train.copy()
env_kwargs_test["is_debug"] = False
env_kwargs_test["is_training"] = False


def make_trading_envs(train_df, test_df, train_num, test_num):
	env = CryptoTradingEnv(df=train_df, **env_kwargs_train)
	
	def get_train_env():
		return CryptoTradingEnv(df=train_df, **env_kwargs_train)
	
	train_envs = SubprocVectorEnv([get_train_env for _ in range(train_num)])  # type: ignore  # noqa: E501
	
	def get_test_env():
		return CryptoTradingEnv(df=test_df, **env_kwargs_test)
	
	test_envs = SubprocVectorEnv([get_test_env for _ in range(test_num)])  # type: ignore  # noqa: E501
	return env, train_envs, test_envs


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", type=str, default="CryptoTradingEnv-v0")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--buffer-size", type=int, default=4096)
	parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--epoch", type=int, default=30)
	parser.add_argument("--step-per-epoch", type=int, default=100000)
	parser.add_argument("--step-per-collect", type=int, default=2048)
	parser.add_argument("--repeat-per-collect", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--training-num", type=int, default=64)
	parser.add_argument("--test-num", type=int, default=16)
	# ppo special
	parser.add_argument("--rew-norm", type=int, default=True)
	# In theory, `vf-coef` will not make any difference if using Adam optimizer.
	parser.add_argument("--vf-coef", type=float, default=0.25)
	parser.add_argument("--ent-coef", type=float, default=0.01)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--bound-action-method", type=str, default="clip")
	parser.add_argument("--lr-decay", type=int, default=True)
	parser.add_argument("--max-grad-norm", type=float, default=0.5)
	parser.add_argument("--eps-clip", type=float, default=0.2)
	parser.add_argument("--dual-clip", type=float, default=None)
	parser.add_argument("--value-clip", type=int, default=0)
	parser.add_argument("--norm-adv", type=int, default=0)
	parser.add_argument("--recompute-adv", type=int, default=1)
	parser.add_argument("--logdir", type=str, default="log")
	parser.add_argument("--render", type=float, default=0.)
	parser.add_argument(
		"--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
	)
	parser.add_argument("--resume-path", type=str, default=None)
	parser.add_argument("--resume-id", type=str, default=None)
	parser.add_argument(
		"--logger",
		type=str,
		default="tensorboard",
		choices=["tensorboard", "wandb"],
	)
	parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
	parser.add_argument(
		"--watch",
		default=False,
		action="store_true",
		help="watch the play of pre-trained policy only",
	)
	return parser.parse_args()


def test_ppo(args=get_args()):
	env, train_envs, test_envs = make_trading_envs(
		train, valid, args.training_num, args.test_num
	)
	args.state_shape = env.observation_space.shape or env.observation_space.n  # type: ignore
	args.action_shape = env.action_space.shape or env.action_space.n  # type: ignore
	args.max_action = env.action_space.high[0]  # type: ignore
	print("Observations shape:", args.state_shape)
	print("Actions shape:", args.action_shape)
	print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))  # type: ignore
	# seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	# model
	net_a = Net(
		args.state_shape,
		hidden_sizes=args.hidden_sizes,
		activation=nn.Tanh,
		device=args.device,
	)
	actor = ActorProb(
		net_a,
		args.action_shape,
		max_action=args.max_action,
		# unbounded=True,
		device=args.device,
	).to(args.device)
	net_c = Net(
		args.state_shape,
		hidden_sizes=args.hidden_sizes,
		activation=nn.Tanh,
		device=args.device,
	)
	critic = Critic(net_c, device=args.device).to(args.device)
	torch.nn.init.constant_(actor.sigma_param, -0.5)
	for m in list(actor.modules()) + list(critic.modules()):
		if isinstance(m, torch.nn.Linear):
			# orthogonal initialization
			torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
			torch.nn.init.zeros_(m.bias)
	# do last policy layer scaling, this will make initial actions have (close to)
	# 0 mean and std, and will help boost performances,
	# see https://arxiv.org/abs/2006.05990, Fig.24 for details
	for m in actor.mu.modules():
		if isinstance(m, torch.nn.Linear):
			torch.nn.init.zeros_(m.bias)
			m.weight.data.copy_(0.01 * m.weight.data)
	
	optim = torch.optim.Adam(
		list(actor.parameters()) + list(critic.parameters()), lr=args.lr
	)
	
	lr_scheduler = None
	if args.lr_decay:
		# decay learning rate to 0 linearly
		max_update_num = np.ceil(
			args.step_per_epoch / args.step_per_collect
		) * args.epoch
		
		lr_scheduler = LambdaLR(
			optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
		)
	
	def dist(*logits):
		return Independent(Normal(*logits), 1)
	
	policy = PPOPolicy(
		actor,
		critic,
		optim,
		dist,  # type: ignore
		discount_factor=args.gamma,
		gae_lambda=args.gae_lambda,
		max_grad_norm=args.max_grad_norm,
		vf_coef=args.vf_coef,
		ent_coef=args.ent_coef,
		reward_normalization=args.rew_norm,
		action_scaling=True,
		action_bound_method=args.bound_action_method,
		lr_scheduler=lr_scheduler,
		action_space=env.action_space,
		eps_clip=args.eps_clip,
		value_clip=args.value_clip,
		dual_clip=args.dual_clip,
		advantage_normalization=args.norm_adv,
		recompute_advantage=args.recompute_adv,
	)
	
	# load a previous policy
	if args.resume_path:
		ckpt = torch.load(args.resume_path, map_location=args.device)
		policy.load_state_dict(ckpt["model"])
		train_envs.set_obs_rms(ckpt["obs_rms"])
		test_envs.set_obs_rms(ckpt["obs_rms"])
		print("Loaded agent from: ", args.resume_path)
	
	# collector
	if args.training_num > 1:
		buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
	else:
		buffer = ReplayBuffer(args.buffer_size)
	train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
	test_collector = Collector(policy, test_envs)
	
	# log
	now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
	args.algo_name = "ppo"
	log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
	log_path = os.path.join(args.logdir, log_name)
	
	# logger
	if args.logger == "wandb":
		logger = WandbLogger(
			save_interval=1,
			name=log_name.replace(os.path.sep, "__"),
			run_id=args.resume_id,
			config=args,
			project=args.wandb_project,
		)
	writer = SummaryWriter(log_path)
	writer.add_text("args", str(args))
	if args.logger == "tensorboard":
		logger = TensorboardLogger(writer)
	else:  # wandb
		logger.load(writer)
	
	def save_best_fn(policy):
		state = {"model": policy.state_dict()}
		torch.save(state, os.path.join(log_path, "policy.pth"))
	
	if not args.watch:
		# trainer
		result = onpolicy_trainer(
			policy,
			train_collector,
			test_collector,
			args.epoch,
			args.step_per_epoch,
			args.repeat_per_collect,
			args.test_num,
			args.batch_size,
			step_per_collect=args.step_per_collect,
			save_best_fn=save_best_fn,
			logger=logger,
			test_in_train=False,
		)
		pprint.pprint(result)
	
	# Let's watch its performance!
	policy.eval()
	# test_envs.seed(args.seed)
	test_collector.reset()
	result = test_collector.collect(n_episode=args.test_num, render=args.render)
	print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
	test_ppo()
