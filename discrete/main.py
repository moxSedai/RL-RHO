import argparse
import copy
import importlib
import json
import os
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np
import torch

import DDQN
import PER_DDQN
import LAP_DDQN
import PAL_DDQN
import utils


def coreset(policy, holdout_policy, replay_buffer, holdout_buffer, coreset_base, coreset_size, coreset_batch_size, add_count, args, kwargs):
	coreset = copy.deepcopy(coreset_base)	# I don't understand how the buffers actually work so I'm just copying through an empty one all the time
	coreset_set = set()

	# PERCENTAGE CORESET SIZE
	coreset_size = int(replay_buffer.size / 2)
	while(len(coreset_set) < coreset_size):
		# Sample replay buffer
		state, action, next_state, reward, done, first_timestep = replay_buffer.sample(coreset_batch_size)

		# Compute the target Q value for the main policy
		with torch.no_grad():
			next_action = policy.Q(next_state).argmax(1, keepdim=True)
			target_Q = (
					reward + done * policy.discount *
					policy.Q_target(next_state).gather(1, next_action).reshape(-1, 1)
			)

		# Compute the target Q value for the holdout policy
		with torch.no_grad():
			holdout_next_action = holdout_policy.Q(next_state).argmax(1, keepdim=True)
			holdout_target_Q = (
					reward + done * holdout_policy.discount *
					holdout_policy.Q_target(next_state).gather(1, next_action).reshape(-1, 1)
			)


		# Get current Q estimate for policy
		current_Q = policy.Q(state).gather(1, action)

		# td_loss = (current_Q - target_Q).abs()
		Q_loss = F.smooth_l1_loss(current_Q, target_Q, reduce=False)
		# weight = td_loss.clamp(min=policy.min_priority).pow(policy.alpha).mean().detach()

		# Compute critic loss
		# Q_loss = policy.PAL(td_loss) / weight.detach()


		# Get current Q estimate for holdout policy
		holdout_current_Q = holdout_policy.Q(state).gather(1, action)

		# holdout_td_loss = (holdout_current_Q - holdout_target_Q).abs()
		holdout_Q_loss = F.smooth_l1_loss(holdout_current_Q, holdout_target_Q, reduce=False)
		# holdout_weight = td_loss.clamp(min=holdout_policy.min_priority).pow(holdout_policy.alpha).mean().detach()

		# Compute critic loss
		# holdout_Q_loss = holdout_policy.PAL(td_loss) / weight.detach()

		# Compute RHO loss
		# rho_loss = td_loss - holdout_td_loss
		rho_loss = Q_loss - holdout_Q_loss
		order = torch.argsort(rho_loss, 0).flip(0)
		num_to_add = min(coreset_size - len(coreset_set), add_count)

		# Add the top num_to_add elements to the coreset
		for i in range(num_to_add):
			coreset_set.add((state[order[i]], action[order[i]], next_state[order[i]], reward[order[i]], done[order[i]], done[order[i]], first_timestep[order[i]]))

	for elm in coreset_set:
		coreset.add(elm[0].cpu().numpy(), elm[1].cpu().numpy(), elm[2].cpu().numpy(), elm[3].cpu().numpy(), elm[4].cpu().numpy(), elm[5].cpu().numpy(), elm[6].cpu().numpy())

	return copy.deepcopy(coreset), copy.deepcopy(coreset)



def normal_training(env, replay_buffer, args, kwargs):
	if args.algorithm == "DDQN":
		policy = DDQN.DDQN(**kwargs)
	elif args.algorithm == "PER_DDQN":
		policy = PER_DDQN.PER_DDQN(**kwargs)

	kwargs["alpha"] = parameters["alpha"]
	kwargs["min_priority"] = parameters["min_priority"]

	if args.algorithm == "LAP_DDQN":
		policy = LAP_DDQN.LAP_DDQN(**kwargs)
	elif args.algorithm == "PAL_DDQN":
		policy = PAL_DDQN.PAL_DDQN(**kwargs)

	evaluations = []

	state, done = env.reset(), False
	if not isinstance(state, list):
		state = state[0]
	episode_start = True
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# if args.train_behavioral:
		if t < parameters["start_timesteps"]:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(state))

		# Perform action and log results
		if (hasattr(env, 'spec')):
			next_state, reward, done, info, _ = env.step(action)
		else:
			next_state, reward, done, info = env.step(action)
		episode_reward += reward

		# Only consider "done" if episode terminates due to failure condition
		done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

		# For atari, info[0] = clipped reward, info[1] = done_float
		if is_atari:
			reward = info[0]
			done_float = info[1]

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
		state = copy.copy(next_state)
		episode_start = False

		# Train agent after collecting sufficient data
		if t >= parameters["start_timesteps"] and (t + 1) % parameters["train_freq"] == 0:
			policy.train(replay_buffer)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(
				f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			if not isinstance(state, list):
				state = state[0]
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % parameters["eval_freq"] == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{setting}.npy", evaluations)
			policy.save(f"./results/policyThingyPong")
	plt.scatter(range(len(evaluations)), evaluations, title="Normal Training Evaluation Rewards over epochs")


def rho_training(env, replay_buffer, holdout_replay_buffer, coreset_base, coreset_freq, coreset_size, coreset_batch_size, coreset_add_size, args, kwargs):
	# Setup for training
	policy = DDQN.DDQN(**kwargs)
	holdout_policy = DDQN.DDQN(**kwargs)

	kwargs["alpha"] = parameters["alpha"]
	kwargs["min_priority"] = parameters["min_priority"]

	evaluations = []

	state, done = env.reset(), False
	if not isinstance(state, list):
		state = state[0]
	episode_start = True
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):
		# Everything here is normal
		episode_timesteps += 1

		# if args.train_behavioral:
		if t < parameters["start_timesteps"]:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(state))

		# Perform action and log results
		if (hasattr(env, 'spec')):
			next_state, reward, done, info, _ = env.step(action)
		else:
			next_state, reward, done, info = env.step(action)
		episode_reward += reward

		# Only consider "done" if episode terminates due to failure condition
		done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

		# For atari, info[0] = clipped reward, info[1] = done_float
		if is_atari:
			reward = info[0]
			done_float = info[1]

		# Now deal with holdout vs main policy
		if random.randint(0, 4) != 0:
			# ============================================================ #
			# ========== Train on normal policy 80% of the time ========== #
			# ============================================================ #

			# Store data in replay buffer
			replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
			state = copy.copy(next_state)
			episode_start = False

			# Train agent after collecting sufficient data
			if t >= parameters["start_timesteps"] and (t + 1) % parameters["train_freq"] == 0:
				policy.train(replay_buffer)

		else:
			# ============================================================= #
			# ========== Train on holdout policy 80% of the time ========== #
			# ============================================================= #
			# Store data in holdout replay buffer
			holdout_replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
			state = copy.copy(next_state)
			episode_start = False

			# Train agent after collecting sufficient data
			if t >= parameters["start_timesteps"] and (t + 1) % parameters["train_freq"] == 0:
				holdout_policy.train(holdout_replay_buffer)

		# Back to normal
		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(
				f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Buffer Size: {replay_buffer.size}")
			# Reset environment
			state, done = env.reset(), False
			if not isinstance(state, list):
				state = state[0]
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % parameters["eval_freq"] == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{setting}.npy", evaluations)
			policy.save(f"./results/policyThingyPong")

		# Create coreset
		if t % coreset_freq == 0:
			replay_buffer, holdout_replay_buffer = coreset(policy, holdout_policy, replay_buffer, holdout_replay_buffer, coreset_base, coreset_size, coreset_batch_size, coreset_add_size, args, kwargs)
			print(f"======================Created coreset at timestep {t}======================")

	plt.scatter(range(len(evaluations)), evaluations, title="Rho Training Evaluation Rewards over epochs")



def main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device, rho_buffer=None, coreset_base=None, coreset_freq=None,  coreset_size=None, coreset_batch_size=None, coreset_add_size=None):
	# Initialize and load policy
	kwargs = {
		"is_atari": is_atari,
		"num_actions": num_actions,
		"state_dim": state_dim,
		"device": device,
		"discount": parameters["discount"],
		"optimizer": parameters["optimizer"],
		"optimizer_parameters": parameters["optimizer_parameters"],
		"polyak_target_update": parameters["polyak_target_update"],
		"target_update_frequency": parameters["target_update_freq"],
		"tau": parameters["tau"],
		"initial_eps": parameters["initial_eps"],
		"end_eps": parameters["end_eps"],
		"eps_decay_period": parameters["eps_decay_period"],
		"eval_eps": parameters["eval_eps"]
	}

	if not rho:
		normal_training(env, replay_buffer, args, kwargs)
	else:
		rho_training(env, replay_buffer, rho_buffer, coreset_base, coreset_freq, coreset_size, coreset_batch_size, coreset_add_size, args, kwargs)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
	if hasattr(env, 'seed'):
		eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		if not isinstance(state, list) or not isinstance(state, np.ndarray):
			state = state[0]
		while not done:
			action = policy.select_action(np.array(state), eval=True)
			if(hasattr(eval_env, 'spec')):
				state, reward, done, _, _ = eval_env.step(action)
			else:
				state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	rho = True
	coreset_size = 500
	coreset_batch_size = 512
	coreset_add_size = 16
	coreset_freq = 1000

	# Atari Specific
	atari_preprocessing = {
		"frame_skip": 4,
		"frame_size": 84,
		"state_history": 4,
		"done_on_life_loss": False,
		"reward_clipping": True,
		"max_episode_timesteps": 27e3
	}

	atari_parameters = {
		# LAP/PAL
		"alpha": 0.6,
		"min_priority": 1e-2,
		# Exploration
		"start_timesteps": 2e4,
		"initial_eps": 1,
		"end_eps": 1e-2,
		"eps_decay_period": 25e4,
		# Evaluation
		"eval_freq": 5e4,
		"eval_eps": 1e-3,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 32,
		"optimizer": "RMSprop",
		"optimizer_parameters": {
			"lr": 0.0000625,
			"alpha": 0.95,
			"centered": True,
			"eps": 0.00001
		},
		"train_freq": 4,
		"polyak_target_update": False,
		"target_update_freq": 8e3,
		"tau": 1
	}

	regular_parameters = {
		# LAP/PAL
		"alpha": 0.4,
		"min_priority": 1,
		# Exploration
		"start_timesteps": 1e3,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 5e3,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 64,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--algorithm", default="DDQN")				# OpenAI gym environment name
	parser.add_argument("--env", default="CartPole-v1")		# OpenAI gym environment name #PongNoFrameskip-v0
	#parser.add_argument("--env", default="PongNoFrameskip-v0")  # OpenAI gym environment name #PongNoFrameskip-v0
	parser.add_argument("--seed", default=0, type=int)				# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")			# Prepends name to filename
	parser.add_argument("--max_timesteps", default=50e6, type=int)	# Max time steps to run environment or train for
	args = parser.parse_args()

	print("---------------------------------------")	
	print(f"Setting: Algorithm: {args.algorithm}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	setting = f"{args.algorithm}_{args.env}_{args.seed}"

	if not os.path.exists("./results"):
		os.makedirs("./results")



	# Make env and determine properties
	env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
	parameters = atari_parameters if is_atari else regular_parameters

	if hasattr(env, 'spec') and env.spec.name != 'CartPole':
		env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	# Initialize buffer
	prioritized = True if args.algorithm == "PER_DDQN" or args.algorithm == "LAP_DDQN" else False
	replay_buffer = utils.ReplayBuffer(
		state_dim, 
		prioritized, 
		is_atari, 
		atari_preprocessing, 
		parameters["batch_size"], 
		parameters["buffer_size"], 
		device
	)
	if rho:
		holdout_replay_buffer = utils.ReplayBuffer(
		state_dim,
		prioritized,
		is_atari,
		atari_preprocessing,
		parameters["batch_size"],
		parameters["buffer_size"],
		device)

		coreset_base = utils.ReplayBuffer(
		state_dim,
		prioritized,
		is_atari,
		atari_preprocessing,
		parameters["batch_size"],
		parameters["buffer_size"],
		device)

		main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device, holdout_replay_buffer, coreset_base, coreset_freq, coreset_size, coreset_batch_size, coreset_add_size)

	elif not rho:
		main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device)

