import argparse
import copy
import importlib
import json
import os
import random
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import pickle

import numpy as np
import torch

import logging

import DDQN
import PER_DDQN
import LAP_DDQN
import PAL_DDQN
import utils


# Class to log print statements
class PrintToLog:

	def __init__(self):
		self.terminal = sys.__stdout__
	def write(self, text):
		logging.info(text)
		self.terminal.write(text)

	def flush(self):
		pass


def supervised_learning(env, pretrained_names, coreset_size, coreset_batch_size, add_count, max_epochs, args, kwargs):
	# Create policy
	policy = DDQN.DDQN(**kwargs)
	holdout_policy = DDQN.DDQN(**kwargs)

	# Load holdout policy and buffer
	replay_buffer = None
	print("Loading buffer...", end='')
	with open(f"{pretrained_names[1]}", 'rb') as f:
		replay_buffer = pickle.load(f)
		print("loaded!")
	holdout_policy.load(f"{pretrained_names[0]}", 832500)


	# Get true coreset size
	coreset_size = coreset_size * len(replay_buffer.buffer)

	# Setup coreset tracker
	coreset_set = set()

	# Setup for training
	evaluations = []
	start_time = time.time()

	calc_losses = True

	# Get the holdout loss for all experiences in the replay buffer
	if calc_losses:
		loss_calc_batch_size = 512
		num_batches = len(replay_buffer.buffer) // loss_calc_batch_size
		if len(replay_buffer.buffer) % loss_calc_batch_size != 0:
			num_batches += 1
		holdout_loss = torch.empty(0)

		print(f"Calculating holdout loss...(0/{num_batches})", end='')
		with torch.no_grad():
			for i in range(num_batches):
				batch = replay_buffer.buffer[i * loss_calc_batch_size: min((i + 1) * loss_calc_batch_size, len(replay_buffer.buffer))]
				batch_elements = (
					torch.ByteTensor(np.array([single.state for single in batch])).to(device).float(),
					torch.unsqueeze(torch.LongTensor(np.array([single.action for single in batch])), 1).to(device),
					torch.ByteTensor(np.array([single.next_state for single in batch])).to(device).float(),
					torch.unsqueeze(torch.FloatTensor(np.array([single.reward for single in batch])), 1).to(device),
					torch.unsqueeze(torch.FloatTensor(np.array([single.not_done for single in batch])), 1).to(device)
				)
				state, action, next_state, reward, done = batch_elements
				# Compute the target Q value for the holdout policy
				with torch.no_grad():
					holdout_next_action = holdout_policy.Q(next_state).argmax(1, keepdim=True)
					holdout_target_Q = (
							reward + done * holdout_policy.discount *
							holdout_policy.Q_target(next_state).gather(1, holdout_next_action).reshape(-1, 1)
					)
				# Compute the current Q estimate for the holdout policy
				holdout_current_Q = holdout_policy.Q(state).gather(1, action)
				holdout_Q_loss = F.smooth_l1_loss(holdout_current_Q, holdout_target_Q, reduce=False)
				holdout_loss = torch.concatenate((holdout_loss, holdout_Q_loss.cpu()))
				print(f"\rCalculating holdout loss...({i+1}/{num_batches})", end='')
		torch.save(holdout_loss, f"./results/PongNoFrameskip-v0_250/holdout_loss")


	else:
		print("Loading loss...", end='')
		holdout_loss = torch.load(f"./results/PongNoFrameskip-v0_250/holdout_loss")
		print("loaded!")

	# Training loop of policy
	for t in range(max_epochs):
		epoch_coreset_set = set()
		while len(epoch_coreset_set) < coreset_size:
			# Sample a batch from the replay buffer
			sample_objects, sample, indices = replay_buffer.sample(coreset_batch_size, with_indices=True, device_override=device)
			state, action, next_state, reward, done = sample

			# Get online loss for all experiences in the batch
			online_loss = torch.empty(0)

			# Compute the target Q value for the main policy
			with torch.no_grad():
				next_action = policy.Q(next_state).argmax(1, keepdim=True)
				target_Q = (
						reward + done * policy.discount *
						policy.Q_target(next_state).gather(1, next_action).reshape(-1, 1)
				)

			# Compute the current Q estimate for the main policy
			current_Q = policy.Q(state).gather(1, action)
			Q_loss = F.smooth_l1_loss(current_Q, target_Q, reduce=False)
			online_loss = torch.concatenate((online_loss, Q_loss.cpu()))

			# Get the rho loss by taking the differences
			rho_loss = online_loss - holdout_loss[indices]

			# Sort the experiences based on the rho_loss in descending order
			order = torch.argsort(rho_loss, 0).flip(0)
			order_list = order.cpu().numpy().flatten()

			# Determine how many experiences to add to the coreset in this iteration
			num_to_add = int(min(coreset_size - len(epoch_coreset_set), add_count))

			# Reorder samples
			sample = (
				sample[0][order_list[:add_count]],
				sample[1][order_list[:add_count]],
				sample[2][order_list[:add_count]],
				sample[3][order_list[:add_count]],
				sample[4][order_list[:add_count]]
			)

			num_added = 0
			num_full_added = 0
			# Train new experiences to coresets
			for i in range(num_to_add):
				# Add the top un-added experiences to the epoch coreset_set (don't allow for duplicates)
				if not indices[order[i]] in epoch_coreset_set:
					epoch_coreset_set.add(indices[order[i]])
					num_added += 1
					if not indices[order[i]] in coreset_set:
						coreset_set.add(indices[order[i]])
						num_full_added += 1

			# Train on this chosen batch
			policy.train_supervised(sample)

			print(f"\rEpoch: {t}\t\tEpoch Coreset Size: {100.*len(epoch_coreset_set)/len(replay_buffer.buffer):.5f}% ({num_added})\t\tFull Coreset Size: {100.*len(coreset_set)/len(replay_buffer.buffer):.5f}% ({num_full_added})", end='')


		# Evaluate after each epoch
		elapsed_time = time.time() - start_time
		evaluations.append(eval_policy(policy, args.env, args.seed, timer=elapsed_time))
		plt.scatter(range(len(evaluations)), evaluations)
		plt.title(f"Supervised Training {args.env} {args.seed} {time.time()-start_time}")
		plt.show()










# todo check if using fully trained network as holdout (WITH DIFFERENT SEED) works well for training

# todo collect the full buffer then train policy on supervise learning from the buffer (create coreset from buffer)
	# Imitation Learning
	# Use this to see how much data is actually needed

def coreset(policy, holdout_policy, replay_buffer, holdout_buffer, coreset_base, coreset_size, coreset_batch_size, add_count, args, kwargs):
	coreset = copy.deepcopy(coreset_base)  # I don't understand how the buffers actually work so I'm just copying through an empty one all the time
	if coreset_size == 1:
		coreset.buffer = replay_buffer.buffer
		coreset.size = replay_buffer.size
		return copy.copy(coreset), copy.copy(coreset)



	# Create a deep copy of the coreset_base which is an empty replay buffer

	# Initialize a set to store unique experiences for the coreset
	coreset_indices_set = set()

	# Convert the coreset_size from a percentage to an absolute number based on the size of the replay buffer
	coreset_size = int(replay_buffer.size * coreset_size)
	coreset.size = coreset_size



	# Keep adding experiences to the coreset until it reaches the desired size
	while(len(coreset_indices_set) < coreset_size):
		# Sample a batch of experiences from the replay buffer
		sample_objects, sample, indices = replay_buffer.sample(coreset_batch_size, with_indices=True)
		state, action, next_state, reward, done = sample

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
					holdout_policy.Q_target(next_state).gather(1, holdout_next_action).reshape(-1, 1)
			)

		# Compute the current Q estimate for the main policy
		current_Q = policy.Q(state).gather(1, action)
		Q_loss = F.smooth_l1_loss(current_Q, target_Q, reduce=False)
		# td_loss = (current_Q - target_Q).abs()
		# weight = td_loss.clamp(min=policy.min_priority).pow(policy.alpha).mean().detach()
		# Compute critic loss
		# Q_loss = policy.PAL(td_loss) / weight.detach()


		# Compute the current Q estimate for the holdout policy
		holdout_current_Q = holdout_policy.Q(state).gather(1, action)
		holdout_Q_loss = F.smooth_l1_loss(holdout_current_Q, holdout_target_Q, reduce=False)
		# holdout_td_loss = (holdout_current_Q - holdout_target_Q).abs()
		# holdout_weight = td_loss.clamp(min=holdout_policy.min_priority).pow(holdout_policy.alpha).mean().detach()
		# Compute critic loss
		# holdout_Q_loss = holdout_policy.PAL(td_loss) / weight.detach()
		# Compute RHO loss
		# rho_loss = td_loss - holdout_td_loss

		# Compute the difference between the Q losses of the main policy and the holdout policy
		rho_loss = Q_loss - holdout_Q_loss

		# Sort the experiences based on the rho_loss in descending order
		order = torch.argsort(rho_loss, 0).flip(0)

		# Determine the number of experiences to add to the coreset in this iteration
		num_to_add = min(coreset_size - len(coreset_indices_set), add_count)

		# Add the top num_to_add un-added experiences to the coreset_set (don't allow for duplicates)
		i, added = 0, 0
		while added < num_to_add and i < len(order):
			if not indices[order[i]] in coreset_indices_set:
				added += 1
				coreset_indices_set.add(indices[order[i]])
				coreset.buffer.append(sample_objects[order[i]])
			i += 1

	# for elm in coreset_set:
	# 	coreset.add(elm[0].cpu().numpy(), elm[1].cpu().numpy(), elm[2].cpu().numpy(), elm[3].cpu().numpy(), elm[4].cpu().numpy(), elm[5].cpu().numpy(), elm[6].cpu().numpy())

	return copy.copy(coreset), copy.copy(coreset)



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

	start_time = time.time()

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
		replay_buffer.add(state, action, next_state[1], reward, done_float, done, episode_start)
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
			if not isinstance(state, list) and not isinstance(state, np.ndarray):
				state = state[0]
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

			# Reset the slide
			replay_buffer.curBufferInstance = utils.SlidingAtariBufferInstance(atari_preprocessing, device)

		# Evaluate episode
		if (t + 1) % parameters["eval_freq"] == 0:
			elapsed_time = time.time() - start_time
			evaluations.append(eval_policy(policy, args.env, args.seed, timer=elapsed_time))
			np.save(f"./results/{setting}.npy", evaluations)
			policy.save(f"./results/{args.env}_{args.seed}/holdout")
			with open(f"./results/{args.env}_{args.seed}/intermediary_buffer_{policy.iterations}.pkl", 'wb') as f:
				pickle.dump(replay_buffer, f)
			plt.scatter(range(len(evaluations)), evaluations)
			plt.title(f"Normal Training Evaluation Rewards over epochs Seed:{args.seed} iter:{policy.iterations}")
			plt.savefig(f"./results/{args.env}_{args.seed}/output_policy_{args.env}_{args.seed}_{policy.iterations}.png")
			plt.close()
	plt.scatter(range(len(evaluations)), evaluations)
	plt.title(f"Normal Training Evaluation Rewards over epochs {time.time() - start_time}")
	plt.show()
	policy.save(f"./results/{args.env}_{args.seed}/output_policy_{args.env}_{args.seed}")
	with open(f"final_buffer_{args.env}_{args.seed}.pkl", 'wb') as f:
		pickle.dump(replay_buffer, f)


def rho_training(env, replay_buffer, holdout_replay_buffer, coreset_base, coreset_freq, coreset_size, coreset_batch_size, coreset_add_size, args, kwargs, slide, name='', ):
	# Setup for training
	policy = DDQN.DDQN(**kwargs)
	holdout_policy = DDQN.DDQN(**kwargs)
	# load holdout policy also when turning off TURN BACK ON SELECTING HOLDOUT POLICY
	holdout_policy.load(f"./results/{args.env}_250/holdout", 832500)

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

	its_holdout = 0
	its_main = 0

	# Start timer
	start_time = time.time()

	holdout = False

	print("Total T | Episode Num | Episode T | Reward | Buffer Size | Holdout Buffer Size")


	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):
		# Everything here is normal
		episode_timesteps += 1


		# Train the holdout policy for a number of steps first.
		# if t < parameters["holdout_timesteps"]:
		# 	policy_used = holdout_policy
		# 	holdout = True

		# Get value to split between main and holdout policy
		if random.randint(0, 4) != -1:  # MAKE SURE TO RESET TO != 0 IF NOT USING PRETRAINED
			# Copy buffer state over if switching from holdout
			if holdout:
				replay_buffer.curBufferInstance = copy.copy(holdout_replay_buffer.curBufferInstance)
			holdout = False
			policy_used = policy
		else:
			if not holdout:
				replay_buffer.curBufferInstance = copy.copy(holdout_replay_buffer.curBufferInstance)
			holdout = True
			policy_used = holdout_policy

		# if args.train_behavioral:
		if t < parameters["start_timesteps"]:
			action = env.action_space.sample()
		else:
			action = policy_used.select_action(np.array(state))

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
		if not holdout:
			# ============================================================ #
			# ========== Train on normal policy 80% of the time ========== #
			# ============================================================ #

			# Store data in replay buffer
			replay_buffer.add(state, action, next_state[1], reward, done_float, done, episode_start)

			state = copy.copy(next_state)
			episode_start = False

			its_main += 1

			# Train agent after collecting sufficient data
			if its_main >= parameters["start_timesteps"] and (its_main + 1) % parameters["train_freq"] == 0:
				policy.train(replay_buffer)

		else:
			# ============================================================= #
			# ========== Train on holdout policy 20% of the time ========== #
			# ============================================================= #
			# Store data in holdout replay buffer
			holdout_replay_buffer.add(state, action, next_state[1], reward, done_float, done, episode_start)
			state = copy.copy(next_state)
			episode_start = False

			its_holdout += 1

			# Train agent after collecting sufficient data
			if its_holdout >= parameters["start_timesteps"] and (its_holdout + 1) % parameters["train_freq"] == 0:
				holdout_policy.train(holdout_replay_buffer)

		if hasattr(env, 'spec'):
			if episode_timesteps > env.spec.max_episode_steps:
				done = True

		# Back to normal
		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"{t+1:^7} | {episode_num+1:^11} | {episode_timesteps+1:^9} | {episode_reward:^6.3f} | {replay_buffer.size:^11} | {holdout_replay_buffer.size}")
			# print(
			# 	f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Buffer Size: {replay_buffer.size} Holdout Buffer Size: {holdout_replay_buffer.size}")
			# Reset environment
			state, done = env.reset(), False
			if not isinstance(state, list) and not isinstance(state, np.ndarray):
				state = state[0]
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

			# Reset the slides
			replay_buffer.curBufferInstance = utils.SlidingAtariBufferInstance(atari_preprocessing, device)
			holdout_replay_buffer.curBufferInstance = utils.SlidingAtariBufferInstance(atari_preprocessing, device)

		# Evaluate episode
		if (t + 1) % parameters["eval_freq"] == 0:
			elapsed_time = time.time() - start_time
			evaluations.append(eval_policy(policy, args.env, args.seed, timer=elapsed_time))
			np.save(f"./results/{setting}.npy", evaluations)
			policy.save(f"./results/policyThingyPong")
			print("Total T | Episode Num | Episode T | Reward | Buffer Size | Holdout Buffer Size")

		# Create coreset
		if t > 0 and t % coreset_freq == 0 and t > parameters["holdout_timesteps"]:
			# print(replay_buffer)
			# Store current buffer objects
			cur_replay_object, cur_holdout_object = replay_buffer.curBufferInstance, holdout_replay_buffer.curBufferInstance

			replay_buffer, holdout_replay_buffer = coreset(policy, holdout_policy, replay_buffer, holdout_replay_buffer, coreset_base, coreset_size, coreset_batch_size, coreset_add_size, args, kwargs)
			replay_buffer.curBufferInstance = cur_replay_object
			holdout_replay_buffer.curBufferInstance = cur_holdout_object

			# print(replay_buffer)
			print(f"======================Created coreset at timestep {t}======================")

	plt.scatter(range(len(evaluations)), evaluations)
	plt.title(f"Rho Training Evaluation Rewards over epochs {name} {time.time()-start_time}")
	plt.show()


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

	if supervised:
		pretrained_names = ["./results/PongNoFrameskip-v0_250/holdout", "./results/final_buffer_PongNoFrameskip-v0_250.pkl"]
		supervised_learning(env, pretrained_names, coreset_size, coreset_batch_size, coreset_add_size, 10000, args, kwargs)

	elif not rho:
		normal_training(env, replay_buffer, args, kwargs)
	else:
		rho_training(env, replay_buffer, rho_buffer, coreset_base, coreset_freq, coreset_size, coreset_batch_size, coreset_add_size, args, kwargs, f"{coreset_freq} {coreset_size}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, timer=0.0):
	eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
	if hasattr(env, 'seed'):
		eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		cur_reward = 0
		state, done = eval_env.reset(), False
		if not isinstance(state, list) and not isinstance(state, np.ndarray):
			state = state[0]
		while not done:
			action = policy.select_action(np.array(state), eval=True)
			if(hasattr(eval_env, 'spec')):
				state, reward, done, _, _ = eval_env.step(action)
				cur_reward += reward
				if cur_reward >= eval_env.spec.max_episode_steps:
					done = True
			else:
				state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} {timer:3f} seconds")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
		# Initialize logger
	logger = logging.getLogger(__name__)
	logging.basicConfig(filename='main.log', level=logging.DEBUG)
	logger.info("Starting main.py")
	sys.stdout = PrintToLog()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	supervised = False
	# Set Rho Parameters
	rho = False
	coreset_size = 0.2
	coreset_batch_size = 512
	coreset_add_size = 16
	coreset_freq = 25000

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
		"tau": 1,
		"holdout_timesteps": 0
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
	# parser.add_argument("--env", default="CartPole-v1")		# OpenAI gym environment name #PongNoFrameskip-v0
	parser.add_argument("--env", default="PongNoFrameskip-v0")  # OpenAI gym environment name #PongNoFrameskip-v0
	parser.add_argument("--seed", default=2000, type=int)				# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")			# Prepends name to filename
	parser.add_argument("--max_timesteps", default=5e6, type=int)	# Max time steps to run environment or train for
	args = parser.parse_args()

	print("---------------------------------------")	
	print(f"Setting: Algorithm: {args.algorithm}, Env: {args.env}, Seed: {args.seed}, Device: {device}")
	print("---------------------------------------")

	setting = f"{args.algorithm}_{args.env}_{args.seed}"

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists(f"./results/{args.env}_{args.seed}"):
		os.makedirs(f"./results/{args.env}_{args.seed}")



	# Make env and determine properties
	env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
	parameters = atari_parameters if is_atari else regular_parameters

	if hasattr(env, 'spec') and env.spec.name != 'CartPole':
		env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)


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

	if supervised:
		main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device, coreset_size=coreset_size, coreset_batch_size=coreset_batch_size, coreset_add_size=coreset_add_size)


	elif rho:
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

		slide = utils.SlidingAtariBufferInstance(atari_preprocessing, device)

		main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device, holdout_replay_buffer, coreset_base, coreset_freq, coreset_size, coreset_batch_size, coreset_add_size)

	elif not rho:
		main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device)

# plot reward over time (time = # steps, rather than evaluations) (done, ish)
# compare to no coreset (done)
# compare low coreset % to high coreset %
	# can low coreset % be more stable but take more steps to train?




# todo read papers again

# todo for pong, need to change datastructure.  Must have buffer elements be sets of 4 frames.  (Start with frames being all 0, every time a new frame is there, shift the previous frames left (deleting the 4th frame) and add the new new one to the front (AKA rolling 4-frame window) -- not necessarily 4, set to same as state history
	# todo How to change loss to work for this?


	# baseline for smoe seeds, then with coresetting
# don't forget, can use both GPUs

# Once everything else is done, investigate episode-wise rather than iteration-wise swaps between online and holdout
# Also another extended goal: After training a policy with standard RL, save the buffer see if we can train a new policy with supervised learning (treat buffer as dataset)
