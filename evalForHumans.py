import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch

import discrete.DDQN as DDQN
import discrete.PER_DDQN
import discrete.LAP_DDQN
import discrete.PAL_DDQN
import discrete.utils as utils
from time import sleep



def main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device):
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

    if args.algorithm == "DDQN":
        policy = DDQN.DDQN(**kwargs)
        policy.load(f"./results/policyThingy", 24000)

    kwargs["alpha"] = parameters["alpha"]
    kwargs["min_priority"] = parameters["min_priority"]
    kwargs["render_mode"] = 'human'

    eval_policy(policy, args.env, args.seed)



def eval_policy(policy, env_name, seed, eval_episodes=10):
    atari_preprocessing['render_mode'] = 'human'
    eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing, render_mode='human')
    if hasattr(env, 'seed'):
        eval_env.seed(seed + 100)


    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        if not isinstance(state, list):
            state = state[0]
        while not done:
            action = policy.select_action(np.array(state), eval=True)
            eval_env.render()
            sleep(0.03)
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
    parser.add_argument("--seed", default=0, type=int)				# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Default")			# Prepends name to filename
    # parser.add_argument("--max_timesteps", default=50e6, type=int)	# Max time steps to run environment or train for
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

    if hasattr(env, 'seed'):
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

    main(env, replay_buffer, is_atari, state_dim, num_actions, args, parameters, device)
