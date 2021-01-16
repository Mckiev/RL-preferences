import os, time, datetime
import pickle
import json, csv
from stable_baselines3 import A2C
import argparse
import numpy as np
import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, VecFrameStack
import torch

def timeitt(method):
    def timed(*args, **kw):
        g = method.__globals__  
        LOG_TIME = g.get('LOG_TIME', None)
        if LOG_TIME:
            if os.path.exists(LOG_TIME):
                f = open(LOG_TIME, 'a')
            else:
                f = open(LOG_TIME, 'w')

        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        
        if LOG_TIME:
            f.write(f"{method.__name__} : {(te - ts) * 1000:2.2f}ms \n")
            f.close()
        else:
            print ('time spent by %r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed 

@timeitt
def save_state(run_dir, i, reward_model, policy, data_buffer):

    save_dir =os.path.join(run_dir, "saved_states", str(i))
    os.makedirs(save_dir, exist_ok=True)

    policy_save_path = os.path.join(save_dir, 'policy')
    rm_save_path = os.path.join(save_dir, 'rm.pth')
    data_buff_save_path = os.path.join(run_dir, 'data_buff.pth')

    with open(rm_save_path, 'wb') as f:
        pickle.dump(reward_model, f)

    with open(data_buff_save_path, 'wb') as f:
        pickle.dump(data_buffer, f)  
        

    policy.save(policy_save_path)

@timeitt
def load_state(run_dir):

    state_dir = os.path.join(run_dir, "saved_states")
    i = max([int(f.name) for f in os.scandir(state_dir) if f.is_dir()])
    load_dir =os.path.join(state_dir, str(i))

    policy_load_path = os.path.join(load_dir, 'policy')
    rm_load_path = os.path.join(load_dir, 'rm.pth')
    data_buff_load_path = os.path.join(run_dir, 'data_buff.pth')

    args_path = os.path.join(run_dir, "config.json")
    with open(args_path) as f:
        args = argparse.Namespace()
        args.__dict__.update(json.load(f))

    reward_model = pickle.load(open(rm_load_path, 'rb'))
    data_buffer = pickle.load(open(data_buff_load_path, 'rb'))
    policy = A2C.load(path = policy_load_path)

    return reward_model, policy, data_buffer, i+1


def setup_logging(args):

    #Setting up directory for logs
    if not args.log_name:
        args.log_name = args.env_name + datetime.datetime.now().strftime('_%Y_%m_%d_%H_%-M_%S')

    run_dir = os.path.join(args.log_dir, args.log_name)
    os.makedirs(run_dir, exist_ok=True)
    monitor_dir = os.path.join(run_dir ,'EnvMonitor', 'monitor')
    os.makedirs(monitor_dir, exist_ok=True)
    video_dir = os.path.join(run_dir ,'video')
    os.makedirs(video_dir, exist_ok=True)

    print('\n=== Logging ===', flush=True)
    print(f'Logging to {run_dir}', flush=True)

    return run_dir, monitor_dir, video_dir

def store_args(args, run_dir):
    args_path = os.path.join(run_dir, 'config.json')
    with open(args_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
        print(f'Config file saved to: {args_path}', flush=True)

def load_args(args):
    run_dir = os.path.join(args.log_dir, args.log_name)
    args_path = os.path.join(run_dir, "config.json")
    with open(args_path) as f:
        args = argparse.Namespace()
        args.__dict__.update(json.load(f))

    args.resume_training = True
    
    return args


def log_iter(run_dir, rl_steps, data_buffer, true_return, proxy_return, rm_train_stats, iter_time):

    info_path = os.path.join(run_dir, 'LOG.csv')

    if not os.path.exists(info_path):
        with open(info_path, 'w') as f: 
            rew_writer = csv.writer(f, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            rew_writer.writerow(['RL_steps', 'iter_time', 'buffer_size', 'true_return', 'proxy_return', 'train_loss', 'val_loss', 'min_val_loss', 'l2'])


    train_loss, val_loss, l2 = rm_train_stats
    with open(info_path, 'a') as f: 
        rew_writer = csv.writer(f, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rew_writer.writerow([rl_steps, int(iter_time), data_buffer.current_size, true_return, proxy_return, train_loss, val_loss, data_buffer.val_loss_lb, l2])


@timeitt
def eval_policy(venv, policy, n_eval_episodes, rand = False):

    finished_eps = 0
    ep_returns = []
    obs_b = venv.reset()
    returns_b = np.zeros(len(obs_b))
    while finished_eps < n_eval_episodes:
        if rand:
            action_b = 16*[venv.action_space.sample()]
        else:
            action_b , _states = policy.predict(obs_b)

        obs_b, r_b, dones, infos = venv.step(action_b)

        returns_b += r_b
        finished_eps += np.sum(dones)
        if dones.any():
            for n, done in enumerate(dones):
                if done:
                    ep_returns.append(returns_b[n])
                    returns_b[n] = 0
    venv.close()
    return np.sum(ep_returns) / finished_eps
    
@timeitt
def record_video(trained_model, env, video_folder, video_length, name):

    obs = env.reset()

    trained_model.set_env(env)
    # Record the video starting at the first step
    env = VecVideoRecorder(env, video_folder,
                   record_video_trigger=lambda x: x == 0, video_length=video_length,
                   name_prefix=name)

    env.reset()
    for _ in range(video_length + 1):
        action = trained_model.predict(obs)
        obs, _, _, _ = env.step(action[0])
    # Save the video
    env.close()


from stable_baselines3.common.callbacks import BaseCallback
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_data, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.buffer_size,  self.loss_lb, self.iter_time, (self.train_loss, self.val_loss, self.l2) = log_data

    def _on_step(self) -> bool:
        self.logger.record('RM/buffer_size', self.buffer_size)
        self.logger.record('RM/iter_time', self.iter_time)
        self.logger.record('RM/loss_lb', self.loss_lb)
        self.logger.record('RM/train_loss', self.train_loss)
        self.logger.record('RM/val_loss', self.val_loss)
        self.logger.record('RM/l2', self.l2)
        return True
