import torch.nn as nn
import torch.optim as optim
import torch

import gym

from stable_baselines3 import A2C
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import safe_mean

#from register_policies import ImpalaPolicy
from utils import *
from env_wrapper import *

import numpy as np
import random
import argparse, pickle
import multiprocessing

import os, time, datetime, sys

class AnnotationBuffer(object):
    """Buffer of annotated pairs of clips

    Each entry is ([clip0, clip1], label)
    clip0, clip2 : lists of observations
    label : float in range {0, 0.5, 1} corresponding to which clip is preferred,
    where 0.5 means that clips are equal
    """

    def __init__(self, max_size=3000):
        self.max_size = max_size
        self.current_size = 0

        #calculate max train and validation set sizes based on total max_size
        self.train_max_size = int(self.max_size * (1 - 1/np.exp(1)))
        self.val_max_size = self.max_size - self.train_max_size

        self.train_data_all = []
        self.val_data_all = []


    def add(self, data):
        '''
        1/e of data goes to the validatation set
        the rest goes to the training set
        '''
        # determine how much goes to train vs val set, such that
        # the total split is proportional to (e-1)/e
        new_train_size = int((self.current_size + len(data)) * (1 - 1/np.exp(1)))
        num_new_train_pairs = new_train_size - len(self.train_data_all)


        new_train_data = data[:num_new_train_pairs]
        new_val_data = data[num_new_train_pairs:]
        
        # Keeping all the samples
        self.val_data_all.extend(new_val_data)
        self.train_data_all.extend(new_train_data)
        self.current_size += len(data)

        # Only recent samples are available for training
        # such that total training data size <= max_size   
        self.train_data = self.train_data_all[-self.train_max_size:]
        self.val_data = self.val_data_all[-self.val_max_size:]


       
    def sample_batch(self, n):
        return random.sample(self.train_data, n)

    def val_iter(self):
        'iterator over validation set'
        return iter(self.val_data)


    @property
    def loss_lb(self):
        '''Train set loss lower bound'''
        even_pref_freq = np.mean([label == 0.5 for (c1, c2, label) in self.train_data])

        #taking into account that label noize is used
        return -((1 - even_pref_freq) * np.log(0.95) + even_pref_freq * np.log(0.5))

    @property
    def val_loss_lb(self):
        '''Validation set loss lower bound'''
        even_pref_freq = np.mean([label == 0.5 for (c1, c2, label) in self.val_data])

        #taking into account that label noize is used
        return -((1 - even_pref_freq) * np.log(0.95) + even_pref_freq * np.log(0.5))

    def get_all_pairs(self):
        '''
        Used to normalize the reward model
        '''
        return self.train_data_all + self.val_data_all

class RewardNet(nn.Module):
    """Here we set up a callable reward model
    Should have batch normalizatoin and dropout on conv layers
    
    """
    def __init__(self, l2 = 0.01, dropout = 0.2, env_type = 'procgen'):
        super().__init__()
        self.env_type = env_type
        if env_type == 'procgen':
            self.model = nn.Sequential(
                #conv1
                nn.Dropout2d(p=dropout),
                nn.Conv2d(3, 16, 3, stride=1),
                nn.MaxPool2d(4, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                #conv2
                nn.Dropout2d(p=dropout),
                nn.Conv2d(16, 16, 3, stride=1),
                nn.MaxPool2d(4, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                #conv3
                nn.Dropout2d(p=dropout),
                nn.Conv2d(16, 16, 3, stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                # 2 layer mlp
                nn.Flatten(),
                nn.Linear(11*11*16, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
            )
        elif env_type == 'atari':
            self.model = nn.Sequential(
                #conv1
                nn.Dropout2d(p=dropout),
                nn.Conv2d(4, 16, 7, stride=3),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                #conv2
                nn.Dropout2d(p=dropout),
                nn.Conv2d(16, 16, 5, stride=2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                #conv3
                nn.Dropout2d(p=dropout),
                nn.Conv2d(16, 16, 3, stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                #conv4
                nn.Dropout2d(p=dropout),
                nn.Conv2d(16, 16, 3, stride=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16, momentum = 0.01),
                # 2 layer mlp
                nn.Flatten(),
                nn.Linear(7*7*16, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1)
            )

        self.mean = 0
        self.std = 0.05
        self.l2 = l2

    def forward(self, clip):
        '''
        predicts the (!) unnormalized sum of rewards for a given clip
        used only for assigning preferences, so normalization is unnecesary
        '''
        # if self.env_type == 'procgen':
        clip = clip.permute(0, 3, 1, 2)

        # normalizing observations to be in [0,1] and adding noize
        clip = clip / 255.0 + clip.new(clip.size()).normal_(0,0.1)

        return torch.sum(self.model(clip))

    def rew_fn(self, x):
        self.eval()
        # if self.env_type == 'procgen':
        x = x.permute(0,3,1,2)

        # we don't add noize during evaluation
        x = x / 255.0

        rewards = torch.squeeze(self.model(x)).detach().cpu().numpy()

        # normalizing output to be 0 mean, 0.05 std over the annotation buffer
        rewards = 0.05 * (rewards - self.mean) / self.std

        return rewards

    def save(self, path):
        torch.save(self.model, path)

    def set_mean_std(self, pairs, device = 'cuda:0'):
        '''
        computes the mean and std over provided pairs data, 
        and sets the relevant properties 
        '''
        rewards = []
        for clip0, clip1 , label in pairs:
            rewards.extend(self.rew_fn(torch.from_numpy(clip0).float().to(device)))
            rewards.extend(self.rew_fn(torch.from_numpy(clip1).float().to(device)))

        unnorm_rewards = self.std * np.array(rewards) / 0.05  + self.mean
        self.mean, self.std = np.mean(unnorm_rewards), np.std(unnorm_rewards)




def rm_loss_func(ret0, ret1, label, device = 'cuda:0'):
    '''custom loss function, to allow for float labels
    unlike in nn.CrossEntropyLoss'''

    #compute log(p1), log(p2) where p_i = exp(ret_i) / (exp(ret_1) + exp(ret_2))
    sm = nn.Softmax(dim = 0)
    preds = sm(torch.stack((ret0, ret1)))
    #getting log of predictions after adding label noize
    log_preds = torch.log(preds * 0.95 + 0.05)

    #compute cross entropy given the label
    target = torch.tensor([1-label, label]).to(device)
    loss = - torch.sum(log_preds * target)

    return loss

@timeitt
def calc_val_loss(reward_model, data_buffer, device):
    '''
    computes average loss over the validation set
    '''

    loss = 0
    num_pairs = 0
    for clip0, clip1 , label in data_buffer.val_iter():

        ret0 = reward_model(torch.from_numpy(clip0).float().to(device))
        ret1 = reward_model(torch.from_numpy(clip1).float().to(device))
        loss += rm_loss_func(ret0, ret1, label, device).item()
        num_pairs += 1

    av_loss = loss / num_pairs

    return av_loss


@timeitt
def train_reward(reward_model, data_buffer, num_samples, batch_size, device = 'cuda:0'):
    '''
    Traines a given reward_model for num_batches from data_buffer
    Returns the new reward_model
    
    Must have:
        Adaptive L2-regularization based on train vs validation loss
        L2-loss on the output
        Output normalized to 0 mean and 0.05 variance across data_buffer
        (Ibarz et al. page 15)
        
    '''
    num_batches = int(num_samples / batch_size)

    reward_model.to(device)
    # current weight decay is stored as the reward_model property
    # and is being adjusted throughout the training
    weight_decay = reward_model.l2
    av_loss = val_loss = 0
    losses = []

    optimizer = optim.Adam(reward_model.parameters(), lr= 0.0003, weight_decay = weight_decay)
    
    print(f'Validation Loss lower bound : {data_buffer.val_loss_lb:6.4f},')

    for batch_i in range(1, num_batches + 1):
        annotations = data_buffer.sample_batch(batch_size)
        loss = 0
        optimizer.zero_grad()
        reward_model.train()
        for clip0, clip1 , label in annotations:
            
            ret0 = reward_model(torch.from_numpy(clip0).float().to(device))
            ret1 = reward_model(torch.from_numpy(clip1).float().to(device))
            loss += rm_loss_func(ret0, ret1, label, device)
        
        loss = loss / batch_size
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch_i % 100 == 0:
            val_loss = calc_val_loss(reward_model, data_buffer, device) 
            av_loss = np.mean(losses[-100:])
            # Adaptive L2 regularization based on the 
            # difference between training and validation losses
            if val_loss > 1.5 * (av_loss):
                for g in optimizer.param_groups: 
                    g['weight_decay'] = g['weight_decay'] * 1.1
                    weight_decay = g['weight_decay']
            elif val_loss < av_loss * 1.1:
                 for g in optimizer.param_groups:
                    g['weight_decay'] = g['weight_decay'] / 1.1   
                    weight_decay = g['weight_decay']

            print(f'batch : {batch_i}, loss : {av_loss:6.4f}, val loss: {val_loss:6.4f},  L2 : {weight_decay:8.6f}')
            
        
    # Storing the weight decay to be used at the next iteration of reward model training
    reward_model.l2 = weight_decay  
    # Adjusting mean and std of the for new version of the reward model 
    reward_model.set_mean_std(data_buffer.get_all_pairs())

    return reward_model, (av_loss, val_loss, weight_decay)

    

@timeitt
def train_policy(policy, num_steps, rl_steps, log_name, callback):
    '''
    Traines policy for num_steps
    Returns retrained policy
    '''
    
    # Implementation of the learning rate decay
    policy.learning_rate = 0.0007*(1 - rl_steps/8e7)
    policy._setup_lr_schedule()

    # reset_num timesteps allows having single TB_log when calling .learn() multiple times
    policy.learn(num_steps, reset_num_timesteps=False, tb_log_name=log_name, callback=callback)

    return policy
   

@timeitt
def collect_annotations(venv, policy, num_pairs, clip_size):
    '''
    Collects episodes using the provided policy, slices them to snippets of given length,
    selects pairs randomly and adds a label based on which snipped had larger reward
    Returns a list of lists [clip0, clip1, label], where label is float in [0,1]
    '''

    n_envs = venv.num_envs

    clip_pool = []
    obs_stack = []
    # we take a noop step in the environment,instead of doing reset(), becase AtariWrapper
    # raises error if you happen to call reset one step before dying

    obs_b, *_ = venv.step(n_envs*[0])

    #collecting 10x as many observations as needed for randomization
    while len(clip_pool) < 10 * num_pairs * 2:
        clip_returns = n_envs * [0]
        for _ in range(clip_size):
            # _states are only useful when using LSTM policies
            action_b , _states = policy.predict(obs_b)
            obs_stack.append(obs_b)

            obs_b, r_b, dones, infos = venv.step(action_b)    
            clip_returns += r_b

        obs_stack = np.array(obs_stack)
        clip_pool.extend([dict(observations = obs_stack[:, i, :], sum_rews = clip_returns[i]) for i in range(n_envs)])

        obs_stack = []

    clip_pairs = np.random.choice(clip_pool, (num_pairs, 2), replace = False)
    data = []
    for clip0, clip1 in clip_pairs:

        if clip0['sum_rews'] > clip1['sum_rews']:
            label = 0.0
        elif clip0['sum_rews'] < clip1['sum_rews']:
            label = 1.0 
        elif clip0['sum_rews'] == clip1['sum_rews']:
            label = 0.5

        data.append([np.array(clip0['observations']), np.array(clip1['observations']), label])

    return data

def main():
    ##setup args
    parser = argparse.ArgumentParser(description='Reward learning from preferences')

    parser.add_argument('--env_type', type=str, default='atari')
    parser.add_argument('--env_name', type=str, default='BeamRider')
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=1)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='LOGS')
    parser.add_argument('--log_name', type=str, default='')

    parser.add_argument('--resume_training', action='store_true')

    parser.add_argument('--init_buffer_size', type=int, default=500)
    parser.add_argument('--init_train_size', type=int, default=10**5, help='number of labels to process during initial training of the reward model')
    parser.add_argument('--clip_size', type=int, default=25, help='number of frames in each clip generated for comparison')
    parser.add_argument('--total_timesteps', type=int, default=5*10**7, help='total number of RL timesteps to be taken')
    parser.add_argument('--n_labels', type=int, default=6800, help="total number of labels to collect throughout the training")
    parser.add_argument('--steps_per_iter', type=int, default=5*10**4, help="number of RL steps taken on each iteration")
    parser.add_argument('--pairs_per_iter', type=int, default=5*10**3, help='number of labels the reward model is trained on each iteration')
    parser.add_argument('--pairs_in_batch', type=int, default=16, help='batch size for reward model training')
    parser.add_argument('--l2', type=float, default=0.0001, help='initial l2 regularization for a reward model')
    parser.add_argument('--dropout', type=float, default=0.2)

    args = parser.parse_args()

    args.ppo_kwargs = dict(verbose=1, n_steps=256, noptepochs=3, nminibatches = 8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'\n Using {device} for training')


    
    run_dir, monitor_dir, video_dir = setup_logging(args)
    global LOG_TIME
    LOG_TIME = os.path.join(run_dir, "TIME_LOG.txt")


    ### Initializing objects ###
    
    # If resuming some earlier training run - load stored objects
    if args.resume_training:
        reward_model, policy, data_buffer, i_num = load_state(run_dir)
        args = load_args(args)    
    
    atari_name = args.env_name + "NoFrameskip-v4"
    venv_fn = lambda: make_atari_continuous(atari_name, n_envs=16)
    annotation_env = make_atari_continuous(atari_name, n_envs=16)  
    annotation_env.reset()
    iter_time = 0

    # In case this is a fresh experiment - initialize fresh objects
    if not args.resume_training:
        policy = A2C('CnnPolicy', venv_fn(), verbose=1, tensorboard_log="TB_LOGS", ent_coef=0.01, learning_rate = 0.0007)
        reward_model = RewardNet(l2= args.l2, dropout = args.dropout, env_type = args.env_type)
        data_buffer = AnnotationBuffer()
        store_args(args, run_dir)  

    #creating the environment with reward replaced by the prediction from reward_model
    reward_model.to(device)
    proxy_reward_function = lambda x: reward_model.rew_fn(torch.from_numpy(x).float().to(device))
    proxy_reward_venv = Vec_reward_wrapper(venv_fn(), proxy_reward_function)

    # resetting the environment to avoid raising error from reset_num_timesteps
    proxy_reward_venv.reset()
    policy.set_env(proxy_reward_venv)


    # eval_env_fn = lambda: make_atari_default(atari_name, n_envs=16, seed = 0, vec_env_cls = SubprocVecEnv)
    # video_env_fn= lambda: make_atari_default(atari_name, vec_env_cls = DummyVecEnv)

    # in case this is a fresh run, collect init_buffer_size samples to AnnotationBuffer
    # and train the reward model on init_train_size number of samples with replacement
    if not args.resume_training:
       
        t_start = time.time()
        print(f'================== Initial iter ====================')

        annotations = collect_annotations(annotation_env, policy, args.init_buffer_size, args.clip_size)
        data_buffer.add(annotations)   

        print(f'Buffer size = {data_buffer.current_size}')
        
        reward_model, rm_train_stats = train_reward(reward_model, data_buffer, args.init_train_size, args.pairs_in_batch)
        # this callback adds values to TensorBoard logs for easier plotting
        callback = TensorboardCallback((data_buffer.current_size, data_buffer.loss_lb, iter_time, rm_train_stats))
        policy = train_policy(policy, args.steps_per_iter, 0,  args.log_name, callback)

        save_state(run_dir, 0, reward_model, policy, data_buffer)

        true_performance = safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])

        t_finish = time.time()
        iter_time = t_finish - t_start
        log_iter(run_dir, args.steps_per_iter, data_buffer, true_performance, 0, rm_train_stats, iter_time)
        
        print(f'Iteration took {time.gmtime(t_finish - t_start).tm_min} min {time.gmtime(t_finish - t_start).tm_sec} sec')
        
        # i_num is the number of training iterations taken      
        i_num = 1 


    num_iters = int(args.total_timesteps / args.steps_per_iter)
    # calculating the initial number of pairs to collect 
    num_pairs = init_num_pairs = round((args.n_labels - args.init_buffer_size) / 0.292 / num_iters) 

    print('init_num_pairs = {}'.format(init_num_pairs))
    for i in range(i_num, num_iters):
        t_start = time.time()
        print(f'================== iter : {i} ====================')

        rl_steps = i * args.steps_per_iter
        # decaying the number of pairs to collect
        num_pairs = round(init_num_pairs / (rl_steps/(args.total_timesteps/10) + 1))

        annotations = collect_annotations(annotation_env, policy, num_pairs, args.clip_size)
        data_buffer.add(annotations)   

        print(f'Buffer size = {data_buffer.current_size}')
        
        reward_model, rm_train_stats = train_reward(reward_model, data_buffer, args.pairs_per_iter, args.pairs_in_batch)

        #TODO : pretify passing data to callback
        callback = TensorboardCallback((data_buffer.current_size, data_buffer.loss_lb, iter_time, rm_train_stats))
        policy = train_policy(policy, args.steps_per_iter, rl_steps, args.log_name, callback)

        # storing the state every 1M steps
        # this assumes that steps_per_iter devides 10**6
        if rl_steps % (10**6) == 0:
            save_state(run_dir, i, reward_model, policy, data_buffer)

        # record_video(policy, video_env_fn(), video_dir, 4000, f"{i}_ITER00_{args.env_name}")
        # true_performance = eval_policy(venv_fn(), policy, n_eval_episodes=50)
        # proxy_performance = eval_policy(test_env, policy, n_eval_episodes=50)

        true_performance = safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer])

        # print(f'True policy preformance = {true_performance}') 
        # print(f'Proxy policy preformance = {proxy_performance}') 

        t_finish = time.time()
        iter_time = t_finish - t_start
        log_iter(run_dir, rl_steps, data_buffer, true_performance, 0 , rm_train_stats, iter_time) 
        
        if LOG_TIME:
            with open(LOG_TIME, 'a') as f:
                f.write(f'Iteration took {time.gmtime(iter_time).tm_min} min {time.gmtime(iter_time).tm_sec} sec\n')
                f.write(f'================== iter : {i+1} ====================\n')
        else:
            print(f'Iteration took {time.gmtime(iter_time).tm_min} min {time.gmtime(iter_time).tm_sec} sec')
     

if __name__ == '__main__':
    main()

