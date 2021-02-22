
import subprocess
import argparse
from itertools import product

parser = argparse.ArgumentParser(description='Experiments parameters')

parser.add_argument('--env_name', type=str, nargs='+', default=['BeamRider'])
parser.add_argument('--num_seeds', type=int, default=3, help="number of random seed for  each experiment")
parser.add_argument('--pass_args', default="", type=str,
                    help="The specified string in quotes would be passed to the train_reward.py script")

args = parser.parse_args()

n_exps = 0

print('Running experiments')

for (seed, env_name) in product(range(args.num_seeds), args.env_name):

    n_exps += 1

    command = ['python', 'cuda_train.py']

    command.append(f'--env_name {env_name}')

    command.append(args.pass_args)

    command = ' '.join(command)

    print(f'Running:\n{command}')

    subprocess.call(command, shell=True)

print(f'Ran {n_exps} experiments.')
