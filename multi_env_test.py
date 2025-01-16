import argparse
from envs.multienv_wrapper import MultiAgentFormation

def create_parses():
    parser = argparse.ArgumentParser()
   ##-----------------------------sim env-------------------------------------
    parser.add_argument('--env_name', default='nav_change', type=str)# navigate nav_change nav_collision
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--obs_dim', default=6, type=int)
    parser.add_argument('--action_dim', default=4, type=int)
    parser.add_argument('--static_action_dim', default=3, type=int)
    parser.add_argument('--aircraft_type', default='f16', type=str)# f16 787-8
    parser.add_argument('--aircraft', default=[], type=list)
    parser.add_argument('--num_agents', default=1, type=int)
    parser.add_argument('--num_opponents', default=0, type=int)
    parser.add_argument('--agent_keys', default=['player1'], type=list)
    parser.add_argument('--opponent_keys', default=[], type=list)
    parser.add_argument('--altitude_steps', default=[0, 10000, 90000], type=list)# feet unit
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()

def pre_parses(args):
    args.aircraft = [args.aircraft_type]
    return args

args = create_parses()
args = pre_parses(args)
env = MultiAgentFormation(args)
obs = env.reset()
# print(obs)

for _ in range(50):
    actions = {}
    for agent_id, agent_name in enumerate(env.agent_keys):
        actions[agent_name] = dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], [0, 0, 0, 1]))
    cur_obs, reward, done, _ = env.step(actions)
    print(cur_obs)