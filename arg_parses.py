import argparse

def create_parses():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--task_name', default='training_kinematic', type=str)#training_kinematic trajectory_plan compared_algorithm
    parser.add_argument('--algorithm', default='kinematic_finetuning', type=str)#FWeP, ppo, ddpg, sac, td3, kinematic_finetuning
    parser.add_argument('--evaluate_times', default=3, type=int)
    ##-----------------------------sim env-------------------------------------
    parser.add_argument('--env_name', default='formation3', type=str)# navigate nav_change nav_collision formation3
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

    #---------------------------trajectory planner---------------------------------------
    parser.add_argument('--sty_llm', default='qwen-max', type=str)#qwen-plus qwen-turbo qwen-max dolly_12b_v2 bailian_v1
    parser.add_argument('--qwen_api_key', default='sk-1c08b502bccb43ad959e9e86612ab995', type=str)
    parser.add_argument('--prompt_epoch', default=50, type=int)
    parser.add_argument('--sear_version', default='v5000', type=str)
    parser.add_argument('--pretrained_checkpoint', default='pretrained_model', type=str)
    parser.add_argument('--finetuned_checkpoint', default='finetuned_model', type=str)
    parser.add_argument('--master_data_dir', default='master_data', type=str)
    parser.add_argument('--result_data_dir', default='result', type=str)
    parser.add_argument('--max_seq_length', default=200, type=int)

    #---------------------------kinematic translater-------------------------------------
    parser.add_argument('--kine_llm', default='gpt2', type=str)#transformer gpt2 qwen2.5_1.5b t5_60m t5_220m t5_3b
    parser.add_argument('--batch_size', default=10, type=int)#32
    parser.add_argument('--max_dataset_size', default=1024, type=int)
    parser.add_argument('--num_epochs', default=1500, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=int)
    parser.add_argument('--token_version', default=1, type=int)
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--fine_tune_mode', default='p-tuning_mlp', type=str)#full p-tuning_mlp p-tuning_lstm prefix-tuning prompt-tuning lora adalora ia3

    #------------------------------PPO------------------------------------------
    parser.add_argument("--max_train_steps", type=int, default=int(3e4), help=" Maximum number of training steps")#3e6
    parser.add_argument("--PPO_evaluate_freq", type=float, default=10, help="Evaluate the policy every 'evaluate_freq' steps")#5e3
    parser.add_argument("--PPO_save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--PPO_policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--PPO_batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--PPO_mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--PPO_hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--PPO_lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--PPO_lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--PPO_gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--PPO_lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--PPO_epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--PPO_K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--PPO_use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--PPO_use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--PPO_use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--PPO_use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--PPO_entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--PPO_use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--PPO_use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--PPO_use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--PPO_set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--PPO_use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    
    #------------------------------DDPG-----------------------------------------
    # parser.add_argument("--DDPG_max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--DDPG_random_steps", type=int, default=int(100), help="Take the random actions in the beginning for the better exploration")
    parser.add_argument("--DDPG_update_freq", type=int, default=int(50), help="Take 50 steps,then update the networks 50 times")
    parser.add_argument("--DDPG_evaluate_freq", type=int, default=10, help="Evaluate the policy every 'evaluate_freq' steps")
    
    #------------------------------SAC------------------------------------------
    # parser.add_argument("--SAC_max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--SAC_random_steps", type=int, default=int(100), help="Take the random actions in the beginning for the better exploration")
    parser.add_argument("--SAC_evaluate_freq", type=int, default=int(10), help="Evaluate the policy every 'evaluate_freq' steps")
    
    #------------------------------TD3------------------------------------------
    # parser.add_argument("--TD3_max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--TD3_random_steps", type=int, default=int(100), help="Take the random actions in the beginning for the better exploration")
    parser.add_argument("--TD3_evaluate_freq", type=int, default=int(10), help="Evaluate the policy every 'evaluate_freq' steps")

    return parser.parse_args()

def pre_parses(args):
    args.aircraft = [args.aircraft_type]
    return args