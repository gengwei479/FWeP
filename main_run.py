from otheralg.ppo.PPO_continuous_main import ppo_main
from otheralg.ppo.mappo_continuous_main import mappo_main
from otheralg.ddpg.DDPG import ddpg_main
from otheralg.ddpg.MADDPG import maddpg_main
from otheralg.sac.SAC_continuous import sac_main
from otheralg.sac.MASAC_continuous import masac_main
from otheralg.td3.TD3 import td3_main
from otheralg.td3.MATD3 import matd3_main
from trans_trainer.epolit_main import epolit_main


from arg_parses import create_parses, pre_parses
from envs.env_wrapper import NavigateEnv, NavigateChangeEnv, AvoidCollisionEnv
from envs.multienv_wrapper import MultiAgentFormation

if __name__ == "__main__":
    args = create_parses()
    args = pre_parses(args)

    if args.env_name == "navigate":
        env = NavigateEnv(args)
        env_evaluate = NavigateEnv(args)
    elif args.env_name == "nav_change":
        env = NavigateEnv(args)
        env_evaluate = NavigateChangeEnv(args)
    elif args.env_name == "nav_collision":
        env = AvoidCollisionEnv(args)
        env_evaluate = AvoidCollisionEnv(args)
    elif args.env_name == "formation3":
        env = MultiAgentFormation(args)
        env_evaluate = MultiAgentFormation(args)
    
    

    assert env is not None
    assert env_evaluate is not None

    if args.algorithm == "FWeP":
        epolit_main(args, env_evaluate)
    elif args.algorithm == "kinematic_finetuning":
        if args.kine_llm == "transformer":
            from trans_trainer.trans_base_eval import TRANSBASE
            trans_kine = TRANSBASE(args)
            trans_kine.tester_fast()
        elif "t5" in args.kine_llm:
            from trans_trainer.trans_t5_eval import TRANST5
            trans_kine = TRANST5(args)
            trans_kine.trainer()
        elif args.kine_llm == "gpt2":
            if args.fine_tune_mode == "full":
                from trans_trainer.trans_gpt2_eval import TRANSGPT
                trans_kine = TRANSGPT(args)
                trans_kine.trainer()
            else:
                from trans_trainer.trans_gpt2_peft import TRANSGPT
                trans_kine = TRANSGPT(args)
                # trans_kine.tester()
                trans_kine.tester_fast()
        elif args.kine_llm == "qwen2.5_1.5b":
            from trans_trainer.trans_qwen2_eval import TRANSQWEN2
            trans_kine = TRANSQWEN2(args)
            trans_kine.trainer()

    elif args.algorithm == "ppo":
        if args.env_name == "formation3":
            mappo_main(args, env, env_evaluate)
        else:
            ppo_main(args, env, env_evaluate)
    elif args.algorithm == "ddpg":
        if args.env_name == "formation3":
            maddpg_main(args, env, env_evaluate)
        else:
            ddpg_main(args, env, env_evaluate)
    elif args.algorithm == "sac":
        if args.env_name == "formation3":
            masac_main(args, env, env_evaluate)
        else:
            sac_main(args, env, env_evaluate)
    elif args.algorithm == "td3":
        if args.env_name == "formation3":
            matd3_main(args, env, env_evaluate)
        else:
            td3_main(args, env, env_evaluate)