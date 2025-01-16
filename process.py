import subprocess

for env_name in ['formation3']:#'navigate', , 'nav_collision' 'nav_change' 'formation3'
    for aircraft_type in ['787-8']:#'787-8' 'f16', 
        for algorithm in ['FWeP']:#, 'ddpg', 'sac', 'td3'
            for sty_llm in ['qwen-turbo']:#qwen-max 'qwen-plus', 'qwen-turbo' 'bailian_v1' , 'dolly_12b_v2'
                subprocess.call(["python", "./main_run.py", "--aircraft_type="+aircraft_type, "--algorithm="+algorithm, "--env_name="+env_name, "--sty_llm="+sty_llm])