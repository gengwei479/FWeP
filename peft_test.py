import subprocess

for fine_tune_mode in ['p-tuning_mlp', 'p-tuning_lstm', 'prefix-tuning', 'prompt-tuning', 'lora', 'adalora', 'ia3']:
    subprocess.call(["python", "./main_run.py", "--kine_llm=gpt2", "--fine_tune_mode="+fine_tune_mode])