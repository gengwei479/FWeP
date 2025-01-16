import subprocess

for aircraft_type in ['f16', '787-8']:
    for algorithm in ['sac']:#, 'ddpg', 'sac', 'td3'
        subprocess.call(["python", "./main_run.py", "--aircraft_type="+aircraft_type, "--algorithm="+algorithm])