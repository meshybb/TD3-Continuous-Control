import re
import matplotlib.pyplot as plt

log_pendulum = """
Starting Hyperparameter Search for InvertedPendulum-v4
Total combinations: 8

--- Running Experiment 1/8 ---
Parameters: {'lr': 0.001, 'batch_size': 100, 'start_steps': 5000}
Step: 10000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 111.50
Step: 20000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 86.30
Step: 30000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 1000.00
Step: 40000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 1000.00
Step: 50000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 1000.00

--- Running Experiment 2/8 ---
Parameters: {'lr': 0.001, 'batch_size': 100, 'start_steps': 1000}
Step: 10000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 41.60
Step: 20000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 193.30
Step: 30000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 1000.00
Step: 40000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 1000.00
Step: 50000 | Config: {'lr': 0.001, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 1000.00

--- Running Experiment 3/8 ---
Parameters: {'lr': 0.001, 'batch_size': 512, 'start_steps': 5000}
Step: 10000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 3.00
Step: 20000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 3.00
Step: 30000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 30.80
Step: 40000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 61.00
Step: 50000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 1000.00

--- Running Experiment 4/8 ---
Parameters: {'lr': 0.001, 'batch_size': 512, 'start_steps': 1000}
Step: 10000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 88.80
Step: 20000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 96.00
Step: 30000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 1000.00
Step: 40000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 1000.00
Step: 50000 | Config: {'lr': 0.001, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 1000.00

--- Running Experiment 5/8 ---
Parameters: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 5000}
Step: 10000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 66.70
Step: 20000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 69.10
Step: 30000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 105.10
Step: 40000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 1000.00
Step: 50000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 5000} | Avg Reward: 1000.00

--- Running Experiment 6/8 ---
Parameters: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 1000}
Step: 10000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 71.30
Step: 20000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 109.00
Step: 30000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 94.90
Step: 40000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 322.20
Step: 50000 | Config: {'lr': 0.0003, 'batch_size': 100, 'start_steps': 1000} | Avg Reward: 190.00

--- Running Experiment 7/8 ---
Parameters: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 5000}
Step: 10000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 94.60
Step: 20000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 51.10
Step: 30000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 96.10
Step: 40000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 1000.00
Step: 50000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 5000} | Avg Reward: 1000.00

--- Running Experiment 8/8 ---
Parameters: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 1000}
Step: 10000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 75.00
Step: 20000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 72.30
Step: 30000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 1000.00
Step: 40000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 1000.00
Step: 50000 | Config: {'lr': 0.0003, 'batch_size': 512, 'start_steps': 1000} | Avg Reward: 1000.00
"""

# Parse
configs_pend = {}
current_exp = None
for line in log_pendulum.split('\n'):
    m_exp = re.search(r'--- Running Experiment (\d+)/8 ---', line)
    if m_exp:
        current_exp = f"Experiment {m_exp.group(1)}"
        configs_pend[current_exp] = {'steps': [], 'rewards': []}
    m_data = re.search(r'Step: (\d+) \| Config: (.*?) \| Avg Reward: ([\-\d\.]+)', line)
    if m_data and current_exp:
        configs_pend[current_exp]['steps'].append(int(m_data.group(1)))
        configs_pend[current_exp]['rewards'].append(float(m_data.group(3)))
        raw_label = m_data.group(2).replace("'", "").replace("{", "").replace("}", "")
        configs_pend[current_exp]['label'] = raw_label

# 1. Plot all 8 for Inverted Pendulum
plt.figure(figsize=(12, 7))
for exp, data in configs_pend.items():
    plt.plot(data['steps'], data['rewards'], label=data.get('label', exp), linewidth=2, alpha=0.8)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Evaluation Reward', fontsize=12)
plt.title('InvertedPendulum-v4: Hyperparameter Grid Search (50k Steps)', fontsize=14)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('hp_search_pendulum_8_configs.png', dpi=150)
plt.close()

# 2. Plot specific comparison for Inverted Pendulum
# We'll isolate:
# Exp 1 (Baseline: lr=1e-3, bs=100, ss=5000)
# Exp 8 (Our Config: lr=3e-4, bs=512, ss=1000)
# Exp 2 (Fast Start: lr=1e-3, bs=100, ss=1000)
plt.figure(figsize=(10, 6))

plt.plot(configs_pend['Experiment 1']['steps'], configs_pend['Experiment 1']['rewards'], 
         label='Baseline Paper (lr=1e-3, bs=100, ss=5000)', linewidth=2.5, color='#d62728', alpha=0.9)

plt.plot(configs_pend['Experiment 8']['steps'], configs_pend['Experiment 8']['rewards'], 
         label='Our Config (lr=3e-4, bs=512, ss=1000)', linewidth=2.5, color='#1f77b4', alpha=0.9, linestyle='--')

plt.plot(configs_pend['Experiment 2']['steps'], configs_pend['Experiment 2']['rewards'], 
         label='Hybrid (lr=1e-3, bs=100, ss=1000)', linewidth=2.5, color='#2ca02c', alpha=0.8)

plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Evaluation Reward', fontsize=12)
plt.title('InvertedPendulum-v4: Convergence Comparison', fontsize=14)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('hp_search_pendulum_comparison.png', dpi=150)
plt.close()