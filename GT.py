import random
import numpy as np
import matplotlib.pyplot as plt


# Parameters

n = 1000                  # Number of nodes
k_max = 10                # Maximum degree per node
p = k_max / float(n)      # Probability of connection
T = 1000                  # Number of time steps
beta = 0.01               # Fermi function parameter (was K)


# Cost and Benefit Range

b = 1.2
c = 0.02
r_values = np.linspace(0, 0.5, 500)


# Payoff Matrix

def get_payoff_matrix(r):
    return np.array([
        [0, 1 + r],
        [-r, 1]
    ])

# Adjacency Matrix

A = np.zeros((n, n))
for i in range(n):
    edges_added = 0
    possible_nodes = list(range(n))
    possible_nodes.remove(i)
    random.shuffle(possible_nodes)

    for j in possible_nodes:
        if edges_added >= k_max:
            break
        if A[i][j] == 0 and random.random() < p:
            A[i][j] = 1
            A[j][i] = 1
            edges_added += 1


# Simulation Function

def simulate_with_fermi_dynamics(S_init, A, payoff_matrix, T=1000, inner_steps=1000, beta=0.01):
    n = len(S_init)
    S = S_init.copy()

    strategy_history = []
    cooperator_density = []
    defector_density = []

    for t in range(1, T + 1):
        num_cooperators = np.sum(S)
        density_C = num_cooperators / n
        density_D = 1 - density_C

        cooperator_density.append(density_C)
        defector_density.append(density_D)

        if t <= 10 or t % 100 == 0:
            print(f"Step {t}/{T}: Cooperator Density = {density_C:.6f}, Defector Density = {density_D:.6f}")

        for _ in range(inner_steps):
            i = random.randint(0, n - 1)
            neighbors_i = [k for k in range(n) if A[i][k] == 1]
            if not neighbors_i:
                continue

            j = random.choice(neighbors_i)

            payoff_i = sum(payoff_matrix[S[i]][S[k]] for k in neighbors_i)
            payoff_j = sum(payoff_matrix[S[j]][S[k]] for k in [k for k in range(n) if A[j][k] == 1])

            try:
                prob = 1.0 / (1.0 + np.exp(-beta * (payoff_j - payoff_i)))
            except OverflowError:
                prob = 0.0 if (payoff_j - payoff_i) < 0 else 1.0

            if random.random() < prob:
                S[i] = S[j]

        strategy_history.append(S.copy())

    return strategy_history, cooperator_density, defector_density


# Main Loop Over r

final_avg_cooperators = []
final_avg_defectors = []

for r in r_values:
    payoff_matrix = get_payoff_matrix(r)
    S_init = np.array([1] * (n // 2) + [0] * (n - n // 2))
    np.random.shuffle(S_init)

    _, coop_density, defect_density = simulate_with_fermi_dynamics(
        S_init, A, payoff_matrix, T=550, inner_steps=1000, beta=beta
    )

    # Averaging over 11 time intervals of length 50 from T=0 to T=550
    coop_avg = [np.mean(coop_density[i:i+50]) for i in range(0, 550, 50)]
    defect_avg = [np.mean(defect_density[i:i+50]) for i in range(0, 550, 50)]

    final_avg_cooperators.append(np.mean(coop_avg))
    final_avg_defectors.append(np.mean(defect_avg))


# Plot

plt.figure(figsize=(10, 6))
plt.plot(r_values, final_avg_cooperators, label='Avg Cooperators', color='green')
plt.plot(r_values, final_avg_defectors, label='Avg Defectors', color='red')
plt.xlabel('Cost-to-benefit ratio (r)')
plt.ylabel('Average Density')
plt.title('Average Densities of Cooperators and Defectors vs. r')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
