import numpy as np
import copy


def standard_simulation_step(agents: np.array([np.double]),
                             J: np.double,
                             alpha: np.double,
                             beta: np.double
                             ) -> np.array(np.double):
    tmp_agents = copy.deepcopy(agents)
    total: np.double = 0.0
    total_tax: np.double = 0.0
    n = agents.shape[0]
    J = J / (n - 1)
    w_delta: np.double = 0.0

    # interaction loop
    for i in range(n):
        for j in range(i + 1, n):
            if J * (agents[j] - agents[i]) >= 0.0:
                w_delta = J * (agents[j] - agents[i])
                # increase wealth of agent i
                tmp_agents[i] = tmp_agents[i] + (1. - alpha) * w_delta
                # transaction tax aggregation
                total_tax += alpha * w_delta
                # decrease wealth of agent j
                tmp_agents[j] = tmp_agents[j] - w_delta

            else:
                w_delta = J * (agents[i] - agents[j])
                # decrease wealth of agent i
                tmp_agents[i] = tmp_agents[i] - w_delta
                # transaction tax aggregation
                total_tax += alpha * w_delta
                # increase wealth of agent j
                tmp_agents[j] = tmp_agents[j] + (1. - alpha) * w_delta

    # collect tax from agent i
    tmp_agents = (1. - beta) * tmp_agents
    # wealth tax aggregation
    total_tax += beta * np.sum(tmp_agents)
    # calculate total wealth in system to normalization
    total = (np.sum(tmp_agents) + total_tax)/n
    # tax redistribution and wealth normalization
    tmp_agents = (tmp_agents + (total_tax / n)) / total
    return tmp_agents


def transaction(w_j: np.double ,w_delta: np.double,w_i: np.double) -> np.double:
    w_max = w_delta if w_delta >= -w_i else -w_i
    w_min = w_max if w_max <= w_j else w_j
    return w_min


def non_linear_transfer_simulation_step(agents: np.array([np.double]),
                                        J: np.double,
                                        alpha: np.double,
                                        beta: np.double,
                                        g: np.double
                                        ) -> np.array(np.double):
    tmp_agents = copy.deepcopy(agents)
    total: np.double = 0.0
    total_tax: np.double = 0.0
    n = agents.shape[0]
    J = J / (n - 1)
    w_delta: np.double = 0.0

    # interaction loop
    for i in range(n):
        for j in range(i + 1, n):
            if J * (agents[j] - agents[i]) >= 0.0:
                w_delta = min( J * (agents[j] - agents[i]), agents[j] / (g * n) )
                # increase wealth of agent i
                tmp_agents[i] = tmp_agents[i] + (1. - alpha) * w_delta
                # transaction tax aggregation
                total_tax += alpha * w_delta
                # decrease wealth of agent j
                tmp_agents[j] = tmp_agents[j] - w_delta

            else:
                w_delta = min( J * (agents[i] - agents[j]), agents[i] / (g * n))
                # decrease wealth of agent i
                tmp_agents[i] = tmp_agents[i] - w_delta
                # transaction tax aggregation
                total_tax += alpha * w_delta
                # increase wealth of agent j
                tmp_agents[j] = tmp_agents[j] + (1. - alpha) * w_delta


    # wealth tax aggregation
    total_tax += beta * np.sum(tmp_agents)
    # collect tax from agent i
    tmp_agents = (1. - beta) * tmp_agents
    # calculate total wealth in system to normalization
    total = (np.sum(tmp_agents) + total_tax)/n
    # tax redistribution and wealth normalization
    tmp_agents = (tmp_agents + (total_tax / n)) / total
    return tmp_agents


if __name__ == "__main__":
    agent_matrix = np.array([1.0, 1.0, 1.0, 1.0], dtype='d')
    print(agent_matrix)
    agent_matrix_1iter = standard_simulation_step(agent_matrix, -3.0, 0.0, 0.0)
    print(agent_matrix_1iter)  # expected [0.55 0.15 0.15 0.15]
    agent_matrix_1iter = standard_simulation_step(agent_matrix_1iter, -1.0, 0.0, 0.0)
    print(agent_matrix_1iter)

    agent_matrix = np.array([1.0, 2.0, 0.0, 1.0], dtype='d')
    agent_matrix_1iter = non_linear_transfer_simulation_step(agent_matrix, -100.0, 0.0, 0.0,1.0)
    print(agent_matrix_1iter) # expected [0.75,2.5,0.0,0.75]
