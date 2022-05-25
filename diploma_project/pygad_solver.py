import pygad

from config import get_config
from environment import *


def fitness(num_cpus, num_vnfds, env_profile, network_service):
    def fitness_func(solution, solution_idx):
        env = Environment(num_cpus, num_vnfds, env_profile, "small_default")
        env.step(len(network_service), network_service, solution)
        reward = env.reward + 1000 * env.constraint_occupancy + 10 * env.constraint_bandwidth + 10 * env.constraint_latency
        env.clear()
        fitness = 1.0 / reward
        return fitness
    return fitness_func


def pygad_main(network_service, length, env):
    config, _ = get_config()
    env.clear()
    env.network_service = network_service

    sol_per_pop = 100
    num_genes = int(length)
    init_range_low = 0
    init_range_high = env.num_cpus - 1
    mutation_percent_genes = 10

    ga_instance = pygad.GA(num_generations=1000,
                           num_parents_mating=40,
                           fitness_func=fitness(env.num_cpus, env.num_vnfds,
                                                config.env_profile, network_service),
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_type=int,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           crossover_type="single_point",
                           crossover_probability=0.1,
                           parent_selection_type="rws",
                           mutation_percent_genes=mutation_percent_genes,
                           )
    ga_instance.run()
    # ga_instance.plot_fitness()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    env.placement = solution
    env.link_used = np.zeros(env.num_cpus)
    env.link_latency = 0
    env._computeLink()
    env._computeReward()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    env = Environment(env.num_cpus, env.num_vnfds, "large_default", "small_default")
    env.step(len(network_service), network_service, solution)
    # env.render()
    penalty = 1000 * env.constraint_occupancy + 10 * env.constraint_bandwidth + 50 * env.constraint_latency
    print("estimate: reward & penalty")
    print(env.reward)
    print(penalty)

    return solution, env.reward, env.constraint_occupancy, env.constraint_bandwidth, env.constraint_latency


if __name__ == '__main__':
    config, _ = get_config()
    env = Environment(config.num_cpus, config.num_vnfd, config.env_profile)
    network_service = [6, 2, 3, 3, 3, 3, 6, 6, 2, 5, 1, 3]
    servivce_length = 12
    # network_service = [6, 5, 3, 2, 6, 1, 5, 6, 4, 1, 3, 8, 7, 4, 7, 3, 7, 8, 3, 8, 2, 8, 5, 8, 2, 8, 1, 8, 3, 8]
    pygad_main(network_service, len(network_service), env)
