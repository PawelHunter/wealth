import simulation as sim
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # an example
    # setup parameters of simulation
    wealth_dist = sim.Simulation(steps=1000,no_agents=1000, sigma=0.02, mu=0.0, J=-0.3, alpha=0.00, beta=0.001, g=1.0)
    wealth_dist.run_nonlinear() # run simulation
    agents, no = wealth_dist.get_simulation() # get results

    # plot wealth distribution in y- log scale
    target_plot_file_path = "figure_1.png"
    to_plot = np.sort(agents)
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    plt.yscale("log")
    ax.plot(to_plot,".g")
    fig.savefig('target_plot_file_path')  # save the figure to file
    plt.close(fig)
