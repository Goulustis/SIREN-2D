import numpy as np
import rich
from tqdm import tqdm
import torch

CONSOLE = rich.get_console()

def parallel_tempering_opt(sampler, 
                            energy_fn, 
                            init_fn, 
                            num_chains=10, 
                            num_iterations=5000, 
                            swap_interval=100):
    # Initialize inverse temperatures (betas)
    betas = np.linspace(1.0, 0.1, num_chains)

    # Initialize chains
    chains = init_fn(num_chains=num_chains)
    energies = [energy_fn(chain) for chain in chains]
    # print("initial energy: ", max(energies))
    CONSOLE.print(f"Initial energy: {max(energies)}", style="yellow")

    # Keep track of the best solution
    best_solution = chains[0]
    best_energy = energies[0]

    # Initialize acceptance counters
    metropolis_accepts = [0 for _ in range(num_chains)]  # Accepted Metropolis updates per chain
    total_metropolis_updates = [0 for _ in range(num_chains)]  # Total Metropolis updates per chain

    swap_attempts = 0
    swap_accepts = 0

    for i in tqdm(range(num_iterations)):
        # Update each chain
        for j in range(num_chains):
            total_metropolis_updates[j] += 1  # Increment total updates
            accept_counter = [0]  # Create a counter for this update
            chains[j] = sampler(chains[j], betas[j], energy_fn, accept_counter)
            if accept_counter[0] > 0:
                metropolis_accepts[j] += 1  # Increment accepted updates
            energies[j] = energy_fn(chains[j])

            # Update the best solution found so far
            if energies[j] > best_energy:
                # best_solution = chains[j].copy()
                best_solution = chains[j].clone()
                best_energy = energies[j]

        # Attempt to swap states between adjacent chains
        if i % swap_interval == 0:
            print("BEST ENERGY: ", best_energy)
            for j in range(num_chains - 1):
                swap_attempts += 1  # Increment swap attempts
                # Compute log acceptance probability for swapping
                delta = (betas[j] - betas[j + 1]) * (energies[j + 1] - energies[j])
                log_swap_prob = delta

                # Swap states if accepted
                if np.log(np.random.rand()) < log_swap_prob:
                    swap_accepts += 1  # Increment swap accepts
                    chains[j], chains[j + 1] = chains[j + 1], chains[j]
                    energies[j], energies[j + 1] = energies[j + 1], energies[j]

    return best_solution, best_energy


class PTOpt:
    def __init__(self, model_def, 
                       init_fn, 
                       obj_fn,
                       num_chains = 10, 
                       num_iterations = 5000, 
                       swap_interval=100, 
                       learning_rate=1e-2):
        self.model_def = model_def
        self.init_fn = init_fn
        self.obj_fn = obj_fn
        self.num_chains = num_chains
        self.num_iterations = num_iterations
        self.swap_interval = swap_interval
        self.learning_rate = learning_rate
    

    def metropolis_update(self, current:np.ndarray, 
                                beta:float, energy_fn, accept_counter):
        """
        Perform one Metropolis update at a given inverse temperature beta,
        ensuring proposals stay within [-1, 1] via reflecting boundaries.
        
        Parameters:
            current (np.ndarray): Current state.
            beta (float): Inverse temperature.
            energy_fn (callable): Energy function.
            accept_counter (list): A list with a single integer to count accepted proposals.
        
        Returns:
            np.ndarray: The next state (accepted or current).
        """
        # Propose a new state
        # proposal = current + np.random.normal(0, self.learning_rate, size=current.shape)
        proposal = current + torch.randn_like(torch.tensor(current)) * self.learning_rate

        # Compute energies
        E_current = energy_fn(current)
        E_proposal = energy_fn(proposal)

        # Compute log acceptance probability
        delta_E = E_proposal - E_current
        log_accept_prob = -beta * delta_E

        # Decide whether to accept the proposal
        if np.log(np.random.rand()) < log_accept_prob:
            accept_counter[0] += 1  # Increment accepted count
            return proposal
        else:
            return current
    
    @torch.no_grad()
    def optimize(self):
        return parallel_tempering_opt(
            sampler = self.metropolis_update, 
            energy_fn=self.obj_fn,
            init_fn=self.init_fn,
            num_chains=self.num_chains,
            num_iterations=self.num_iterations,
            swap_interval=self.swap_interval
        )