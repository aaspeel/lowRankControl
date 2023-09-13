# A Low Rank Approach to Minimize Sensor-to-Actuator Communication in Finite Horizon Output Feedback

The code accompaning our joint ACC 2024 and IEEE L-CSS paper submission.

**Authors:** Antoine Aspeel, Jakob Nylof, Jing Shuang (Lisa) Li and Necmiye Ozay

The code reproduces the results in the section "Numerical Demonstrations" of the paper. In particular, the code:
1. Solves the nuclear norm minimization problem and computes a causal factorization of the resulting optimal controller.
2. Solves the actuator and sensor norm minimization problems.
3. Plots the figures in the paper and prints out runtimes of steps 1 and 2.

## Setup
From the base directory of this repository, install dependencies with:
~~~~
pip install -r requirements.txt
~~~~

## Run
To run the code solving the optimization problems for the nuclear norm, sensor norm and actuator norm cases and reproducing the results and figures in section "Numerical Demonstrations", run the following command:
~~~~
python simulation.py
~~~~

The figures and the file `simulationT20.pickle` containing the simulation data is saved in `simulation_results`.

To run the code only reproducing the figures using the previously saved simulation data in `simulation_results/simulationT20.pickle`, run the following command:
~~~~
python plots.py
~~~~

