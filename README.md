# A Low Rank Approach to Minimize Sensor-to-Actuator Communication in Finite Horizon Output Feedback

The code accompaning our ACC 2024 and IEEE L-CSS submission. The code implements the causal factorization algorithm that generates the figures

Authors: Antoine Aspeel, Jakob Nylof, Jing Shuang (Lisa) Li and Necmiye Ozay

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

To run the code only reproducing the figures using the previously saved simulation data in `simulation_results`, run the following command:
~~~~
python plots.py
~~~~

