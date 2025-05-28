# MSA, SRA, MSAD, and SRAD Algorithms

This repository contains a Python implementation of a transit assignment model using four different algorithms for solving the passenger flow distribution problem in a congested transit network.

## 📌 Description

The following four algorithms are implemented in this module:

- **MSA (Method of Successive Averages)**
- **SRA (Self-Regulated Averaging)**
- **MSAD (MSA with Detection)** — MSA enhanced with autocorrelation-based acceleration
- **SRAD (SRA with Detection)** — SRA enhanced with autocorrelation-based acceleration

## 🧠 Core Components

### 1. `NETWORK` Class

Defines the transit network structure, including:

- Zones and lines
- Itinerary and scheduled frequency
- OD demand matrix
- Effective frequency and capacity parameters

### 2. `PLM` Class (Pre-Line Marked Model)

Handles:

- Initialization of decision proportions
- Passenger flow computation
- Effective frequency calculation
- Travel time and optimal line set (ALS) estimation
- Proportion updates (for both MSA and SRA)

### 3. `MSA` Class

Controls the iterative solving procedure:

- Runs MSA and SRA algorithms
- Performs auxiliary proportion calculation
- Detects convergence
- Integrates acceleration based on **autocorrelation detection** (for MSAD and SRAD)

## 🔧 How to Use

### Run the Script

A small-scale sample network is already embedded in the module. You can run the script directly to get the assignment result using the SRA algorithm by default.
Example usage:

```python
net = NETWORK(None, 1)  # The first argument is the network input; None uses the built-in toy network. The second argument specifies the effective frequency function parameter (alpha).
plm = PLM()
msa = MSA(0.000001, 100)  # Initialize the network, model, and algorithm (with tolerance and max iterations)
msa.Self_regulated_averaging_method_Algorithm(net, plm, 1, [], [])  # Run the SRA algorithm; `1` enables autocorrelation-based acceleration
# msa.Method_of_Successive_Algorithm(net, plm, 1, [], [])  # Uncomment to run the MSA algorithm instead
# print(plm.v)  # Uncomment to print the resulting passenger flow distribution
```
