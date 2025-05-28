# Experiments for PLM

This directory contains all the experiment codes and data designed for the PLM model and the SRAD algorithm.

---

## 📁 Directory Structure

```
PLM
├── README.md                        # Project description (this file)
├── Sensitivity_Analysis/           # Sensitivity tests for ODA parameters
│   └── test2_opt_parame.py         # Code for sensitivity analysis
├── Source_Code/                    # Core algorithm implementation
│   ├── MSA_with_detection.py       # Main file implementing MSA/SRA/MSAD/SRAD
│   ├── init.py
│   └── readme.md                   # Module usage
├── Toy_Network/                    # Experiments on a small test network
│   └── MSA_SRA_with_ODA_small_net.py  # Compares SRA and SRAD under different efff parameters
└── Winnipeg_Network/               # Large-scale case study based on Winnipeg network
    ├── examples/                   # JSON inputs under different demand levels
    │   ├── net_high.json
    │   ├── net_low.json
    │   └── net_medium.json
    ├── s2s_report.txt              # Link data of the Winnipeg transit lines
    ├── test4_large_scale.py        # Code for testing on the Winnipeg network
    └── transit_lines_1000.txt      # Full set of transit lines data
```

---

## ⚙️ Features

- Implements 4 transit assignment algorithms:
  - **MSA (Method of Successive Averages)**
  - **SRA (Self-Regulated Averaging)**
  - **MSAD (MSA + autocorrelation detection)**
  - **SRAD (SRA + autocorrelation detection)**
- Models passenger behavior under frequency-based assignment
- Considers vehicle capacity and saturation effects
- Visualizes convergence using log-scaled gap plots

---

## 🧪 Details of Experiments

### 🔬 `Toy_Network/`

Contains experiments conducted on a small-scale synthetic network. This script:

- Compares **SRA** and **SRAD** algorithms
- Tests under varying values of **effective frequency function parameters (efff)**
- Helps evaluate convergence efficiency and accuracy in a controlled environment

### 📊 `Sensitivity_Analysis/`

Analyzes the sensitivity of the **ODA (One-Decision Assignment)**-based algorithm to its core parameters.

- Three key parameters are varied systematically (`MSA.queue_length`, `MSA.variance_threshold`, `MSA.coefficient_requirement`)
- Allows evaluation of parameter influence on model performance and stability
- Results help guide parameter calibration in practice

### 🏙 `Winnipeg_Network/`

Applies the PLM model to a real large-scale network (Winnipeg transit data):

- Includes full bus line definitions (`transit_lines_1000.txt`)
- Provides pre-prepared network instances with different demand levels:
  - `net_low.json`, `net_medium.json`, `net_high.json`

---

## 🚀 How to Run

To run toy network experiments:

```bash
cd ../Toy_Network
python MSA_SRA_with_ODA_small_net.py
```

To run sensitivity analysis:

```bash
cd ../Sensitivity_Analysis
python test2_opt_parame.py
```

To test on the Winnipeg network:

```bash
cd ../Winnipeg_Network
python test4_large_scale.py
```

Make sure the paths are correctly set, and modify the output file paths if needed.
If you're using your own script, please refer to the `readme.md` file in the `Source_Code/` directory for module structure and usage, and make sure to import the correct module as follows:

```python
import sys
sys.path.append('your_path/PLM_Transit_Assignment_model')
from PLM.Source_Code.MSA_with_detection import PLM, MSA, NETWORK

# experiment code here
```
