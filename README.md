# 🚌 Pre-line Marked Transit Assignment Model

This project implements a transit assignment model tailored for frequency-based public transportation networks. It addresses the **oscillation issue** commonly observed in iterative solutions to the *Common Line Problem* (CLP), where passengers select transit lines based on frequency rather than schedules.

The model in this repository is based on the **Pre-Line Marked (PLM) transit assignment** approach, which differentiates passengers based on the transit line they previously used before arriving at a transfer node. This refined modeling framework prevents unrealistic re-boarding behavior and enables more accurate modeling of transfer decisions.

To enhance convergence, two improved heuristic algorithms are implemented:
- **MSAD** – Method of Successive Averages with Detection
- **SRAD** – Self-Regulated Averaging with Detection

These are based on classical **MSA** and **SRA** algorithms, respectively, and are equipped with an **Oscillation Detection Algorithm (ODA)** that dynamically identifies and mitigates periodic oscillation during the iteration process.

---

## 📁 Project Structure

This repository includes implementations of both the **PLM model** and a **strategy-based model** for comparison. It also provides implementations of four iterative algorithms: `MSA`, `SRA`, and their improved versions `MSAD` and `SRAD`.

> 📌 **Usage**: Please refer to the document within each model's subdirectory for detailed instructions on usage, data format, and configuration

```bash
PLM_Transit_Assignment
├── PLM/                       # Pre-Line Marked model implementation
│   ├── Source_Code            # Source code files for the PLM model
│   ├── Toy_Network            # a toy network example for testing
│   ├── Sensitivity_Analysis   # Sensitivity analysis scripts for ODA
│   ├── Winnipeg_Network       # Example network for Winnipeg
│   └── README.md              # Detailed usage guide for solving PLM model

├── StrategyModel/             # Strategy-based model
│   ├── LNS.py                 # A transit assignment model based on line and node strategies
│   └── README.md              # Detailed usage guide

└── README.md                  # Main documentation
```
## 🧰 Environment Requirements

- Python **3.6 or higher** is required.
- Tested under **Python 3.9**.
- Please make sure all libraries used in `PLM/Source_Code/MSA_with_detection.py` are installed:
