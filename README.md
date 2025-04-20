# Bayesian Coin Toss Analysis

This project implements a Bayesian analysis of coin toss sequences, providing detailed visualization of how our beliefs about a coin's bias evolve as we observe more tosses. The analysis includes posterior distributions, statistical measures, and their evolution over time.

## Features

- Interactive terminal-based UI with rich formatting
- Support for multiple input formats (H/T, 1/0, comma-separated)
- Visualization of posterior distributions
- Evolution of statistical measures (mean, variance, mode, median)
- Custom prior distributions
- Progress tracking and analysis summary

## Prerequisites

- Python 3.6 or higher
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the program using:

```bash
python main.py
```

### Input Options

The program accepts coin toss sequences in three formats:

1. **H/T string format**:

   - Example: `HTHHTT`
   - H represents Heads, T represents Tails
   - Case insensitive (h or H, t or T)

2. **Comma-separated 1/0 format**:

   - Example: `1,0,1,1,0,0`
   - 1 represents Heads, 0 represents Tails
   - Spaces around commas are optional

3. **Continuous 1/0 string format**:
   - Example: `101100`
   - 1 represents Heads, 0 represents Tails

### Program Flow

1. **Startup**:

   - You can choose between interactive mode or quick analysis

2. **Quick Analysis**:

   - Uses default sequence: `HTHHHHTTTH`
   - Uses default priors:
     - Uniform (1,1)
     - Fair (20,20)
     - Biased (2,8)

3. **Interactive Mode**:
   - Supports all priors from Quick analysis.
   - Optionally enter your own coin toss sequence (minimum 10 tosses)
   - Optionally add custom prior distributions
   - View analysis summary before proceeding
   - See progress during analysis
   - View results and visualizations

## Output Visualizations

The program generates two main types of plots:

1. **Posterior Distribution Plots**:

   - Plots the evolution of the posterior distribution after each toss
   - Includes mean, median, mode, and variance
   - 95% credible interval shown

2. **Statistics Evolution Plots**:
   - Shows how statistical measures change over time
   - Includes mean, variance, mode, and median
   - Color-coded for different priors

## Custom Priors

You can add custom prior distributions by specifying:

- Prior name
- Alpha parameter (α)
- Beta parameter (β)

Example custom prior:

```
Prior name: Strong-Heads
Alpha value: 10
Beta value: 2
```

## Notes

- The sequence must be at least 10 tosses long
- All visualizations are generated using matplotlib
- The program uses Bayesian updating to calculate posterior distributions
- Results are displayed in the terminal and saved as plots in the ./plots directory in the main project directory.

## Author

[@Abhinav Deshpande](https://github.com/Abhinav-gh)
