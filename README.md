# DQATool

## Description
This repository contains tools for data quality assessment of radio interferometric data sets.

## Features
- RFI detection, flagging, and logging
- Quickview Imaging
- Plotting

## Installation

### Prerequisites

- **Mamba** (or Conda)  
- **Git**  
- **Poetry** (optional, for dependency management)

---

### 1. Create & Activate the Environment

Create a new environment named `dqa` (or choose your own name) and activate it:

```bash
mamba create --name dqa python=3.10 -y
conda activate dqa
```

### 2. Install Core Dependencies

Install the CASA libraries plus Jupyter tools in one go:

```bash
mamba install casacore python-casacore casatasks casatools ipython ipykernel ipywidgets notebook jupyterlab -y
```

### 3. Register the Jupyter Kernel

Register this environment so you can select it in Jupyter notebooks:

```bash
python -m ipykernel install --user --name dqa --display-name "dqa"
```

This creates a kernel spec under `~/.local/share/jupyter/kernels/dqa/` pointing at your `dqa` environment’s Python.

### 4. Clone & Install the Repository

```bash
git clone https://github.com/iniyannatarajan/dqatool.git
cd dqatool
```

If you use Poetry:

```bash
poetry install
```

Alternatively, install in “editable” mode with pip:

```bash
pip install -e .
```

### 5. Verify Installation

Launch JupyterLab:

```bash
jupyter lab
```

In the launcher, choose the “dqa” kernel.

Open one of the example notebooks (e.g. `examples/demo.ipynb`) and confirm everything runs.

## Usage
```bash
# Example usage
<usage-command>
```

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
