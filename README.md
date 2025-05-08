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
mamba activate dqa
```

### 2. Install casadata using pip

Install the `casadata` pacakge using pip:

```bash
pip install casadata
```

### 3. Clone & Install the Repository

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

### 4. Install and Register the Jupyter Kernel

Install packages necessary for running Jupyter under this mamba environment and register this environment so that it can be selected to run Jupyter notebooks. The library uses `bokeh` to generate interactive plots:

```bash
mamba install ipython ipykernel ipywidgets notebook jupyterlab
python -m ipykernel install --user --name dqa --display-name "dqa"
```

This creates a kernel spec under `~/.local/share/jupyter/kernels/dqa/` pointing to your `dqa` environment’s Python.

## Usage
In the Python interpreter, a Jupyter notebook, or in your own scripts, the module can be imported in the standard way. For instance, to access
the RFI routines, do the following:

```python
from dqatool import rfi
rfi.detect_rfi_1d("msname.ms", overwrite=False, flagfile="rfiflags.txt")
```

The example notebook located at *examples/dqa.ipynb* contains further information on how the various submodules can be accessed and used.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
