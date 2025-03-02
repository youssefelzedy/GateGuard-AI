# Installation and Usage Guide

## Prerequisites

Ensure you have Python **3.11.7** installed. If not, you can install it using **pyenv**.

## Installing pyenv

To install `pyenv`, run the following commands:

```bash
curl https://pyenv.run | bash
```

Then, add the following lines to your shell configuration file (`~/.bashrc`, `~/.zshrc`, or `~/.profile`):

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

Apply the changes:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Installing a Specific Python Version

To install Python **3.11.7** using `pyenv`, run:

```bash
pyenv install 3.11.7
pyenv global 3.11.7
pyenv rehash
```

Verify the installation:

```bash
python --version
```

## Setup Instructions

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**

   ```bash
   python main.py
   ```

4. **Run the visualization script (if needed):**

   ```bash
   python visualize.py
   ```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named '_bz2'`

If you encounter this error, install the missing dependencies:

```bash
sudo apt update && sudo apt install -y \
    libbz2-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    zlib1g-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev
```

### Reinstalling Python (if needed)

If the issue persists, reinstall Python **3.11.7** using **pyenv**:

```bash
pyenv install 3.11.7 --force
pyenv local 3.11.7
pyenv rehash
```

### Upgrading the virtual environment

To ensure the virtual environment is up to date:

```bash
source myenv/bin/activate
python -m venv myenv --upgrade
```

---

Now, your environment should be properly configured and ready to use! 🚀

