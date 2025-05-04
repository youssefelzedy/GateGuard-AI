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

## Cloning Required Repository

Before proceeding, clone the `filterpy` repository:

```bash
git clone https://github.com/rlabbe/filterpy.git
```

## Setup Instructions

### Using Your Machine to RUN the model

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
   uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
   ```

---

### 🐳 Docker Setup

You can easily run this project using Docker. Follow the steps below to build and run the application in a containerized environment.

1. Prerequisites

   * [Docker](https://www.docker.com/get-started) installed on your machine

2. Build the Docker Image

   ```bash
   docker build -t gate-guard-ai .
   ```

   > This will:
   >
   > * Use Python 3.11-slim as a base
   > * Copy your project files into the container
   > * Install required system dependencies (`libgl1`, `libglib2.0-0`)
   > * Install Python dependencies from `requirements.txt`

3. Run the Docker Container

   ```bash
   docker run -p 8000:8000 gate-guard-ai
   ```

   > This starts the FastAPI server using `uvicorn` on port `8000`.

   You can now access your API at: [http://localhost:8000](http://localhost:8000)

4. Auto-Reload (Development Mode)

   To enable live reloading (already enabled via `--reload` in CMD), make sure you **mount your local volume** like this:

   ```bash
   docker run -p 8000:8000 -v ${PWD}:/app gate-guard-ai
   ```

   > Changes made to the code will automatically reflect inside the container.

   ---

   Would you like me to generate a `docker-compose.yml` file for easier multi-container setup or add examples for endpoints in FastAPI?


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

