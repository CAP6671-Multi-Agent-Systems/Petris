# Petris

Tetris clone with AI agents to play the game.

# Requirements

-   Python3.8 >=

## Set Python Virtual Environment

Create `venv` Directory:

```bash
python3 -m venv ./venv
```

Activate virtual environment:

```bash
source venv/bin/activate
```

Upgrade pip to 23.0.1 by running the command:

```bash
pip install --upgrade pip
```

To exit the virtual environment:

```bash
deactivate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

# Startup

```bash
python3 src/petris.py
```

## Optional Arguments

`-s/--speed`: Sets the speed of the clock. Higher the faster the piece falls down.

`-p/--parameters`: Sets the parameter file to use. Depending on which file you use, activates a different agent.

`-r/--random-iterations`: The amount of random iterations you want the Bayesian Optimization to take. More can help diversify the exploration space.

`-i/--iterations`: The amount of iterations you want the Bayesian Optimization to take. More steps lead to better maximums.

## Example Arguments

```bash
python3 src/petris.py -p 1000 -p input_reinforce.json -r 25 -i 35
```
Note: Depending on your harware, this could take 12+ hours to run.

# Acknowledgments

-   https://docs.python.org/3/library/venv.html
-   https://www.tensorflow.org/agents
