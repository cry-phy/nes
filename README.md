# NES

The Theroretical background:

[Handout](res/Handout.md)

## Usage

 Look at [Examples](res/example.ipynb)!

This project uses poetry for managing dependencies for Python 3.10. To use use the environment run
```bash
poetry install --no-root
poetry env activate
```

To install pre-commit hooks use
```bash
poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
```
