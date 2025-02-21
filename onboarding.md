# Quick Guide to Sacred

Sacred is used for experiment configuration and tracking. Here's how to use it in this codebase:

## Config Access in Functions

Use `@ex.capture` to access config parameters:
```python
@ex.capture
def train_model(model_config, train_config):  # These params come from config
    batch_size = train_config["batch_size"]
    # Your code here
```

## Running Experiments with Config Updates

### From Python (e.g. runner.py)
```python
config_updates = {
    "train_config": {
        "epochs": 200,
        "batch_size": 64
    },
    "model_config": {
        "seq_len": 32
    }
}
ex.run(config_updates=config_updates)
```

### From CLI
```bash
# Update nested parameters
python main.py with train_config.epochs=200 model_config.seq_len=32

# Run specific command with updates
python main.py evaluate with model_path="model.pt" train_config.batch_size=64
```

## Adding New Commands
```python
@ex.command
def evaluate(_run, model_path="model.pt"):
    # Log metrics with _run
    _run.log_scalar("accuracy", accuracy)

@ex.automain  # Main command
def main():
    # Main execution
```

## Best Practices
1. Always provide config updates when running experiments
2. Use nested dicts for related params (e.g. `model_config`, `train_config`)
3. Log metrics consistently with `_run.log_scalar()`
