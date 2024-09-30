
## Parameters

`finetune.py` and `eval.py` has three parameters.

1. `config_path` is the finetuning/predicting configuration. We provide demo cases in this [directory](./config/finetune)
2. `project` is the project name (for WANDB). Optional.
3. `name` is the name of this finetuning/predicting (for WANDB). Optional.


`finetune-select.py` has one new parameters.

1. `select_path` is the path of selected coreset. We provide all coresets selected by Staff and baselines in our experiments in this [path](../selection/selected_data)

