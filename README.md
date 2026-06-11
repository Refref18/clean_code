## Usage


the data file and the trained model is already saved to bserve oddl you can directly start from learn_rules script. 

### 1. Data Collection


```bash

python exp.py
```

### 2. Training

Train the model:

```bash
python train.py -c configs/train.yaml
```

### 3. Rule Learning

Learn symbolic operators and generate PDDL domain:

```bash
python learn_rules.py -n <model_name> -n_eps 500 
```

### 4. Evaluation

```bash

python plan_height.py -c configs/eval.yaml # tallest - shortest

python plan_task_achievable.py -c configs/eval_task.yaml # inside - occlude

```

# clean_code
