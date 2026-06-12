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

python plan_task.py -c configs/eval_task.yaml -w 8 # inside - occlude

```

# clean_code
python extend_towers_from_height4.py   --in_csv data.csv   --out_dir extended_height_csv   --start_height 4   --max_height 8   --workers 8


python extend_towers_pool_based.py \
  --out_dir pool_based_debug \
  --min_height 5 \
  --max_height 5 \
  --pools_per_height 2 \
  --max_sequences_per_pool 10 \
  --workers 1

python extend_towers_pool_based.py \
  --out_dir pool_based_height_csv \
  --min_height 5 \
  --max_height 8 \
  --pools_per_height 20 \
  --workers 8 \
  --seed 0

python extend_towers_pool_based.py \
  --out_dir pool_based_height_csv \
  --min_height 9 \
  --max_height 16 \
  --pools_per_height 20 \
  --workers 8 \
  --seed 0

python extend_towers_pool_based.py \
  --out_dir pool_based_height_csv \
  --min_height 5 \
  --max_height 8 \
  --continue \
  -n 10 \
  --collapse_threshold 0.70 \
  --max_extra_pools_per_height 10 \
  --workers 8 \
  --seed 123


  python /home/color/Masaüstü/FD/downward/fast-downward.py --overall-time-limit 300 save/symcan_trial3/domain.pddl save/symcan_trial3/eval_task/inside/pddl/p_0053_N4_d79dcad31f7d.pddl \
  --search "eager_greedy([ff()])" \
  --search-memory-limit 4G \
  2>&1 | tee log.txt