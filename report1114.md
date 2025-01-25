

## 1 训练

### 1.1 pdm_open

`train_pdm_open.sh`

```shell
TRAIN_EPOCHS=100
TRAIN_LR=1e-4
TRAIN_LR_MILESTONES=[50,75]
TRAIN_LR_DECAY=0.1
BATCH_SIZE=64
SEED=0

JOB_NAME=training_pdm_open_model
CACHE_PATH=$NUPLAN_DEVKIT_ROOT/nuplan/exp
USE_CACHE_WITHOUT_DATASET=False

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_training.py \
seed=$SEED \
py_func=train \
+training=training_pdm_open_model \
job_name=$JOB_NAME \
scenario_builder=nuplan \
cache.cache_path=$CACHE_PATH \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
lightning.trainer.params.max_epochs=$TRAIN_EPOCHS \
data_loader.params.batch_size=$BATCH_SIZE \
optimizer.lr=$TRAIN_LR \
lr_scheduler=multistep_lr \
lr_scheduler.milestones=$TRAIN_LR_MILESTONES \
lr_scheduler.gamma=$TRAIN_LR_DECAY \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.training, pkg://tuplan_garage.planning.script.experiments, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```



`tuplan_garage` 的`training_pdm_open_model.yaml`(line 15)

```yaml
# @package _global_
experiment_name: training_pdm_open_model
py_func: train
objective_aggregate_mode: mean

defaults:
  - override /data_augmentation:
  - override /objective:
      - l1_objective
  - override /splitter: nuplan
  - override /model: pdm_open_model
  - override /scenario_filter: train150k_split
  - override /training_metric:
      - avg_displacement_error
      - avg_heading_error
      - final_displacement_error
      - final_heading_error
```

`model`(line 11):

`tuplan_garage` 的`planning/script/training/modeling/models/pdm_open_model.py` 定义了`class PDMOpenModel`

```python
class PDMOpenModel(TorchModuleWrapper):
    '''
    直接预测自车的未来的轨迹。
    输入：
    1. 自车历史状态（位置、速度、加速度）
    2. 中心线
    输出：直接预测的轨迹
    
    包含两个编码器：state_encoding 和 centerline_encoding，分别用于编码自车历史状态和中心线。
	在前向传播中，它将编码后的特征拼接在一起，并通过 planner_head 网络直接预测未来的轨迹。
    '''
```



### 1.2 pdm_offset

只有`model`不同

```python
class PDMOffsetModel(TorchModuleWrapper):
    '''
    预测相对于 PDM-Closed 轨迹的修正量（offset）。
    输入：
    1. 自车历史状态（位置、速度、加速度）。
    2. PDM-Closed 轨迹。
    3. 中心线。
    输出：修正后的轨迹，它是通过对 PDM-Closed 轨迹加上预测的修正量得到的。
    
    包含三个编码器：state_encoding、centerline_encoding 和 trajectory_encoding，分别用于编码自车历史状态、中心线和 PDM-Closed 轨迹。
	在前向传播中，它将编码后的特征拼接在一起，并通过 planner_head 网络预测修正量。
    '''
```



## 2 评估/simulation

### 2.1 open_loop_boxes

`open_loop_boxes.sh` (no need CHECKPOINT)

```shell
SPLIT=val14_split
CHALLENGE=open_loop_boxes # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT=../epoch50-step6425.ckpt

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pdm_open_planner \
planner.pdm_open_planner.checkpoint_path=$CHECKPOINT \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```

### 2.2 closed_loop_nonreactive_agents

`closed_loop_noreactive_agents.sh` (no need CHECKPOINT)

```shell
SPLIT=val14_split
CHALLENGE=closed_loop_nonreactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pdm_closed_planner \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```



### 2.3 closed_loop_reactive_agents

`closed_loop_reactive_agents.sh`(need CHECKPOINT, line3)

```shell
SPLIT=val14_split
CHALLENGE=closed_loop_reactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT=../epoch35-step4535.ckpt

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=pdm_hybrid_planner \
planner.pdm_hybrid_planner.checkpoint_path=$CHECKPOINT \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```



----

```
两个train
	- pdm_open:   epoch50-step6425.ckpt(best model)
	- pdm_offset: epoch35-step4535.ckpt(best model)
	
三类simulation
	- pdm_open(open_loop_boxes)
	- pdm_closed(closed_loop_nonreactive_agents)
	- pdm_hybrid(closed_loop_reactive_agents)
```



## 3 场景

```yaml
_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
_convert_: 'all'
scenario_types:                     # List of scenario types to include
  - changing_lane
  - changing_lane_to_left
  - changing_lane_to_right
  - changing_lane_with_lead
  - changing_lane_with_trail
scenario_tokens: null               # List of scenario tokens to include
log_names: ${splitter.log_splits.test}  # Use all logs present in the test split
map_names: null                     # Filter scenarios by map names

num_scenarios_per_type: 50         # Number of scenarios per type
limit_total_scenarios: null         # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
timestamp_threshold_s: 0         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
ego_displacement_minimum_m: null    # Whether to remove scenarios where the ego moves less than a certain amount
ego_start_speed_threshold: null  # Limit to scenarios where the ego reaches a certain speed from below
ego_stop_speed_threshold: null   # Limit to scenarios where the ego reaches a certain speed from above
speed_noise_tolerance: null         # Value at or below which a speed change between two timepoints should be ignored as noise.

expand_scenarios: false   # Whether to expand multi-sample scenarios to multiple single-sample scenarios
remove_invalid_goals: false          # Whether to remove scenarios where the mission goal is invalid
shuffle: false                      # Whether to shuffle the scenarios
```





## 4 config

`/tuplan_garage/tuplan_garage/planning/script`

```
scripts
    ├── config 
    │	├── common
    │   ├── simulation
    │   └── training
    └── experiments
        └── training  
```



```
scripts
    ├── config 
    │     ├── common 
    │     │    ├── model 
    │     │    │    ├── pdm_offset_model.yaml 
    │     │    │    └── pdm_open_model.yaml
    │     │    └── scenario_filter
    │     │         ├── lane_change_split.yaml
    │     │         ├── reduced_val14_split.yaml
    │     │         ├── train150k_split.yaml
    │     │         └── val14_split.yaml
    │     ├── simulation
    │     │    └── planner
    │     │         ├── pdm_closed_planner.yaml
    │     │         ├── pdm_hybrid_planner.yaml
    │     │         └── pdm_open_planner.yaml        
    │     └── training
    │          ├── callbacks
    │          │    └── multimodal_visualization_callback.yaml
    │          ├── data_augmentation
    │          │	├── pgp_agent_dropout_augmentation.yaml
    │          │    ├── pgp_ego_history_dropout_augmentation.yaml
    │          │    └── pgp_kinematic_agent_augmentation.yaml 
    │          ├── objective
    │          │	├── l1_objective.yaml
    │          │    ├── pgp_minADE_objective.yaml
    │          │    └── pgp_traversal_objective.yaml    
    │          └── training_metric
    │           	├── min_avg_displacement_error.yaml
    │               └── min_final_displacement_error.yaml      
    └── experiments
          └── training  
               ├── training_pdm_offset_model.yaml  
               └── training_pdm_open_model.yaml    
```

