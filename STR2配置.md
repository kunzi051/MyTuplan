`git clone https://github.com/Tsinghua-MARS-Lab/StateTransformer.git` 
`cd StateTransformer`

查阅conda nuplan环境的python版本`(nuplan)python --version`为3.9.20，创建一个新的conda环境 `conda create -n str2 python=3.9.20`

激活`str2`后，下载CUDA`pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

进入到StateTransformer根目录下，安装其他依赖 `pip install -r requirements.txt`

仍在StateTransformer根目录下，安装transformer4planning `pip install -e .`

下载模型（位于：https://huggingface.co/JohnZhan/StateTransformer2/tree/main） `huggingface-cli download --resume-download JohnZhan/StateTransformer2 --local-dir ./model`

run

```
python run_simulation_closed.py \
    --data_path ${NUPLAN_DATA_MINI_ROOT} \
    --map_path ${NUPLAN_MAPS_ROOT} \
    --model_path ${MODEL_PATH} \
    --split_filter_yaml ${SPLIT_FILTER_YAML} \
    --initstable_time 8 \
    --conservative_factor 0.8 \
    --comfort_weight 10.0 \
    --batch_size 7 \
    --processes-repetition 5 \
	--test_type 'closed_loop_reactive_agents' \
    --pdm_lateral_offsets none;
```

```
python run_nuplan_simulation.py \
--test_type closed_loop_nonreactive_agents \
--data_path ${NUPLAN_DATA_MINI_ROOT} \
--map_path ${NUPLAN_MAPS_ROOT} \
--model_path ${MODEL_PATH} \
--split_filter_yaml nuplan_simulation/lane_change_split.yaml \
--max_scenario_num 10 \
--batch_size 8 \
--device cuda \
--exp_folder TestHard14_MixtralM_CKS_SepLoss_AugCur50Pct_PastProj_S6_bf16_Jun28_ckpt150k\
--processes-repetition 8
```

```
/mnt/home/jiayukun/StateTransformer/nuplan_simulation/lane_change_split.yaml
```



逐个安装requirement.txt

.bashrc中，添加 `export NUPLAN_DEVKIT_PATH="/mnt/home/jiayukun/nuplan-devkit/"`，以运行 `run_nuplan_board.py`

报错

```
Exception caught in scenario: 0                                                                                                                            
Error message: local variable 'pred_length' referenced before assignment
Fetch Results:  50%|████████████████████████████████████████████████████▌                                                    | 1/2 [00:21<00:21, 21.74s/it]Exception caught in scenario: 1
Error message: local variable 'pred_length' referenced before assignment
Fetch Results: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:23<00:00, 11.68s/it]
2 scenarios failed during simulation.
writting to err_scenarios.log.
Traceback (most recent call last):
  File "/mnt/home/jiayukun/StateTransformer/run_simulation_closed.py", line 905, in <module>
    main(args)
  File "/mnt/home/jiayukun/StateTransformer/run_simulation_closed.py", line 842, in main
    build_simulation_in_batch_multiprocess(experiment_name, scenarios, output_dir, simulation_dir, metric_dir, args.batch_size, args=args)
  File "/mnt/home/jiayukun/StateTransformer/run_simulation_closed.py", line 608, in build_simulation_in_batch_multiprocess
    overall_score_dic, overall_score = compute_overall_score(over_all_metric_results, experiment)
  File "/mnt/home/jiayukun/StateTransformer/run_simulation_closed.py", line 115, in compute_overall_score
    for key in metric_dic.keys():
AttributeError: 'NoneType' object has no attribute 'keys'
[W117 03:03:38.115987956 CudaIPCTypes.cpp:96] Producer process tried to deallocate over 1000 memory blocks referred by consumer processes. Deallocation might be significantly slowed down. We assume it will never going to be the case, but if it is, please file but to https://github.com/pytorch/pytorch
[W117 03:03:38.414711699 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
```

