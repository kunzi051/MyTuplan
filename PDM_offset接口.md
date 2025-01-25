# 1 yukun

## 1 PDM_feature_builder接口

| 变量名称                | 维度                          | 行数                                                         | 列数                  |
| ----------------------- | ----------------------------- | ------------------------------------------------------------ | --------------------- |
| ego_position            | 11*3                          | past 2s * 5Hz +  current_state                               | (x, y, heading)       |
| ego_velocity            | 11*3                          | past 2s * 5Hz +  current_state                               | (v_x, v_y, v_heading) |
| ego_acceleration        | 11*3                          | past 2s * 5Hz +  current_state                               | (a_x, a_y, a_heading) |
| planner_centerline      | 120*3                         | 120m，分辨率1m。centerline_samples: int =  120, centerline_interval: float = 1.0 | (x, y, heading)       |
| (del)planner_trajectory | 16*3                          | num_poses = 16，用16个点表示8s                               | (x, y, heading)       |
| closed_loop_trajectory  | InterpolatedTrajectory object |                                                              |                       |



## 2 PDM_offset Model

```python
torch_module_wrapper = PDMOffsetModel(
  (state_encoding): Sequential(
    (0): Linear(in_features=99, out_features=512, bias=True) # in_features解释：ego_position特征是11*3=33，ego_velocity和ego_acceleration同理，因此是33+33+33=99
    (1): ReLU()
  )
  (centerline_encoding): Sequential(
    (0): Linear(in_features=360, out_features=512, bias=True) # in_features解释：120*3 = 360
    (1): ReLU()
  )
  (trajectory_encoding): Sequential(
    (0): Linear(in_features=48, out_features=512, bias=True)# in_features解释：16*3 = 48
    (1): ReLU()
  )  
   (planner_head): Sequential(
    (0): Linear(in_features=1536, out_features=512, bias=True) # in_features解释：这里的1536就是上面state_encoding、centerline_encoding、trajectory_encoding输出特征512+512+512
    (1): Dropout(p=0.1, inplace=False)
    (2): ReLU()
    (3): Linear(in_features=512, out_features=512, bias=True)
    (4): ReLU()
    (5): Linear(in_features=512, out_features=48, bias=True)
  )
)
```



## 3 debug过程中的输出

### 3.1 pdm_features

```
{'pdm_features': PDMFeature(ego_position=tensor([[-2.2154e+01, -9.5604e-02,  8.0270e-03],
        [-1.9907e+01, -7.4342e-02,  7.5223e-03],
        [-1.7654e+01, -5.6905e-02,  7.9175e-03],
        [-1.5422e+01, -3.7455e-02,  8.3791e-03],
        [-1.3198e+01, -2.9224e-02,  6.5672e-03],
        [-1.0978e+01, -2.1101e-02,  5.4435e-03],
        [-8.7625e+00, -1.5822e-02,  4.6055e-03],
        [-6.5544e+00, -1.2576e-02,  3.3516e-03],
        [-4.3526e+00, -4.3854e-03,  2.1749e-03],
        [-2.1756e+00, -2.9015e-03,  1.3980e-03],
        [ 4.6566e-10,  1.1642e-10,  0.0000e+00]]), ego_velocity=tensor([[ 1.1541e+01, -2.1420e-01,  2.6081e-03],
        [ 1.1559e+01, -2.1076e-01,  7.8846e-04],
        [ 1.1549e+01, -2.2011e-01, -1.0312e-03],
        [ 1.1517e+01, -2.1141e-01, -2.7534e-03],
        [ 1.1487e+01, -2.1884e-01, -4.7789e-03],
        [ 1.1449e+01, -2.2863e-01, -6.0072e-03],
        [ 1.1404e+01, -2.3673e-01, -5.4372e-03],
        [ 1.1350e+01, -2.4136e-01, -5.2597e-03],
        [ 1.1306e+01, -2.3174e-01, -5.5811e-03],
        [ 1.1250e+01, -2.1443e-01, -5.5014e-03],
        [ 1.1207e+01, -2.0479e-01, -5.4217e-03]], dtype=torch.float64), ego_acceleration=tensor([[ 0.0185,  0.0854,  0.0702],
        [-0.0274,  0.0296,  0.0306],
        [-0.0670, -0.0106, -0.0091],
        [-0.1004, -0.0352, -0.0190],
        [-0.1263, -0.0327, -0.0068],
        [-0.1526, -0.0536,  0.0050],
        [-0.1899, -0.0207, -0.0019],
        [-0.2065, -0.0048,  0.0007],
        [-0.2450,  0.0345,  0.0004],
        [-0.2907,  0.0841, -0.0083],
        [-0.3436,  0.1438, -0.0171]], dtype=torch.float64), planner_centerline=tensor([[ 4.4645e-04,  1.9313e-01, -2.4408e-03],
        [ 1.0004e+00,  1.9032e-01, -3.2941e-03],
        [ 2.0004e+00,  1.8735e-01, -2.3624e-03],
        [ 3.0004e+00,  1.8536e-01, -1.3624e-03],
        [ 4.0004e+00,  1.8438e-01, -3.6242e-04],
        [ 5.0004e+00,  1.8439e-01,  6.3759e-04],
        [ 6.0004e+00,  1.8540e-01,  1.6376e-03],
        [ 7.0004e+00,  1.8741e-01,  2.6376e-03],
        [ 8.0004e+00,  1.9043e-01,  3.6376e-03],
        [ 9.0004e+00,  1.9429e-01,  3.9710e-03],
        [ 1.0000e+01,  1.9826e-01,  3.9710e-03],
        [ 1.1000e+01,  2.0223e-01,  3.9710e-03],
        [ 1.2000e+01,  2.0620e-01,  3.9710e-03],
        [ 1.3000e+01,  2.1017e-01,  3.9710e-03],
        [ 1.4000e+01,  2.1414e-01,  3.9710e-03],
        [ 1.5000e+01,  2.1811e-01,  3.9710e-03],
        [ 1.6000e+01,  2.2208e-01,  3.9710e-03],
        [ 1.7000e+01,  2.2605e-01,  3.9710e-03],
        [ 1.8000e+01,  2.3002e-01,  3.9710e-03],
        [ 1.9000e+01,  2.3400e-01,  3.9710e-03],
        [ 2.0000e+01,  2.3797e-01,  3.9710e-03],
        [ 2.1000e+01,  2.4194e-01,  3.9710e-03],
        [ 2.2000e+01,  2.4591e-01,  3.9710e-03],
        [ 2.3000e+01,  2.4988e-01,  3.9710e-03],
        [ 2.4000e+01,  2.5385e-01,  3.9710e-03],
        [ 2.5000e+01,  2.5766e-01,  3.2823e-03],
        [ 2.6000e+01,  2.6073e-01,  3.3035e-03],
        [ 2.7000e+01,  2.6440e-01,  4.3035e-03],
        [ 2.8000e+01,  2.6908e-01,  5.3035e-03],
        [ 2.9000e+01,  2.7476e-01,  6.3035e-03],
        [ 3.0000e+01,  2.8144e-01,  7.3035e-03],
        [ 3.1000e+01,  2.8912e-01,  8.3035e-03],
        [ 3.2000e+01,  2.9780e-01,  9.3035e-03],
        [ 3.3000e+01,  3.0748e-01,  1.0303e-02],
        [ 3.4000e+01,  3.1793e-01,  1.0524e-02],
        [ 3.5000e+01,  3.2846e-01,  1.0524e-02],
        [ 3.6000e+01,  3.3898e-01,  1.0524e-02],
        [ 3.7000e+01,  3.4950e-01,  1.0524e-02],
        [ 3.8000e+01,  3.6003e-01,  1.0524e-02],
        [ 3.9000e+01,  3.7055e-01,  1.0524e-02],
        [ 4.0000e+01,  3.8108e-01,  1.0524e-02],
        [ 4.1000e+01,  3.9160e-01,  1.0524e-02],
        [ 4.2000e+01,  4.0213e-01,  1.0524e-02],
        [ 4.3000e+01,  4.1265e-01,  1.0524e-02],
        [ 4.4000e+01,  4.2317e-01,  1.0524e-02],
        [ 4.4999e+01,  4.3370e-01,  1.0524e-02],
        [ 4.5999e+01,  4.4422e-01,  1.0524e-02],
        [ 4.6999e+01,  4.5475e-01,  1.0524e-02],
        [ 4.7999e+01,  4.6527e-01,  1.0524e-02],
        [ 4.8999e+01,  4.7580e-01,  1.0524e-02],
        [ 4.9999e+01,  4.8632e-01,  1.0524e-02],
        [ 5.0999e+01,  4.9684e-01,  1.0524e-02],
        [ 5.1999e+01,  5.0737e-01,  1.0524e-02],
        [ 5.2999e+01,  5.1789e-01,  1.0524e-02],
        [ 5.3999e+01,  5.2842e-01,  1.0524e-02],
        [ 5.4999e+01,  5.3894e-01,  1.0385e-02],
        [ 5.5999e+01,  5.4896e-01,  9.4031e-03],
        [ 5.6999e+01,  5.5799e-01,  8.4031e-03],
        [ 5.7999e+01,  5.6602e-01,  7.4031e-03],
        [ 5.8999e+01,  5.7305e-01,  6.4031e-03],
        [ 5.9999e+01,  5.7907e-01,  5.4031e-03],
        [ 6.0999e+01,  5.8410e-01,  4.4031e-03],
        [ 6.1999e+01,  5.8813e-01,  3.4661e-03],
        [ 6.2999e+01,  5.9193e-01,  4.3890e-03],
        [ 6.3999e+01,  5.9669e-01,  5.3890e-03],
        [ 6.4999e+01,  6.0246e-01,  6.3890e-03],
        [ 6.5999e+01,  6.0912e-01,  6.8364e-03],
        [ 6.6999e+01,  6.1596e-01,  6.8364e-03],
        [ 6.7999e+01,  6.2279e-01,  6.8364e-03],
        [ 6.8999e+01,  6.2963e-01,  6.8364e-03],
        [ 6.9999e+01,  6.3647e-01,  6.8364e-03],
        [ 7.0999e+01,  6.4330e-01,  6.8364e-03],
        [ 7.1999e+01,  6.5014e-01,  6.8364e-03],
        [ 7.2998e+01,  6.5697e-01,  6.8364e-03],
        [ 7.3998e+01,  6.6381e-01,  6.8364e-03],
        [ 7.4998e+01,  6.7065e-01,  6.8364e-03],
        [ 7.5998e+01,  6.7748e-01,  6.8364e-03],
        [ 7.6998e+01,  6.8432e-01,  6.8364e-03],
        [ 7.7998e+01,  6.9116e-01,  6.8364e-03],
        [ 7.8998e+01,  6.9799e-01,  6.8364e-03],
        [ 7.9998e+01,  7.0483e-01,  6.8364e-03],
        [ 8.0998e+01,  7.1166e-01,  6.8364e-03],
        [ 8.1998e+01,  7.1850e-01,  6.8364e-03],
        [ 8.2998e+01,  7.2534e-01,  6.8364e-03],
        [ 8.3998e+01,  7.3217e-01,  6.8364e-03],
        [ 8.4998e+01,  7.3901e-01,  6.8364e-03],
        [ 8.5998e+01,  7.4585e-01,  6.8364e-03],
        [ 8.6998e+01,  7.5268e-01,  6.8364e-03],
        [ 8.7998e+01,  7.5952e-01,  6.8364e-03],
        [ 8.8998e+01,  7.6636e-01,  6.8364e-03],
        [ 8.9998e+01,  7.7319e-01,  6.8364e-03],
        [ 9.0998e+01,  7.8003e-01,  6.8364e-03],
        [ 9.1998e+01,  7.8686e-01,  6.8364e-03],
        [ 9.2998e+01,  7.9370e-01,  6.8364e-03],
        [ 9.3998e+01,  8.0054e-01,  6.8364e-03],
        [ 9.4998e+01,  8.0737e-01,  6.8364e-03],
        [ 9.5998e+01,  8.1421e-01,  6.8364e-03],
        [ 9.6998e+01,  8.2105e-01,  6.8364e-03],
        [ 9.7998e+01,  8.2788e-01,  6.8364e-03],
        [ 9.8998e+01,  8.3472e-01,  6.8364e-03],
        [ 9.9998e+01,  8.4156e-01,  6.8364e-03],
        [ 1.0100e+02,  8.4840e-01,  6.9835e-03],
        [ 1.0200e+02,  8.5574e-01,  7.8468e-03],
        [ 1.0300e+02,  8.6358e-01,  7.8468e-03],
        [ 1.0400e+02,  8.7143e-01,  7.8468e-03],
        [ 1.0500e+02,  8.7928e-01,  7.8468e-03],
        [ 1.0600e+02,  8.8712e-01,  7.8468e-03],
        [ 1.0700e+02,  8.9492e-01,  7.4144e-03],
        [ 1.0800e+02,  9.0196e-01,  6.4144e-03],
        [ 1.0900e+02,  9.0800e-01,  5.4144e-03],
        [ 1.1000e+02,  9.1304e-01,  4.4144e-03],
        [ 1.1100e+02,  9.1708e-01,  3.4144e-03],
        [ 1.1200e+02,  9.2024e-01,  3.0329e-03],
        [ 1.1300e+02,  9.2328e-01,  3.0329e-03],
        [ 1.1400e+02,  9.2631e-01,  3.0329e-03],
        [ 1.1500e+02,  9.2934e-01,  3.0329e-03],
        [ 1.1600e+02,  9.3237e-01,  3.0329e-03],
        [ 1.1700e+02,  9.3541e-01,  3.0329e-03],
        [ 1.1800e+02,  9.3844e-01,  3.0329e-03],
        [ 1.1900e+02,  9.4147e-01,  3.0329e-03]], dtype=torch.float64), planner_trajectory=tensor([[4.9997e+00, 1.8439e-01, 6.3688e-04],
        [1.0719e+01, 2.0111e-01, 3.9710e-03],
        [1.6610e+01, 2.2450e-01, 3.9710e-03],
        [2.2626e+01, 2.4839e-01, 3.9710e-03],
        [2.8725e+01, 2.7310e-01, 6.0284e-03],
        [3.4874e+01, 3.2713e-01, 1.0524e-02],
        [4.1046e+01, 3.9209e-01, 1.0524e-02],
        [4.7224e+01, 4.5711e-01, 1.0524e-02],
        [5.3393e+01, 5.2204e-01, 1.0524e-02],
        [5.9545e+01, 5.7646e-01, 5.8567e-03],
        [6.5673e+01, 6.0689e-01, 6.8364e-03],
        [7.1770e+01, 6.4858e-01, 6.8364e-03],
        [7.7845e+01, 6.9010e-01, 6.8364e-03],
        [8.4074e+01, 7.3269e-01, 6.8364e-03],
        [9.0474e+01, 7.7644e-01, 6.8364e-03],
        [9.6990e+01, 8.2099e-01, 6.8364e-03]], dtype=torch.float64))}
```

### 3.2 targets

```
targets = {'trajectory': Trajectory(data=tensor([[ 5.3920e+00, -5.3059e-03, -2.4199e-03],
        [ 1.0726e+01, -2.8940e-02, -5.1878e-03],
        [ 1.6020e+01, -4.8746e-02, -8.1108e-03],
        [ 2.1258e+01, -7.4496e-02,  4.5784e-04],
        [ 2.6415e+01, -3.9504e-02,  1.2896e-02],
        [ 3.1528e+01,  5.1336e-02,  2.1029e-02],
        [ 3.6540e+01,  1.7505e-01,  2.7319e-02],
        [ 4.1498e+01,  3.1559e-01,  2.5495e-02],
        [ 4.6357e+01,  4.3224e-01,  2.4148e-02],
        [ 5.1117e+01,  5.3710e-01,  2.1872e-02],
        [ 5.5688e+01,  6.2269e-01,  1.8109e-02],
        [ 6.0079e+01,  6.8063e-01,  1.4353e-02],
        [ 6.4286e+01,  7.2021e-01,  1.1143e-02],
        [ 6.8308e+01,  7.4568e-01,  7.9490e-03],
        [ 7.2070e+01,  7.7218e-01,  5.2630e-03],
        [ 7.5516e+01,  7.8670e-01,  3.6246e-03]]))}
```



## 4 位置

就在pdm_feature_builder return PDMFeature处截断



# 2 ouyang

接口整理

## 1 一帧结构

拿到的都是⼀帧⼀帧进来的结构，所以如果要使⽤历史 10 帧的结果是需要自己保存的。保存的过程中最主要是注意坐标的转换问题



## 2 自车历史信息

对应 tuplan 中的 ego position, ego velocity, ego acceleration  结构类型

### 2.1 结构类型

```cpp
struct PI_VehicleMotionStateData final
 {
     /// \see PI_VehicleMotionState
     PI_VehicleMotionState longitudinal_direction{};
     /// \see PI_VelocityWithQuality
     PI_VelocityWithQuality longitudinal_velocity{};
     /// \see PI_VelocityWithQuality
     PI_VelocityWithQuality lateral_velocity{};
     /// \see PI_VelocityMinMaxWithQuality
     PI_VelocityMinMaxWithQuality vehicle_speed_asilb{};
     /// \see PI_AccelerationWithQuality
     PI_AccelerationWithQuality longitudinal_acceleration{};
     /// \see PI_AccelerationMinMaxWithQuality
     PI_AccelerationMinMaxWithQuality longitudinal_acceleration_asilb{};
     /// \see PI_AccelerationWithQuality
     PI_AccelerationWithQuality longitudinal_acceleration_unfiltered{};
     /// \see PI_AccelerationWithQuality
     PI_AccelerationWithQuality lateral_acceleration{};
     /// \see PI_AccelerationWithQuality
     PI_AccelerationWithQuality vertical_acceleration{};
     /// \see PI_AngleRateWithQuality
     PI_AngleRateWithQuality roll_rate{};
     /// \see PI_AngleRateWithQuality
     PI_AngleRateWithQuality pitch_rate{};
     /// \see PI_AngleRateWithQuality
     PI_AngleRateWithQuality yaw_rate{};
     /// \see PI_RoadInclination
     PI_RoadInclination road_inclination{};
     /// \see PI_Timestamp_int64
     PI_Timestamp_int64 timestamp{{0}};
 };
 // 速度的结构
struct PI_VelocityWithQuality final
 {
     /// \see PI_EgovehicleVelocity_f32
     PI_EgovehicleVelocity_f32 velocity{{0.0F}};
     /// \see PI_EgoDataQuality
     PI_EgoDataQuality quality{PI_EgoDataQuality::kUndefined};
 };
 // 加速度的结构
struct PI_AccelerationWithQuality final
{
     /// \see PI_EgovehicleAcceleration_f32
     PI_EgovehicleAcceleration_f32 acceleration{{0.0F}};
     /// \see PI_EgoDataQuality
     PI_EgoDataQuality quality{PI_EgoDataQuality::kUndefined};
};
 // 转角速率的结构
struct PI_AngleRateWithQuality final
{
     /// \see PI_EgovehicleAngleRate_f32
     PI_EgovehicleAngleRate_f32 angle_rate{{0.0F}};
     /// \see PI_EgoDataQuality
     PI_EgoDataQuality quality{PI_EgoDataQuality::kUndefined};
};
```



### 2.2 数据示例及使用

比如说现在拿到⼀个类型为 `PI_VehicleMotionStateData`  的变量 `ego_state`  ，要取纵向速度 方式是 `ego_state.longitudinal_velocity.velocity.meters_per_second.value`，通过这种方式拿到的就是⼀个 float 类型的值；

| frame | value    |
| ----- | -------- |
| 0     | 9.459597 |
| 1     | 9.45812  |
| 2     | 9.454987 |



## 3 centerline

是车道中心线

- 首先这里有⼀个问题是这个 centerline 是只要自车所走的中心线吗？--是的
- 其他障碍物需不要车道中心线？ --不需要
- 如果是自车的那么用下面这个接口应该就行



### 3.1 结构类型

```cpp
struct PI_LaneCenterModel final
 {
     /// \see PI_EgoLaneGeometry
     PI_EgoLaneGeometry ego_lane{};
     /// \see PI_AdjacentLaneGeometry
     PI_AdjacentLaneGeometry adjacent_right_lane{};
     /// \see PI_AdjacentLaneGeometry
     PI_AdjacentLaneGeometry adjacent_left_lane{};
     /// \see PI_LaneChangeType
     PI_LaneChangeType lane_change{PI_LaneChangeType::kUnchanged};
     /// \see PI_RoadEstimationSource
     PI_RoadEstimationSource source{PI_RoadEstimationSource::kUnknown};
     /// \see PI_Timestamp_int64
     PI_Timestamp_int64 timestamp{{0}};
 };

```

```cpp
 // 其中，PI_LANECENTER_NUMBEROFROADMODELPOINTS = 30，也就是最多保存30个点
struct PI_EgoLaneGeometry final
 {
    /// \see PI_LongPosition_f32
    /// Longitude of the ego lane geometry along the road. The large min/max
    /// numbers are required to handle some u-turns.
    zenseact::ztd::Array<PI_LongPosition_f32, 
	PI_LANECENTER_NUMBEROFROADMODELPOINTS> x{};
    /// \see PI_LatPosition_f32
    /// Latitude of the ego lane geometry along the road. The large min/max numbers
    /// are required to handle some u-turns.
    zenseact::ztd::Array<PI_LatPosition_f32, 
	PI_LANECENTER_NUMBEROFROADMODELPOINTS> y{};
    /// \see PI_NormalVecStdDev_f32
    /// Std of the normal vector to the estimated lane geometry along the road.
    zenseact::ztd::Array<PI_NormalVecStdDev_f32, 
	PI_LANECENTER_NUMBEROFROADMODELPOINTS> normal_vector_std{};
    /// \see PI_Distance_f32
    /// Difference in estimated lateral displacement (inconsistency) compared to
    /// previous geometries at each arc length.
    zenseact::ztd::Array<PI_Distance_f32, 
    PI_LANECENTER_NUMBEROFROADMODELPOINTS> estimation_inconsistency{};
    /// \see PI_ArcLength_f32
    /// Distance along the road.
    zenseact::ztd::Array<PI_ArcLength_f32, 
    PI_LANECENTER_NUMBEROFROADMODELPOINTS> arc_length{};
    /// \see PI_HeadingAngle_f32
    /// Heading of the ego lane geometry along the road.
    zenseact::ztd::Array<PI_HeadingAngle_f32, 
	PI_LANECENTER_NUMBEROFROADMODELPOINTS> heading{};
    /// \see PI_CurvatureForLongitudinalSample_f32
    /// Curvature of the ego lane geometry along the road.
    zenseact::ztd::Array<PI_CurvatureForLongitudinalSample_f32,
    I_LANECENTER_NUMBEROFROADMODELPOINTS> curvature{};
    /// \see PI_Generic_uint8
    /// Number of points that have been populated with data.
    PI_Generic_uint8 n_populated_points{{0U}};
    /// \see PI_CurvatureForLongitudinalSample_f32
    /// Curvature of ego lane center at ego vehicle position, i.e at x = 0.
    PI_CurvatureForLongitudinalSample_f32 current_curvature{{0.0F}};
    /// \see PI_HeadingAngle_f32
    /// The heading angle of the ego lane geometry at ego vehicle position, i.e at
    /// x = 0. It is defined relative to the longitudinal axis of the ego vehicle.
    /// E.g. if the road is pointing left viewed from ego vehicle, this value will
    /// be positive.
    PI_HeadingAngle_f32 current_heading{{0.0F}};
    /// \see PI_Width_f32
    /// Width of ego lane at ego vehicle position, i.e at x = 0. Specified as the
    /// distance between the inner edge of the lane markers or road edges.
    PI_Width_f32 lane_width{{0.0F}};
    /// \see PI_Generic_bool
	/// Flag for lane width being available and valid
     PI_Generic_bool valid_lane_width{{false}};
     /// \see PI_Generic_uint8
     /// Number of points in model range, model range is the distance up to which
     /// the model has been confirmed to align with measurements.
     PI_Generic_uint8 n_points_in_model_range{{0U}};
     /// \see PI_LaneModelValidity
     PI_LaneModelValidity validity{PI_LaneModelValidity::kInvalid};
     /// \see PI_LaneModelValidity
     /// Temporary signal to indicate wheter the lane center model would have been
     /// active in case self assessment would have been enabled in RGF. Will be
     /// removed once it is enabled.
     PI_LaneModelValidity self_assessment_result{PI_LaneModelValidity::kInvalid};
 };
```

- `x` 和 `y`：这两个数组分别存储了自车道几何沿道路的经度和纬度位置。
- `normal_vector_std`：存储了沿道路的自车道几何的法向量的标准差。
- `estimation_inconsistency`：存储了与之前几何估计的不一致性，即横向位移的差异。
- `arc_length`：存储了沿道路的距离。
- `heading`：存储了自车道几何沿道路的朝向角度。
- `curvature`：存储了自车道几何沿道路的曲率。
- `n_populated_points`：存储了已填充数据的点的数量。
- `current_curvature` 和 `current_heading`：分别存储了自车位置处的当前曲率和朝向角度。
- `lane_width` 和 `valid_lane_width`：分别存储了自车位置处的车道宽度和车道宽度是否有效。
- `n_points_in_model_range`：存储了模型范围内的点的数量，即模型已确认与测量结果对齐的距离。
- `validity` 和 `self_assessment_result`：分别存储了车道模型的有效性和自我评估结果。

---

应该需要`x y heading` 、取120米长。



### 3.2 如何使用

比如说你要取 ego lane 的 30 个点的 x，`x = view(lane_center_model.ego_lane.x)`  ，这个拿到的 x 可以认为是个数组，比如要取第一个点`x0=x.at(0).meters.value`



## 4 planner trajectory

输出没有要求要写成 zen 的格式，因为我们是开环的，用模型本身的这种数据类型就可以 -- `InterpolatedTrajectory`类

```python
class InterpolatedTrajectory(AbstractTrajectory):
    """Class representing a trajectory that can be interpolated from a list of points."""
```



## 5 targets

怎么没在表格里？ --targets是真值。

```cpp
struct PI_TargetTrackingOutput final
 {
	/// \see PI_TargetTrackingStatus
    /// Defines if the target tracking module is active or not.
    PI_TargetTrackingStatus is_active{};
    /// \see PI_StatusPerSensor
    /// Describes the current status for each sensor.
    PI_StatusPerSensor sensor_status{};
    /// \see PI_Timestamp_int64
    /// Indicates the time since the sensor was updated with a valid detection.
    /// Sensors are ordered according to SensorIdentifier. Should be used to
    /// determine if the field-of-view of a sensor is being tracked. It is left up
    /// to the consumer to determine how long a sensor's field-of-view is allowed
    /// to go without any updates.
    zenseact::ztd::Array<PI_Timestamp_int64, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFSENSORS>
        time_since_sensor_is_updated{};
    /// \see PI_Timestamp_int64
    /// Timestamp of current output.
    PI_Timestamp_int64 timestamp{};
    /// \see PI_ObjectStates
    /// Contains information about the object states.
    PI_ObjectStates object_states{};
    /// \see PI_ConfidenceIntervals
    /// State uncertainty information for attributes in PI_ObjectStates.
    PI_ConfidenceIntervals confidence_intervals{};
    /// \see PI_ObjectBoxes
    /// Contains the extent of the object boxes.
    PI_ObjectBoxes object_boxes{};
    /// \see PI_ObjectAttributes
    /// Contains attributes of the objects.
    PI_ObjectAttributes object_attributes{};
    /// \see PI_ObjectRoadStates
    /// Object states given in a lane center coordinate system.
    PI_ObjectRoadStates object_road_states{};
    /// \see PI_AssociationData
    /// Contains information about which camera detection that has been associated
    /// to what track.
    PI_AssociationData association_data{};
    /// \see PI_ObjectValidation
    /// Whether a track has been validated to be present based on sensor detections
    /// or not.
    zenseact::ztd::Array<PI_ObjectValidation, 
	PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS>
        validation{};
    /// \see PI_TrackedObjectsApiData
    /// The API for tracked objects.
    PI_TrackedObjectsApiData tracked_objects_api_data{};
    /// \see PI_TargetMotionEstimateOutput
    /// Target motion estimate associated with the object.
    zenseact::ztd::Array<PI_TargetMotionEstimateOutput, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS>
        target_motion_estimate{};
 };
 // 
struct PI_ObjectStates final
 {
    /// \see PI_ObjLonPosition_f32
    /// The longitudinal position relative ego for the specified reference point.
    zenseact::ztd::Array<PI_ObjLonPosition_f32, 
	PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS>
        lon_pos{};
    /// \see PI_ObjLatPosition_f32
    /// The lateral position relative ego for the specified reference point.
    zenseact::ztd::Array<PI_ObjLatPosition_f32, 
	PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS>
        lat_pos{};
    /// \see PI_ObjHeadingAngle_f32
    /// The angle in which the object will move relative to the ground from the
    /// x-axis in the ego vehicle coordinate system
    zenseact::ztd::Array<PI_ObjHeadingAngle_f32, 
	PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS>
        heading{};
    /// \see PI_ObjSpeed_f32
    /// Speed in the direction of the heading, relative to the ground.
    zenseact::ztd::Array<PI_ObjSpeed_f32, 
	PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS> speed{};
    /// \see PI_ObjAcceleration_f32
    /// Acceleration in the direction of the heading, relative to the ground.
    zenseact::ztd::Array<PI_ObjAcceleration_f32, 
	PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS>
	acceleration{};
    /// \see PI_Generic_bool
    /// If the acceleration value is valid or not.
    zenseact::ztd::Array<PI_Generic_bool, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS>
        is_acceleration_valid{};
    /// \see PI_ObjYawRate_f32
    /// Yaw rate relative to the ground.
    zenseact::ztd::Array<PI_ObjYawRate_f32, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS> yawrate{};
    /// \see PI_Generic_bool
    /// If the yaw rate value is valid or not.
    zenseact::ztd::Array<PI_Generic_bool, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS>
        is_yawrate_valid{};
    /// \see PI_OutputRefPoint
    /// Which point of the box that the position is given for.
    zenseact::ztd::Array<PI_OutputRefPoint, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS> ref_point{};
};
struct PI_ObjectBoxes final
{
    /// \see PI_ObjLength_f32
    zenseact::ztd::Array<PI_ObjLength_f32, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS> length{};
    /// \see PI_Width_f32
    zenseact::ztd::Array<PI_Width_f32, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS> width{};
    /// \see PI_ObjHeight_f32
    zenseact::ztd::Array<PI_ObjHeight_f32, 
    PI_TARGETTRACKINGOUTPUT_MAXNUMBEROFOBJECTS> height{};
};
```



使用：这个和 centerline 的⽤法是⼀样的



## 6 接到车上的流程

* ⻋端⼜叫 dds ，会发送数据，然后模型作为⼀个使⽤者只需要订阅数据即 topic 就⾏了（这个逻辑和 ros 应该是差不多的） —— 所以通信⽂件这部分是要⾃⼰写的，参考下⾯这个（应该是 read data 和 write data 相关的三个⽂件）； 
* route_matching_reader.py
* route_writer.py
* base.py
* 订阅了数据之后就能收到数据，然后就组装数据 —— 这部分要完全⾃⼰写的就是保存历史帧的数据 并做坐标转换，然后组装操作参考模型原来的应该也差不多，反正都是⽤ python 写； 
* 数据喂给模型 —— ⼀定记住这不是在训练模型，这是已经有了⼀个模型在进⾏预测了，所以不要照 搬 train 模型相关的脚本命令； 
* 输出可视化 —— 可以先在本地简单可视化，然后上⻋的时候估计还要接到其他可视化⼯具⾥⾯；



# 3 Q&A

1 planner需要的是goals和maps?

--不完全是。需要的是：

```python
def create_pdm_feature(
    model: TorchModuleWrapper,
    planner_input: PlannerInput,
    centerline: PDMPath,
    closed_loop_trajectory: Optional[InterpolatedTrajectory] = None,
    device: str = "cpu",
) -> PDMFeature:
其中：
class PlannerInput:
    """
    Input to a planner for which a trajectory should be computed.
    """

    iteration: SimulationIteration  # Iteration and time in a simulation progress
    history: SimulationHistoryBuffer  # Rolling buffer containing past observations and states.
    traffic_light_data: Optional[List[TrafficLightStatusData]] = None  # The traffic light status data
```

综上，总计`iteration: SimulationIteration、history: SimulationHistoryBuffer、traffic_light_data: Optional[List[TrafficLightStatusData]] = None、centerline: PDMPath、closed_loop_trajectory: Optional[InterpolatedTrajectory] = None`

`simulation.py`

```python
    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps 
         都是从scenario中得到的：
         得到route_roadblock_ids、mission_goal和map_api。
         route_roadblock_ids: List[str]  # Roadblock ids comprising goal route
    	 mission_goal: StateSE2  # The mission goal which commonly is not achievable in a single scenario
        :return data needed for planner initialization.
        """
        self.reset()

        # Initialize history from scenario
        self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
            self._history_buffer_size, self._scenario, self._observations.observation_type()
        )

        # Initialize observations
        self._observations.initialize()

        # Add the current state into the history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

        # Return the planner initialization structure for this simulation
        return PlannerInitialization(
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            mission_goal=self._scenario.get_mission_goal(),
            map_api=self._scenario.map_api,
        )
```



2 targets

train的时候可能是从cache中导入feature，复用。但simulation时都是计算feature

```python
def compute_or_load_feature(
    scenario: AbstractScenario,
    cache_path: Optional[pathlib.Path],
    builder: Union[AbstractFeatureBuilder, AbstractTargetBuilder],
    storing_mechanism: FeatureCache,
    force_feature_computation: bool,
) -> Tuple[AbstractModelFeature, Optional[CacheMetadataEntry]]:
    """
    Compute features if non existent in cache, otherwise load them from cache
    :param scenario: for which features should be computed
    :param cache_path: location of cached features
    :param builder: which builder should compute the features
    :param storing_mechanism: a way to store features
    :param force_feature_computation: if true, even if cache exists, it will be overwritten
    :return features computed with builder and the metadata entry for the computed feature if feature is valid.
    """
    ...

		if isinstance(builder, AbstractFeatureBuilder):
            feature = builder.get_features_from_scenario(scenario)
        elif isinstance(builder, AbstractTargetBuilder):
            feature = builder.get_targets(scenario)
```



```python
class AbstractTargetBuilder(ABC):
    """
    Abstract class that creates model output targets from database samples.
    """

    @classmethod
    @abstractmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """
        :return type of feature which will be generated
        """
        pass

    @classmethod
    @abstractmethod
    def get_feature_unique_name(cls) -> str:
        """
        :return a unique string identifier of generated feature
        """
        pass

    @abstractmethod
    def get_targets(self, scenario: AbstractScenario) -> AbstractModelFeature:
        """
        Constructs model output targets from database scenario.

        :param scenario: generic scenario
        :return: constructed targets
        """
        pass
```



3 cache

cache 缓存在nuplan/../exp下，约45个log文件。为什么会有trajectory? -- 输出结果

用的哪个model？上一次的model？pdm-open还是offset? --  内部应该有分类？如果当前训练的是open模型，就使用之前训练open时的缓存，如果offset，就之前offset缓存。

如果上一次model不理想，也会对本次训练结果产生影响？--是的

有pdm_feature但是没有targets?  -- targets是真值



4 对前面整理接口的一些确认

确认targets 包含什么，是不是他车及其他障碍物的信息。--不是。

targets是在PDM-closed里作为输入的？--不是。

确认centerline是否需要他车的道路信息、还是只需要自车即可 --自车。

simu 中 整理出来的接口（自车历史信息等）是在buffer initialization中出现的。 -- 也不是。是在

​	`simulation.py `- def initialize - self._history_buffer - SimulationHistoryBuffer.initialize_from_scenario

​	`simulation_history_buffer.py` - observation_getter = scenario.get_past_tracked_objects/observation_getter = scenario.get_past_sensors - get_ego_past_trajectory (`abstract_scenario.py`)

observation中的可能是除模型外唯一需要的：`observation_builder.py` --嗯，是需要observation，

```
        model = LightningModuleWrapper.load_from_checkpoint(
            observation_cfg.checkpoint_path, model=torch_module_wrapper
        ).model
```



# 4 所需的类

## 1 SimulationIteration类

simulation中的步长和迭代（从 0 开始）

路径：`nuplan-devkit/nuplan/planning/simulation/simulation_time_controller/simulation_iteration.py`

```python
SimulationIteration(time_point=TimePoint(time_us=1630335846200414), index=0)
```

实例化时，需传入：

* `time_point`: `TimePoint`对象，表示simulation时的时间点
* `index`：`int`类型，表示simulation中的迭代，从 0 开始

属性：

* `time_us`
* `time_s`

## 2 PlannerInput类

输入给规划器，以计算轨迹

路径：`nuplan-devkit/nuplan/planning/simulation/planner/abstract_planner.py`

实例化时需传入：

* `iteration`: `SimulationIteration`对象，表示simulation进程中的迭代数和时间
* `history`: `SimulationHistoryBuffer`对象，滚动buffer，包含过去的观察结果observations和状态states
* `traffic_light_data`:` Optional[List[TrafficLightStatusData]]` None 

## 3 EgoState类

自动驾驶车辆（即ego车辆）的当前状态，包括其位置、朝向、速度、加速度以及其他相关动态属性

路径 `/nuplan-devkit/nuplan/common/actor_state/ego_state.py`

实例化时：

- `car_footprint`: `CarFootprint`对象，表示车辆的几何形状和朝向。
- `dynamic_car_state`: `DynamicCarState`对象，包含车辆的动态状态信息，如速度和加速度。
- `tire_steering_angle`: 车辆前轮的转向角度。
- `is_in_auto_mode`: 表示车辆是否处于自动驾驶模式。
- `time_point`: 时间戳，标识状态的时间点。

属性：

- `is_in_auto_mode`: 返回车辆是否处于自动驾驶模式。
- `car_footprint`: 获取车辆的几何足迹信息。
- `tire_steering_angle`: 获取车辆前轮的转向角度。
- `center`: 返回车辆的中心位置，使用`StateSE2`表示。
- `rear_axle`: 返回车辆后轴的位置和朝向，同样使用`StateSE2`表示。
- `time_point`, `time_us`, `time_seconds`: 提供对状态时间戳的访问。
- `dynamic_car_state`: 获取车辆的动态状态信息。
- `scene_object_metadata`: 返回与`EgoState`相关的场景对象元数据。
- `agent`: 将`EgoState`转换为`AgentState`对象，便于与环境中的其他代理进行交互。



## 4 LaneGraphEdgeMapObject类

还是不太理解是怎么记录的

1. **route_lane_dict: Dict[str, LaneGraphEdgeMapObject]**
   - 路线上的车道字典，键是车道的 ID，值是 `LaneGraphEdgeMapObject` 对象，表示车道的详细信息。



# 5 pdm_closed_planner

pdm_closed_planner需要：

```python
    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """
```

```python
@dataclass(frozen=True)
class PlannerInput:
    """
    Input to a planner for which a trajectory should be computed.
    """

    iteration: SimulationIteration  # Iteration and time in a simulation progress
    history: SimulationHistoryBuffer  # Rolling buffer containing past observations and states.
    traffic_light_data: Optional[List[TrafficLightStatusData]] = None  # The traffic light status data
```



# 6 待解决问题

确认tuplan使用的是什么坐标系、历史帧变换到当前帧，需要坐标转换。

如何设计实验来看从中间阶段是否ok？涉及到一帧一帧

InterpolatedTrajectory object
