## 0 ref&TODO

https://github.com/Tsinghua-MARS-Lab/StateTransformer

STR Visualization

https://arxiv.org/abs/2410.15774



https://nuplan-devkit.readthedocs.io/_/downloads/en/latest/pdf/

https://www.shenlanxueyuan.com/open/course/214/lesson/196/liveToVideoPreview

## 1 nuplan 

### 1 introduction

nuPlan is the world’s first closed-loop ML-based planning benchmark for autonomous driving.

It provides a high quality dataset with 1500h of human driving data from 4 cities across the US and Asia with widely varying traffic patterns (Boston, Pittsburgh, Las Vegas and Singapore). In addition, it provides a closed-loop simulation framework with reactive agents, a training platform as well as a large set of both general and scenario-specific planning metrics.

Train ML planner --> Simulate planner&agents --> Measure performance --> Compare&visualize planners

### 2 Training & simulation framework

![image-20241101171403373](https://raw.githubusercontent.com/kunzi051/BlogImage/main/Project/202411011714506.png)

framework.png

### 3 Scenarios & database

both in expert imitation (open-loop) and reactive planning (closed-loop)

example scenarios:

| Unprotected cross turn | Dense vehicle interactions | Jaywalker in front    |
| ---------------------- | -------------------------- | --------------------- |
| Lane change            | Ego at pickup/dropoff area | Ego following vehicle |

| Database                       | Size  | Duration | Num Logs | Cities                                   | Num Scenarios | Sensor Data | Description                                     |
| :----------------------------- | :---- | :------- | :------- | :--------------------------------------- | :------------ | :---------- | :---------------------------------------------- |
| nuplan_v1.1_mini (recommended) | 13GB  | 7h       | 64       | Las Vegas, Boston, Pittsburgh, Singapore | 67            | N/A         | The mini split used for prototyping and testing |
| nuplan_v1.1                    | 1.8TB | 1282h    | 15910    | Las Vegas, Boston, Pittsburgh, Singapore | 73            | N/A         | The full dataset for training and evaluation    |



## 2 Planner

ref: `nuplan_planner_tutorial.ipynb`

### 1 planner

构建和训练自己的规划器，在nuPlan的pipline中运行以评估

接入nuPlan接口的基本元素、可视化、评估指标



planner:  responsible for determining the **ego** vehicle's behavior

planner输入：自身姿态ego pose、其他代理姿态、静态和动态地图信息、目标

planner输出：产生轨迹

![image-20241101190530716](https://raw.githubusercontent.com/kunzi051/BlogImage/main/Project/202411011905780.png)

继承`AbstractPlanner`类

```python
initialize: 静态信息初始化，姿态表示为(x,y,heading)
name: name of planner
observation_type: Options here include Sensors (raw sensor information such as images or pointclouds) and DetectionsTracks (outputs of an earlier perception system designed to consume sensor information and produce meaningful detections对环境中的对象（如车辆、行人、骑行者等）的识别和跟踪，以及它们的属性（如位置、速度、加速度等）)
compute_planner_trajectory: responsible for producing the trajectory dictating the path the ego vehicle will attempt to follow in the future
```

### 2 Code-SimplePlanner

`class SimplePlanner(AbstractPlanner)`：预测时间、采样时间，加速度、最大速度、转向角，车辆参数、动力学模型，自车状态

```python
# doing: just drive straight according to a specified steering angle
class SimplePlanner(AbstractPlanner):
    """
    Planner going straight
    """

    def __init__(self,
                 horizon_seconds: float,
                 sampling_time: float,
                 acceleration: npt.NDArray[np.float32],
                 max_velocity: float = 5.0,
                 steering_angle: float = 0.0):
        self.horizon_seconds = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.acceleration = StateVector2D(acceleration[0], acceleration[1])
        self.max_velocity = max_velocity
        self.steering_angle = steering_angle
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        """ Inherited, see superclass. """
        pass

    def name(self) -> str:
        """ Inherited, see superclass. """
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """ Inherited, see superclass. """
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> List[AbstractTrajectory]:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """
        # Extract iteration and history
        iteration = current_input.iteration
        history = current_input.history

        ego_state = history.ego_states[-1]
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                self.acceleration,
            ),
            tire_steering_angle=self.steering_angle,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        for _ in np.arange(
            iteration.time_us + self.sampling_time.time_us,
            iteration.time_us + self.horizon_seconds.time_us,
            self.sampling_time.time_us,
        ):
            if state.dynamic_car_state.speed > self.max_velocity:
                accel = self.max_velocity - state.dynamic_car_state.speed
                state = EgoState.build_from_rear_axle(
                    rear_axle_pose=state.rear_axle,
                    rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
                    rear_axle_acceleration_2d=StateVector2D(accel, 0),
                    tire_steering_angle=state.tire_steering_angle,
                    time_point=state.time_point,
                    vehicle_parameters=state.car_footprint.vehicle_parameters,
                    is_in_auto_mode=True,
                    angular_vel=state.dynamic_car_state.angular_velocity,
                    angular_accel=state.dynamic_car_state.angular_acceleration,
                )

            state = self.motion_model.propagate_state(state, state.dynamic_car_state, self.sampling_time)
            trajectory.append(state)

        return InterpolatedTrajectory(trajectory)
```



### 3 Open- & Closed loop

Simulating the planner

* Open-loop: ego  vehicle log replay & other agents log replay, to imitate the expert driver's behavior, provide a high-level performance overview

* Closed-loop
  * ego closed-loop simulation with agents replayed from log (open-loop, non reactive)
  
  * ego closed-loop simulation with agents controlled by a rule-based or learned policy (closed-loop, reactive)
  
    ps：在tuplan中，reactive other agents使用的IDM



### 4 Simulation parameters

| Options | **Ego controller**          | Observation            |
| ------- | --------------------------- | ---------------------- |
|         | Log play back controller    | Box observation        |
|         | Perfect tracking controller | IDM agents observation |
|         |                             | Lidar pc observation   |





## 3 nuBoard visualized

nuBoard repository path: 

```
E:\Pycharm\nuplan-devkit\nuplan\planning\nuboard
```



```
python E:/Pycharm/nuplan-devkit/nuplan/planning/script/run_nuboard.py
```

报错没有`hydra`





## 4 Compare

### 1 IDM

![image-20241104143959596](https://raw.githubusercontent.com/kunzi051/BlogImage/main/Project/202411041441221.png)



由常微分方程组控制的二阶汽车跟随模型*car-following model*.
$$
a(t) = a_{\max} \left[ 1 - \left( \frac{v(t)}{v_{\max}} \right)^4 - \left( \frac{s^*(v(t), \Delta v(t))}{s_0} \right)^2 \right] 
$$

$$
s^*(v, \Delta v) = s_0 + v \cdot T + \frac{v \cdot \Delta v}{2 \sqrt{a_{\max} \cdot b}}
$$

- 影响因素（模型输入）是自身的速度 *v*、与前车车距 *s* 以及两辆车的相对速度（速度差）*Δ v = v-v_领先*。
- 模型输出是驾驶员在这种情况下选择的加速度 *dv/dt*。
- 模型*参数*描述了驾驶风格，即模拟驾驶员是慢速驾驶还是快速驾驶，小心驾驶还是鲁莽驾驶
- 优点：IDM模型的参数数量少、意义明确，并且能用统一的模型描述从自由流到完全拥堵流的不同状态。
- 缺点：缺乏随机项，也就是输入一定时，输出是确定的，这与现实中车辆行为的随机性有所差异。 例子：在交通流模拟中我们可以观察到相同参数的两辆车从路口停止线前同时起步后并行向前行驶，并在较长时间内保持同样的行驶状态，与实际车辆驾驶行为不符。



ref

https://traffic-simulation.de/info/info_IDM.html

https://blog.csdn.net/m0_46499664/article/details/130731097

https://zhuanlan.zhihu.com/p/412617108



## 5 nuPlan Planning Challenge

ref `competition.md`

### 1 Common rules

- The current and past (up to 2s) scene information for each simulation iteration will be
  passed to the planner - that includes ego pose, ego route, agents, and static/dynamic map
  information.
- Each planner will have a fixed time budget of 1s for each simulation iteration, after
  which the simulation will time out.
- The simulation horizon will be up to 15s for each scenario and the simulation frequency
  will be 10Hz.

Planner Output Requirements

| Planner Trajectory Requirements | Value                     |
| ------------------------------- | ------------------------- |
| Expected minimum horizon length | 8s                        |
| Expected minimum horizon steps  | 2                         |
| Expected signals                | x, y, heading, time stamp |

| Common Challenge Configuration | Value |
| ------------------------------ | ----- |
| Frequency                      | 10Hz  |
| Rollout horizon                | 15s   |
| Past observation horizon       | 2s    |
### 2 Challenge 1: Open-loop

| Configuration              | Value                        |
| -------------------------- |----------------------------- |
| Metric computation horizon | 3s, 5s, 8s @ 1Hz sampling    |
| Controller                 | N/A                          |
| Observations               |  <ul><li>Vehicles<li>Pedestrians<li>Cyclists<li>Generic Objects<li>Traffic cones<li>Barriers<li>Construction Zone Signs</li></ul>                          |

Scoring

The overall score of challenge 1 considers the following factors:

*   Average Displacement Error (ADE)
*   Final Displacement Error (FDE)
*   Average Heading Error (AHE)
*   Final Heading Error (FHE)
*   Miss Rate

The details of each metric can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/metrics_description.html#).
The details of the scoring hierarchy can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/nuplan_metrics_aggeregation.html)

### 3 Challenge 2: Closed-loop non-reactive agents

| Configuration | Value |
| ------------- | ----- |
| Controller    | [LQR](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/controller/tracker/lqr.py)  |
| Observations  | <ul><li>Vehicles<li>Pedestrians<li>Cyclists<li>Generic Objects<li>Traffic cone<li>Barriers<li>Construction Zones Signs</li></ul>                          |

Scoring

The overall score of challenge 2 considers the following factors:

*   At-fault collision
*   Drivable area compliance
*   Driving direction compliance
*   Making progress
*   Time to collision (TTC)
*   Speed limit compliance
*   Ego progress along the expert's route ratio
*   Comfort

The details of each metric can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/metrics_description.html).
The details of the scoring hierarchy can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/nuplan_metrics_aggeregation.html)

### 4 Challenge 3: Closed-loop reactive agents

The policy determining the behavior of agents is an Intelligent Driver Model ([IDM](https://en.wikipedia.org/wiki/Intelligent_driver_model)) policy. The rest of the observations are still replayed in open-loop.

| Configuration | Value |
| ------------- | ----- |
| Controller    | [LQR](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/controller/tracker/lqr.py) |
| Observations  | <ul><li>Reactive:<ul><li>Vehicles</li></ul><li>Open-loop<ul><li>Pedestrians<li>Generic Objects<li>Traffic cone<li>Barriers<li>Construction Zones Signs</li></ul> </li></ul>|

Scoring: Same as Challenge2

### 5 Scenario Selection

All three challenges will be run on the following scenario types.

| 场景类型                                               | 描述                                                         |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| starting_straight_traffic_light_intersection_traversal | Ego在交通灯控制的交叉口区域开始直行穿越，而没有停止          |
| high_lateral_acceleration                              | Ego在不转弯的情况下，横向轴上高加速度（1.5 < 加速度 < 3 m/s^2）和高偏航率 |
| changing_lane                                          | Ego在开始向相邻车道变道                                      |
| high_magnitude_speed                                   | Ego高速度幅度（速度 > 9 m/s）和低加速度（速度 > 9 m/s）      |
| low_magnitude_speed                                    | Ego低速度幅度（0.3 < 速度 < 1.2 m/s）和低加速度，而没有停止  |
| starting_left_turn                                     | Ego在交叉口区域开始向左转弯，而没有停止                      |
| starting_right_turn                                    | Ego在交叉口区域开始向右转弯，而没有停止                      |
| stopping_with_lead                                     | Ego开始减速（加速度幅度 < -0.6 m/s^2, 速度幅度 < 0.3 m/s）并且前方有领先车辆（距离 < 6 m）在任何区域 |
| following_lane_with_lead                               | Ego以（速度 > 3.5 m/s）跟随其当前车道，并在同一车道上有移动的领先车辆（速度 > 3.5 m/s, 纵向距离 < 7.5 m） |
| near_multiple_vehicles                                 | Ego附近（距离 < 8 m）有多辆（>6）移动车辆，而Ego正在移动（速度 > 6 m/s） |
| traversing_pickup_dropoff                              | Ego在不停止的情况下穿越接送/下车区域                         |
| behind_long_vehicle                                    | Ego在同一车道上跟随（纵向距离 3 m < < 10 m）较长（长度 > 8 m）的车辆（横向距离 < 0.5 m） |
| waiting_for_pedestrian_to_cross                        | Ego等待附近（距离 < 8 m, 时间到交叉口 < 1.5 m）的行人穿越人行横道区域，而Ego没有停止，行人不在接送/下车区域 |
| stationary_in_traffic                                  | Ego与多辆（>6）车辆静止（距离 < 8 m）在一起                  |



## 6 nuPlan baselines

ref `baselines.md`

| Name                                 | 说明                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| SimplePlanner                        | 以恒定速度规划一条直线路径。这个规划器的唯一逻辑是，如果当前速度超过了`max_velocity`，则会减速 |
| IDMPlanner                           | 路径规划（广度优先搜索找到通往任务目标的路径）+纵向控制（IDM策略：沿着这条路径以多快的速度行驶） |
| UrbanDriverOpenLoopModel (MLPlanner) | 将矢量化的代理和地图输入处理成局部特征描述符，这些描述符被传递给全局注意力机制，以产生预测的自我轨迹。使用模仿学习训练 |



