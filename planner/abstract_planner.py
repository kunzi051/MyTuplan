from simulation.simulation_history_buffer import SimulationHistoryBuffer

@dataclass(frozen=True)
class PlannerInput:
    """
    Input to a planner for which a trajectory should be computed.
    """

    # iteration: SimulationIteration  # Iteration and time in a simulation progress
    time_point: int # 微秒 since epoch
    index: int      # simulation中的迭代，从 0 开始
    history: SimulationHistoryBuffer  # Rolling buffer containing past observations and states.
    traffic_light_data: Optional[List[TrafficLightStatusData]] = None  # The traffic light status data
