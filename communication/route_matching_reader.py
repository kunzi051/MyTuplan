# (c) Copyright 2023-2024 Zenseact AB
from .base import BaseReader, DomainKind
from zenuity_internal_if.interfaces.zendds.route_access_data_py_generated.zendds.a
utogen.interfaces_pi import (
    route_access_data as route_access_data_zendds,
)


def _get_coordinate_from_geo_id(geo_id):
    '''
    从一个地理 ID (geo_id) 中提取纬度和经度坐标
    输入
        geo_id 一个 64 位整数，其中高 32 位表示纬度，低 32 位表示经度
    输出
        返回一个元组，包含两个浮点数：(lat, lgt)，分别表示纬度和经度

    '''
    lsb32 = 0x00000000FFFFFFFF
    degree_factor = 1e7
    lat = float((geo_id >> 32) & lsb32) / degree_factor
    lgt = float(geo_id & lsb32) / degree_factor
    return lat, lgt


class RouteMatchingDataReader(BaseReader):
    """Route matching data reader."""
    def __init__(self):
        domain_id = DomainKind.Logging
        self.decoding_frequency = 10  # Hz

        super().__init__(
            qos_provider=route_access_data_zendds.get_qos_provider(), # 来源
            pd_type_name="PD_RouteHdSdMapMatchingDebugData",
            topic_name="zen_qm_sensorfusion_a_route_map_matching_debug_data",
            domain_id=domain_id,
            logger_name=self.__class__.__name__,
            decoding_frequency=self.decoding_frequency,
        )
        self.prev_horizon_start_geo_id = 0
        self.prev_horizon_end_geo_id = 0 # 上一次处理的路线段的起始和结束坐标

    def get_segments_coordinate(self, route_type: str):
        # 从数据源中获得当前路线类型下的匹配段数量，获取路线段的起始和结束坐标
        route_type = route_type.lower()
        updated = True
        segments_coordinate = []
        data = self.read()

        if data is not None and route_type in ("hd", "sd"):
            nof_segments = data.get_value(f"{route_type}.number_of_matched_objects.unitless.value")
            self._logger.debug(f"Number of matched segments: {nof_segments}")
            if nof_segments == 0:
                self.prev_horizon_start_geo_id = 0
                self.prev_horizon_end_geo_id = 0
            else:
                matched_object_ids = data.get_values(f"{route_type}.matched_object_ids")
                horizon_start_geo_id = matched_object_ids[0].get_value("part1.unitless.value")
                horizon_end_geo_id = matched_object_ids[nof_segments - 1].get_value(
                    "part2.unitless.value"
                )
                updated = (
                    horizon_start_geo_id != self.prev_horizon_start_geo_id
                    or horizon_end_geo_id != self.prev_horizon_end_geo_id
                )
                if updated:
                    self.prev_horizon_start_geo_id = horizon_start_geo_id
                    self.prev_horizon_end_geo_id = horizon_end_geo_id
                    for object_id in matched_object_ids[:nof_segments]:
                        start_lat, start_lon = _get_coordinate_from_geo_id(
                            object_id.get_value("part1.unitless.value"),
                        )
                        end_lat, end_lon = _get_coordinate_from_geo_id(
                            object_id.get_value("part2.unitless.value"),
                        )
                        segments_coordinate.append(
                            {
                                "start_lat": start_lat,
                                "start_lon": start_lon,
                                "end_lat": end_lat,
                                "end_lon": end_lon,
                            }
                        )
        return updated, segments_coordinate