# (c) Copyright 2023-2024 Zenseact AB
from .base import BaseWriter, DomainKind
from zenuity_internal_if.interfaces.zendds.route_py_generated.zendds.autogen.inter
faces_pi import (
    route as route_zendds,
 )
from zenuity_internal_if.zif.interfaces.ctypes import route as pi_route

class RouteDataWriter(BaseWriter):
    """Route data writer."""
    def __init__(self):
        # The default domain kind of dds reader in rapid navi agent is Standard
        domain_id = DomainKind.Standard
        super().__init__(
            qos_provider=route_zendds.get_qos_provider(),
            pd_type_name="PD_RouteData",
            topic_name="navigation_route",
            domain_id=domain_id,
            logger_name=self.__class__.__name__,
        )
    def set_route(self, navi_path, is_matched_to_hd_map):
        data = self.init_dds_data()
        data_segment_info = data.get_values("route.segments_info")
        data_points = data.get_values("route.points")
        segment_idx = 0
        point_idx = 0
        for segment in navi_path:
            if segment_idx >= pi_route.PI_MAXNUMBEROFSEGMENTS:
                self._logger.debug("Reach max number of segments")
                break
            point_idx_offset = point_idx
            number_of_points = len(segment)
            if number_of_points > pi_route.PI_MAXNUMBEROFROUTEPOINTSPERSEGMENT:
                self._logger.debug("Reach max number of route points per segment")
                number_of_points = pi_route.PI_MAXNUMBEROFROUTEPOINTSPERSEGMENT
            if (point_idx_offset + number_of_points) >= pi_route.PI_MAXNUMBEROFROUTEPOINTS:
                self._logger.debug("Reach max number of points")
                break
            for point in segment:
                data_points[point_idx].set_value("is_valid.unitless.value", True)
                data_points[point_idx].set_value(
                    "lat_position.nanodegrees.value", int(point.lat * 1e9)
                )
                data_points[point_idx].set_value(
                    "lon_position.nanodegrees.value", int(point.lon * 1e9)
                )
                point_idx += 1
            data_segment_info[segment_idx].set_value(
                "point_index_offset.unitless.value", point_idx_offset
            )
            data_segment_info[segment_idx].set_value(
                "number_of_points.unitless.value", point_idx - point_idx_offset
            )
            segment_idx += 1
            data.set_value("route.is_valid.unitless.value", True)
            data.set_value("route.is_matched_to_hd_map.unitless.value", is_matched_to_hd_map)
            data.set_value(
                "route.source_of_mpp_route",
                pi_route.PI_SourceOfMppRoute_NavigationRouteDestinationSet,
            )
        data.set_value("route.number_of_segments.unitless.value", segment_idx)
        data.set_values("route.segments_info", data_segment_info)
        data.set_values("route.points", data_points)
        self.write(data)