# (c) Copyright 2023-2024 Zenseact AB
import logging
from rti.connextdds import DynamicData

from frameworks.rapid.swu.zendds_py import qos_config
from frameworks.rapid.swu.zendds_py.data_reader import DataReader
from frameworks.rapid.swu.zendds_py.data_writer import DataWriter
qos_config.init(qos_config.QosMode.Live)
class DomainKind:
    """DDS domain kind."""
    Standard = 0
    Scheduler = 2
    Logging = 4
    RapidPlatform = 5
class BaseReader:
    """Base DDS reader."""
    def __init__(
        self,
        qos_provider: qos_config.QosProfile,
        pd_type_name: str,
        topic_name: str,
        domain_id: int,
        logger_name: str,
        decoding_frequency: int,
    ):
        self._data_reader = DataReader.create(
            provider=qos_provider,
            msg_name=pd_type_name,
            topic_name=topic_name,
            profile=qos_config.QosProfile.PeriodicKeepLastSingle,
            domain_id=domain_id,
        )
        self._logger = logging.getLogger(logger_name)
        self._logger.info(
            f"Decoder created.\n\tTopic: {topic_name}\n\t"  # pylint: disable=no-member
            f"Reliability: {self._data_reader.qos.reliability.kind}\n\t"
            f"History: {self._data_reader.qos.history.kind}\n\t"
            f"Decoding frequency: {decoding_frequency}\n"
        )
    def read(self) -> DynamicData:
        data = None
        for sample in self._data_reader.take_data():
            # Return first valid sample
            try:
                data = DynamicData(sample)
            except Exception as e:  # pylint: disable=broad-except
                self._logger.exception(e)
            break
        return data
    def set_logging_level(self, logging_level):
        self._logger.setLevel(logging_level)


class BaseWriter:
    """Base DDS writer."""
    def __init__(
        self,
        qos_provider: qos_config.QosProfile,
        pd_type_name: str,
        topic_name: str,
        domain_id: int,
        logger_name: str,
    ):
        self._data_writer = DataWriter.create(
            provider=qos_provider,
            msg_name=pd_type_name,
            topic_name=topic_name,
            profile=qos_config.QosProfile.PeriodicKeepLastSingle,
            domain_id=domain_id,
        )
        self._logger = logging.getLogger(logger_name)
        self._logger.info(
            f"Writer created.\n\tTopic: {topic_name}\n\t"  # pylint: disable=no-member
            f"Reliability: {self._data_writer.qos.reliability.kind}\n\t"
            f"History: {self._data_writer.qos.history.kind}\n"
        )
    def init_dds_data(self) -> DynamicData:
        return DynamicData(self._data_writer.msg_type)
    def write(self, data: DynamicData):
        try:
            self._data_writer.write(data)
        except Exception as e:  # pylint: disable=broad-except
            self._logger.exception(e)
    def set_logging_level(self, logging_level):
        self._logger.setLevel(logging_level)