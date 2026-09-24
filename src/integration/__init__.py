"""TSS Integration Layer connecting DRDO simulation software with multi-domain RL policies."""

from src.integration.domain_randomization import DomainRandomizer
from src.integration.http_wrapper import HTTPTSSWrapper, TSSConnectionError
from src.integration.in_process_wrapper import InProcessTSSWrapper
from src.integration.parallel_env import TSSParallelEnv
from src.integration.protocol_docs import generate_integration_md
from src.integration.single_agent_env import TSSSingleAgentEnv
from src.integration.tss_action_mapper import TSSActionMapper
from src.integration.tss_observation_mapper import TSSMappingError, TSSObservationMapper

__all__ = [
    "InProcessTSSWrapper",
    "HTTPTSSWrapper",
    "TSSParallelEnv",
    "TSSSingleAgentEnv",
    "DomainRandomizer",
    "TSSObservationMapper",
    "TSSActionMapper",
    "TSSMappingError",
    "TSSConnectionError",
    "generate_integration_md",
]
