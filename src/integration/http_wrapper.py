"""HTTP client wrapper (Mode B) connecting DRDO TSS to the FastAPI inference service."""

import time
from typing import Any
import httpx

from src.api.action_codec import ActionCodec
from src.core.actions import AirAction, CommanderAction, GroundAction, SeaAction
from src.integration.config import HTTP_BASE_URL, HTTP_MAX_RETRIES, HTTP_TIMEOUT_S
from src.integration.tss_action_mapper import TSSActionMapper
from src.integration.tss_observation_mapper import TSSMappingError, TSSObservationMapper


class TSSConnectionError(Exception):
    """Raised when communication with the TSS FastAPI inference service fails after retries."""
    pass


class HTTPTSSWrapper:
    """Mode B: HTTP client wrapper exposing the same API as InProcessTSSWrapper.

    Connects across network sockets to decoupled C++/Java/C# simulation architectures.
    Implements robust connection pooling, automatic retries with exponential backoff,
    and external tracking of Commander recurrent hidden states.
    """

    def __init__(
        self,
        base_url: str = HTTP_BASE_URL,
        timeout_s: float = HTTP_TIMEOUT_S,
        max_retries: int = HTTP_MAX_RETRIES,
        protocol_path: str = "src/integration/protocol.yaml",
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.obs_mapper = TSSObservationMapper(protocol_path=protocol_path)
        self.action_mapper = TSSActionMapper(protocol_path=protocol_path)

        if client is not None:
            self.client = client
            self._owns_client = False
        else:
            self.client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_s,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
            self._owns_client = True

        # Per-agent recurrent memory storage for Commander policy
        self._hidden_states: dict[str, list[float]] = {}

    def _post_with_retry(self, endpoint: str, json_data: dict[str, Any]) -> dict[str, Any]:
        """Execute HTTP POST with exponential backoff on transient 5xx / timeout errors."""
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                resp = self.client.post(endpoint, json=json_data)

                # Client errors (4xx) are non-retryable
                if 400 <= resp.status_code < 500:
                    detail = resp.text
                    try:
                        detail = resp.json().get("detail", detail)
                    except Exception:
                        pass
                    raise TSSMappingError(
                        f"Inference API rejected request with HTTP {resp.status_code}: {detail}"
                    )

                # Server errors (5xx)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text}",
                        request=resp.request,
                        response=resp,
                    )

                return resp.json()

            except (httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                last_exception = exc
                if attempt < self.max_retries - 1:
                    sleep_time = 0.05 * (2**attempt)
                    time.sleep(sleep_time)

        raise TSSConnectionError(
            f"Failed communicating with inference API at '{self.base_url}{endpoint}' "
            f"after {self.max_retries} attempts. Cause: {str(last_exception)}"
        )

    def health(self) -> dict[str, Any]:
        """Query service health and available loaded policies."""
        try:
            resp = self.client.get("/health")
            if resp.status_code == 200:
                return resp.json()
            raise TSSConnectionError(f"Health check failed with HTTP {resp.status_code}")
        except Exception as exc:
            raise TSSConnectionError(f"Cannot reach inference service health endpoint: {str(exc)}")

    def reset(self, episode_id: str | None = None) -> None:
        """Clear recurrent hidden memory on remote server and local cache."""
        self._hidden_states.clear()
        try:
            self._post_with_retry("/reset", {"episode_id": episode_id})
        except Exception:
            # If server is temporarily unreachable during reset, local clear is preserved
            pass

    def get_action(
        self,
        agent_id: str,
        domain: str,
        variant: str = "default",
        tss_observation: dict[str, Any] | None = None,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Execute single-agent decision cycle via HTTP POST /act."""
        obs_dict = tss_observation or {}
        dom = domain.lower().strip()

        if dom == "commander":
            return self.get_commander_action(agent_id, obs_dict, deterministic=deterministic)

        # 1. Map observation to normalized vector
        if dom == "air":
            vec = self.obs_mapper.to_air_observation(obs_dict)
        elif dom == "ground":
            vec = self.obs_mapper.to_ground_observation(obs_dict)
        elif dom == "sea":
            vec = self.obs_mapper.to_sea_observation(obs_dict)
        else:
            raise ValueError(f"Unknown domain: '{domain}'")

        # 2. Remote HTTP call
        payload = {
            "agent_id": agent_id,
            "domain": dom,
            "variant": variant,
            "observation": vec.tolist(),
            "deterministic": deterministic,
        }
        resp_data = self._post_with_retry("/act", payload)

        # 3. Decode action and map to TSS command
        action = ActionCodec.decode(dom, variant, resp_data["action"])
        if isinstance(action, AirAction):
            return self.action_mapper.from_air_action(action, obs_dict)
        elif isinstance(action, GroundAction):
            return self.action_mapper.from_ground_action(action, obs_dict)
        elif isinstance(action, SeaAction):
            return self.action_mapper.from_sea_action(action, obs_dict)

        raise TypeError(f"Unexpected action type received for domain '{domain}': {type(action)}")

    def get_commander_action(
        self,
        agent_id: str,
        tss_observation: dict[str, Any],
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Execute Commander decision via HTTP POST /act with recurrent hidden state pass-through."""
        padded_obs = self.obs_mapper.to_commander_padded_vector(tss_observation)
        hidden = self._hidden_states.get(agent_id)

        payload = {
            "agent_id": agent_id,
            "domain": "commander",
            "variant": "default",
            "observation": padded_obs.tolist(),
            "hidden_state": hidden,
            "deterministic": deterministic,
        }
        resp_data = self._post_with_retry("/act", payload)

        # Update local recurrent state
        new_hidden = resp_data.get("hidden_state")
        if new_hidden:
            self._hidden_states[agent_id] = new_hidden

        cmd_action = CommanderAction(action=int(resp_data["action"][0]))
        return self.action_mapper.from_commander_action(cmd_action, tss_observation)

    def get_actions_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch execution of multiple vehicle requests via high-throughput HTTP POST /act/batch."""
        batch_items: list[dict[str, Any]] = []

        for req in requests:
            dom = req["domain"].lower().strip()
            obs_dict = req.get("tss_observation", {})
            if dom == "air":
                vec = self.obs_mapper.to_air_observation(obs_dict).tolist()
            elif dom == "ground":
                vec = self.obs_mapper.to_ground_observation(obs_dict).tolist()
            elif dom == "sea":
                vec = self.obs_mapper.to_sea_observation(obs_dict).tolist()
            elif dom == "commander":
                vec = self.obs_mapper.to_commander_padded_vector(obs_dict).tolist()
            else:
                raise ValueError(f"Unknown domain: '{dom}'")

            batch_items.append({
                "agent_id": req["agent_id"],
                "domain": dom,
                "variant": req.get("variant", "default"),
                "observation": vec,
                "deterministic": req.get("deterministic", False),
            })

        resp_data = self._post_with_retry("/act/batch", {"requests": batch_items})
        responses = resp_data["responses"]

        results: list[dict[str, Any]] = []
        for orig_req, act_resp in zip(requests, responses):
            dom = orig_req["domain"].lower().strip()
            var = orig_req.get("variant", "default")
            action = ActionCodec.decode(dom, var, act_resp["action"])

            obs_dict = orig_req.get("tss_observation", {})
            if isinstance(action, AirAction):
                results.append(self.action_mapper.from_air_action(action, obs_dict))
            elif isinstance(action, GroundAction):
                results.append(self.action_mapper.from_ground_action(action, obs_dict))
            elif isinstance(action, SeaAction):
                results.append(self.action_mapper.from_sea_action(action, obs_dict))
            elif isinstance(action, CommanderAction):
                results.append(self.action_mapper.from_commander_action(action, obs_dict))

        return results

    def shutdown(self) -> None:
        """Close persistent HTTP client connections."""
        if self._owns_client:
            self.client.close()
