import abc
from typing import Any, Dict, Optional


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict, *, noise: Optional[Any] = None) -> Dict:
        """Infer actions from observations.

        `noise` lets the CALLER own the sampler's randomness. Left None, the server falls back to
        its own per-process key stream, which advances once per request and therefore makes a
        result depend on how many requests preceded it. Callers that need a rollout to be
        reproducible in isolation -- or to survive being served by a reused server, a different
        worker, or a retry -- must supply it.
        """

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
