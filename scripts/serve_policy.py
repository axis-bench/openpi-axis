import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import axis_config as _axis_config
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Must stay False: the quality tag reaches the model through --quality-tag, and a launch line
    # that sets this flag is refused at start.
    cfg_conditioning: bool = False

    # Classifier-free guidance scale `w`: v = v_uncond + (1 + w)(v_cond - v_uncond), so beta = 1 + w
    # (a beta = 1.3 cell passes 0.3). Inference-only; 0 is plain conditional sampling. The server
    # publishes `beta` in its metadata so a client can check it.
    guidance_scale: float = 0.0

    # Quality tag appended to every prompt ("\nQuality: {q}"); the unconditional branch of a guided
    # policy is the bare prompt. A flag, not an environment variable, so it is in the launch line.
    quality_tag: int | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            if args.cfg_conditioning:
                raise ValueError("--cfg-conditioning is not served by this line; use --quality-tag")
            config = (
                _axis_config.load_train_config(args.policy.config)
                if args.policy.config.endswith(".json")
                else _config.get_config(args.policy.config)
            )
            return _policy_config.create_trained_policy(
                config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                guidance_scale=args.guidance_scale,
                quality_tag=args.quality_tag,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    # Publish the sampler contract so a client can verify it: a client that sends `__noise__` to a
    # server that ignores it would run under the server's own key stream without knowing.
    policy_metadata = {
        **policy.metadata,
        "supports_client_noise": True,
        "action_horizon": int(policy._model.action_horizon),  # noqa: SLF001
        "action_dim": int(policy._model.action_dim),  # noqa: SLF001
    }

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
