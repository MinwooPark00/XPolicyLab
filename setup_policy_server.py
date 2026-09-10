import asyncio
import os
import threading
import ast
import time
import yaml
import importlib
import argparse
import traceback
from client_server.tcp.model_server import ModelServer


_SEEDING = False   # set by seed_server() before main()


def _default_protocol() -> str:
    """Default to the websocket policy protocol."""
    return "ws"

def eval_function_decorator(policy_model_name, Func_and_Class_name):
    """Load a specified function (e.g., get_model) from a policy module"""
    module = importlib.import_module(policy_model_name)
    return getattr(module, Func_and_Class_name)

def main(deploy_cfg):
    """Main entry: load model, start server, run indefinitely"""
    # Extract basic arguments
    policy_name = deploy_cfg.get("policy_name")
    port = deploy_cfg.get("port")
    host = deploy_cfg.get("host", "0.0.0.0")
    protocol = deploy_cfg.get("protocol", "ws")

    # Instantiate model
    model_class_func = eval_function_decorator(f"XPolicyLab.policy.{policy_name}.model", "Model")
    model = model_class_func(deploy_cfg)
    attach_episode_seeding(model, _SEEDING)

    if protocol == "ws":
        try:
            from client_server.ws.model_server import PolicyServer, PolicyServerConfig
        except ModuleNotFoundError as exc:
            if exc.name == "client_server":
                # client_server.ws ships in this repo; make it importable even when
                # XPolicyLab is not pip-installed in the current environment.
                import sys
                repo_root = os.path.dirname(os.path.abspath(__file__))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                try:
                    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
                except ModuleNotFoundError as dep_exc:
                    raise RuntimeError(
                        "ws policy server requires XPolicyLab websocket dependencies "
                        f"(missing module: {dep_exc.name}). Install in the policy env with: "
                        "pip install -e . from the XPolicyLab root."
                    ) from dep_exc
            else:
                raise RuntimeError(
                    "ws policy server requires XPolicyLab websocket dependencies "
                    f"(missing module: {exc.name}). Install in the policy env with: "
                    "pip install -e . from the XPolicyLab root."
                ) from exc

        server = PolicyServer(
            model,
            PolicyServerConfig(
                host=host,
                port=int(port),
                ws_ping_interval_s=deploy_cfg.get("ws_ping_interval_s", 20.0),
                ws_ping_timeout_s=deploy_cfg.get("ws_ping_timeout_s", 20.0),
            ),
        )
        try:
            asyncio.run(server.serve_forever())
        except KeyboardInterrupt:
            print("\nShutting down websocket policy server...")
        return
    if protocol != "legacy_tcp":
        raise ValueError(f"unsupported policy server protocol: {protocol}")

    # Wrap server.start so exceptions inside thread are fully printed
    def run_server():
        try:
            server.start()
        except Exception:
            print("\033[31m[ERROR] Exception occurred inside server thread:\033[0m")
            traceback.print_exc()
            raise

    # Start server in background thread
    server = ModelServer(model, host=host, port=port)
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Keep main thread alive until KeyboardInterrupt
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.stop()
        thread.join()

def parse_args_and_config():
    """Parse CLI args and YAML config, merge overrides"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "--config-path", dest="config_path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--protocol", choices=("legacy_tcp", "ws"), help="Policy server protocol")
    parser.add_argument("--host", help="Policy server bind host")
    parser.add_argument("--port", type=int, help="Policy server bind port")
    parser.add_argument("--relay-url", dest="relay_url", help="Relay URL for future relay mode")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, help="Override config values")
    args = parser.parse_args()

    # Load base config
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Parse overrides: --key value pairs

    def _parse_val(s: str):
        # safer than eval; supports numbers/bool/None/list/dict when properly quoted
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
        
    if args.overrides:
        tokens = args.overrides

        # Case A: key=value key=value ...
        if all(("=" in t and not t.startswith("-")) for t in tokens):
            for t in tokens:
                k, v = t.split("=", 1)
                cfg[k] = _parse_val(v)
        else:
            # Case B: --key value --key value ...
            if len(tokens) % 2 != 0:
                raise ValueError(f"--overrides expects key value pairs, got: {tokens}")

            it = iter(tokens)
            for key in it:
                val = next(it)
                cfg[key.lstrip("-")] = _parse_val(val)

    if args.protocol is not None:
        cfg["protocol"] = args.protocol
    else:
        cfg.setdefault("protocol", _default_protocol())
    if args.host is not None:
        cfg["host"] = args.host
    if args.port is not None:
        cfg["port"] = args.port
    if args.relay_url is not None:
        cfg["relay_url"] = args.relay_url

    def _require_non_empty(key: str):
        if key not in cfg:
            raise ValueError(f"{key} must be specified in config or overrides")
        val = cfg[key]
        if val is None:
            raise ValueError(f"{key} must be non-empty")
        if isinstance(val, str) and not val.strip():
            raise ValueError(f"{key} must be non-empty")

    _require_non_empty("host")
    _require_non_empty("port")
    return cfg


def _seed_server_enabled() -> bool:
    """`XPOLICYLAB_SEED_SERVER`: on unless set to 0/false/no/off.

    The eval CLI's `--seed` seeds the *client* -- the scene randomization and
    the client's own `random`/`numpy`/`torch` -- while the sampling that turns
    an observation into an action happens over here: Diffusion Policy draws its
    initial trajectory from the global generator (`conditional_sample`,
    generator=None), the flow-matching baselines draw their noise the same way,
    and pi0.5 walks one JAX key from `key(0)`. Left unseeded, two evaluations of
    one checkpoint on one seed do not produce the same actions (MHBench's
    docs/eval.md 6 has the measurement). So the server seeds itself at start
    and again at every `seed_episode` the client sends, which makes an
    episode's noise a function of its seed alone -- not of the shard it ran
    in, the episodes before it, or a crash in between. Opt out with
    `XPOLICYLAB_SEED_SERVER=0` to get the pre-2026-09-09 behaviour back.
    """
    value = os.environ.get("XPOLICYLAB_SEED_SERVER", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def seed_rngs(seed: int) -> None:
    """Seed python, numpy and torch (CPU and every CUDA device) with `seed`."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seed_server(deploy_cfg) -> bool:
    """Seed this process's RNGs once at start, and pin cuDNN to one algorithm.

    `eval_seed` is the evaluation's own seed when the runner passes one; `seed`
    is otherwise the checkpoint's training seed (it names the run directory,
    utils/checkpoint_resolver.py), which is fine as a fallback but not what an
    evaluation means by its seed. Returns whether seeding is on.
    """
    if not _seed_server_enabled():
        print("[server] RNGs not seeded (XPOLICYLAB_SEED_SERVER=0)")
        return False
    seed = deploy_cfg.get("eval_seed", deploy_cfg.get("seed"))
    if seed is None:
        print("[server] no seed/eval_seed in the config -- seeding from 0")
        seed = 0
    seed = int(seed)
    seed_rngs(seed)
    try:
        import torch

        # Autotuned cuDNN picks a different convolution algorithm from one
        # process to the next; pinned, two servers given the same inputs and
        # the same seed run the same arithmetic.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    print(f"[server] RNGs seeded with {seed}; cudnn.benchmark=False, cudnn.deterministic=True (XPOLICYLAB_SEED_SERVER)")
    return True


def attach_episode_seeding(model, enabled: bool):
    """Give the model a `seed_episode(obs)` RPC the eval client calls per episode.

    Attached here rather than written into every adapter: the ws server
    dispatches any public method name to the model (`_handle_call`), so one
    bound function covers all of them. The payload is `{"seed": int}`. When
    seeding is on it reseeds python/numpy/torch and, if the adapter defines
    `seed(int)`, hands the seed to it too -- Pi_05 restarts its JAX key there,
    since torch's generators cannot reach JAX. The reply says whether anything
    was seeded, and the client records that per episode.
    """
    def seed_episode(obs=None):
        if not isinstance(obs, dict) or "seed" not in obs:
            raise ValueError("seed_episode expects {'seed': int}")
        seed = int(obs["seed"])
        if not enabled:
            return {"seeded": False, "seed": seed}
        seed_rngs(seed)
        hook = getattr(model, "seed", None)
        if callable(hook):
            hook(seed)
        return {"seeded": True, "seed": seed}

    model.seed_episode = seed_episode
    return model


if __name__ == "__main__":
    deploy_cfg = parse_args_and_config()
    _SEEDING = seed_server(deploy_cfg)
    main(deploy_cfg)
