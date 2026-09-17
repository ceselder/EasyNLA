"""Recipe config: YAML defaults + optional override YAML + `--set a.b=c` dotted overrides. Resolved config is saved per run."""
import copy, os, yaml

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "glp", "default_27b_l42.yaml")


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _coerce(s):
    if isinstance(s, str):
        if s.lower() in ("true", "false"): return s.lower() == "true"
        if s.lower() in ("inf", "infinity"): return float("inf")
        try: return int(s)
        except ValueError:
            try: return float(s)
            except ValueError: return s
    return s


def load_config(override_path=None, sets=()):
    cfg = yaml.safe_load(open(DEFAULT_PATH))
    if override_path:
        cfg = _merge(cfg, yaml.safe_load(open(override_path)) or {})
    for kv in sets:
        if not kv: continue
        k, v = kv.split("=", 1); node = cfg; parts = k.split(".")
        for p in parts[:-1]: node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(v)
    return _coerce_tree(cfg)   # PyYAML reads 2.0e9 / inf as strings (needs 2.0e+9); coerce every numeric-looking leaf


def _coerce_tree(x):
    if isinstance(x, dict): return {k: _coerce_tree(v) for k, v in x.items()}
    if isinstance(x, list): return [_coerce_tree(v) for v in x]
    return _coerce(x)


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
