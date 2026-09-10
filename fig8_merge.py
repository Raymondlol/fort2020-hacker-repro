"""Merge fig8_<tag>.json (parent + kernel) with fig8_<tag>_onset<step>.json files into one file."""
import json, sys
from pathlib import Path

def merge(res_dir, tag):
    res_dir = Path(res_dir)
    base = json.loads((res_dir / f"fig8_{tag}.json").read_text())
    parts = sorted(res_dir.glob(f"fig8_{tag}_onset*.json"), key=lambda p: int(p.stem.split("onset")[-1]))
    base["onsets"] = [json.loads(p.read_text())["onsets"][0] for p in parts]
    out = res_dir / f"fig8_{tag}_merged.json"
    out.write_text(json.dumps(base, indent=1))
    print(f"{out}: {len(base['onsets'])} onsets")
    return out

if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2])
