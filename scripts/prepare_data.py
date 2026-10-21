"""Build the processed XYZ benchmark from the public TrajAir and TartanAviation (KBTP) files and check it
byte for byte against the build used in the paper (pa_tsmixer/data_manifest.json).

    python scripts/prepare_data.py --trajair <TrajAir 111_days folder> --tartan-zip <kbtp/processed.zip> --output data
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pa_tsmixer import data

MANIFEST = json.loads((Path(data.__file__).with_name("data_manifest.json")).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajair", type=Path, help="folder with the TrajAir 111-day processed .txt files")
    p.add_argument("--tartan-zip", type=Path, help="TartanAviation KBTP processed.zip")
    p.add_argument("--output", type=Path, default=Path("data"))
    a = p.parse_args()
    ok = True
    for name, source in (("trajair", a.trajair), ("tartanaviation", a.tartan_zip)):
        if source is None:
            continue
        target = a.output / name
        print(f"building {name} from {source} ...", flush=True)
        summary = data.build(name, source, target)
        expected = MANIFEST[name]
        print(f"  points {summary['points']:,}, segments {summary['segments']:,}, windows {summary['windows']}")
        for artifact, digest in expected["artifact_sha256"].items():
            match = data.sha256(target / artifact) == digest
            ok &= match
            print(f"  {artifact:14s} {'identical to the paper build' if match else 'DIFFERS from the paper build'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
