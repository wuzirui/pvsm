import json, hashlib, math, subprocess
from pathlib import Path

CATEGORIES = [
    "Human-Shape",
    "Animals",
    "Daily-Used",
    "Furnitures",
    "Buildings_Outdoor",
    "Transportations",
    "Plants",
    "Food",
    "Electronics",
    "Poor-quality",
]

BASE_URL = "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_280k_split"

IN_DIR = Path("gobjaverse_280k_splits")  # download directory
OUT_DIR = Path("gobjaverse_280k_splits")  # output directory
IN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_SIZE = 15_229
TRAIN_SIZES = [2_000, 10_000, 50_000]  # full is written separately


def stable_key(s: str) -> float:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12)


def download_category_jsons():
    for cat in CATEGORIES:
        fn = f"gobjaverse_280k_{cat}.json"
        out = IN_DIR / fn
        if out.exists():
            continue
        url = f"{BASE_URL}/{fn}"
        print(f"Downloading {url}")
        subprocess.run(["wget", "-q", "-O", str(out), url], check=True)


def read_category_lists():
    cat2seqs = {}
    for cat in CATEGORIES:
        p = IN_DIR / f"gobjaverse_280k_{cat}.json"
        with p.open("r") as f:
            seqs = json.load(f)
        seqs = list(dict.fromkeys(seqs))  # remove duplicates
        cat2seqs[cat] = seqs
    return cat2seqs


def stratified_val(cat2seqs, val_size):
    total = sum(len(v) for v in cat2seqs.values())
    target_float = {c: val_size * (len(v) / total) for c, v in cat2seqs.items()}
    base_quota = {c: int(math.floor(x)) for c, x in target_float.items()}
    assigned = sum(base_quota.values())
    remainder = val_size - assigned

    frac = sorted(
        [(c, target_float[c] - base_quota[c]) for c in CATEGORIES],
        key=lambda x: (-x[1], x[0]),
    )
    for i in range(remainder):
        base_quota[frac[i][0]] += 1

    val, train_pool = [], []
    for c in CATEGORIES:
        arr = sorted(cat2seqs[c], key=stable_key)
        q = base_quota[c]
        val.extend(arr[:q])
        train_pool.extend(arr[q:])

    assert len(val) == val_size
    return val, train_pool


def write_json(path: Path, seqs):
    with path.open("w") as f:
        json.dump(seqs, f, indent=0)


def main():
    download_category_jsons()
    cat2seqs = read_category_lists()

    val, train_pool = stratified_val(cat2seqs, VAL_SIZE)
    train_pool_sorted = sorted(train_pool, key=stable_key)
    n_train = len(train_pool_sorted)

    # write val
    write_json(OUT_DIR / "val.json", val)

    # write train subsets
    for n in TRAIN_SIZES:
        n = min(n, n_train)
        tag = f"{n//1000:03d}k"
        write_json(OUT_DIR / f"train_{tag}.json", train_pool_sorted[:n])

    # full
    write_json(OUT_DIR / "train_full.json", train_pool_sorted)

    meta = {
        "total_sequences_reported": sum(len(v) for v in cat2seqs.values()),
        "val_size": len(val),
        "train_pool_size": n_train,
        "train_sizes": [min(n, n_train) for n in TRAIN_SIZES] + [n_train],
        "categories": {c: len(cat2seqs[c]) for c in CATEGORIES},
    }
    write_json(OUT_DIR / "meta.json", meta)


if __name__ == "__main__":
    main()
