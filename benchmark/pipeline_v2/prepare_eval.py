"""Build an evaluation question file for a folder of drawing images.

The images are matched to the gold questions by path, never by order:
  1. exact:      <images>/data/<source>/<file> == question "path"
  2. same_stem:  same folder and file name, other extension (README: "files with the same name and a
                 different extension are the same drawing in another format"), e.g. a swiss_dwellings
                 PNG answers the questions asked about its PDF.

Output (one line per question that has an image; all gold fields kept):
  path         the image file to run on (relative to the working directory)
  source_path  the path written in the gold file
  matched_by   exact | same_stem

    python prepare_eval.py
    python answer.py --questions work/eval_preview/questions.jsonl --no-llm --out work/eval_preview/out
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
PREVIEW = Path("data/junction-dataset-preview")


def load_questions(path: Path) -> list[dict]:
    items = []
    for n, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{n}: invalid JSON ({e})")
    return items


def match_images(
    image_root: Path, question_paths: set[str]
) -> dict[str, tuple[str, str]]:
    """Map gold question path -> (image path, matched_by). Fails on ambiguous matches."""
    by_stem = defaultdict(list)
    for q in question_paths:
        by_stem[Path(q).with_suffix("").as_posix()].append(q)

    matches: dict[str, tuple[str, str]] = {}
    for img in sorted(
        p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    ):
        rel = img.relative_to(image_root).as_posix()  # data/<source>/<file>
        if rel in question_paths:
            gold, how = rel, "exact"
        else:
            cands = by_stem.get(Path(rel).with_suffix("").as_posix(), [])
            if not cands:
                continue
            if len(cands) > 1:
                raise SystemExit(
                    f"ambiguous: image {rel} matches several gold paths {cands}"
                )
            gold, how = cands[0], "same_stem"
        if (
            gold in matches
        ):  # e.g. foo.png and foo.jpg both present: keep the exact one, never two
            prev = matches[gold]
            if prev[1] == "exact" or how != "exact":
                print(f"skip {rel}: {gold} already matched by {prev[0]}")
                continue
        matches[gold] = (img.as_posix(), how)
    return matches


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--images",
        default=str(PREVIEW / "junction-dataset-preview"),
        help="folder that contains data/<source>/<file>",
    )
    ap.add_argument("--questions", default=str(PREVIEW / "questions.jsonl"))
    ap.add_argument("--out", default="work/eval_preview/questions.jsonl")
    args = ap.parse_args()

    image_root, out = Path(args.images), Path(args.out)
    if not (image_root / "data").is_dir():
        raise SystemExit(f"{image_root} has no data/ folder")
    questions = load_questions(Path(args.questions))
    matches = match_images(image_root, {q["path"] for q in questions})

    rows = [
        {
            **q,
            "path": matches[q["path"]][0],
            "source_path": q["path"],
            "matched_by": matches[q["path"]][1],
        }
        for q in questions
        if q["path"] in matches
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_images = [p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    used = {m[0] for m in matches.values()}
    unused = sorted(
        p.relative_to(image_root).as_posix()
        for p in all_images
        if p.as_posix() not in used
    )
    print(
        f"{len(all_images)} images, {len(matches)} matched to gold "
        f"({dict(Counter(m[1] for m in matches.values()))}), {len(unused)} without questions"
    )
    print(f"{len(rows)} of {len(questions)} questions kept")
    print(
        "by source/type:",
        dict(Counter((Path(r["path"]).parts[-2], r["type"]) for r in rows)),
    )
    if unused:
        print(
            "images without questions:", dict(Counter(Path(u).parts[1] for u in unused))
        )
    (out.parent / "unmatched_images.txt").write_text(
        "\n".join(unused) + "\n", encoding="utf-8"
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
