"""Render a local-only human review UI for an ABO image packet."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import stat

from skillchain.data.abo import ABOProvenanceError
from skillchain.data.abo_archive_review import verify_abo_archive_review_packet
from skillchain.data.abo_original_review import verify_abo_original_review_packet
from skillchain.data.abo_review_replenishment import (
    verify_abo_review_decision_carry_forward,
)
from skillchain.synthesis.store import atomic_create_file, sha256_bytes
from skillchain.tools.serialization import parse_canonical_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--carry-forward-root", type=Path)
    parser.add_argument("--carry-forward-manifest-sha256")
    parser.add_argument("--source-packet-root", type=Path)
    parser.add_argument("--source-packet-manifest-sha256")
    parser.add_argument("--source-human-review-root", type=Path)
    parser.add_argument("--source-human-review-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    carry_values = (
        arguments.carry_forward_root,
        arguments.carry_forward_manifest_sha256,
        arguments.source_packet_root,
        arguments.source_packet_manifest_sha256,
        arguments.source_human_review_root,
        arguments.source_human_review_manifest_sha256,
    )
    if any(value is not None for value in carry_values) and not all(
        value is not None for value in carry_values
    ):
        parser.error(
            "carry-forward rendering requires the carry bundle, source packet, "
            "source human-review roots, and all three external manifest digests"
        )

    try:
        manifest = verify_abo_original_review_packet(
            arguments.packet_root,
            expected_manifest_sha256=arguments.manifest_sha256,
        )
        source_label = "官方 original 对象（完整原图，页面按比例缩放且不裁剪）"
    except ABOProvenanceError:
        if arguments.carry_forward_root is not None:
            raise
        manifest = verify_abo_archive_review_packet(
            arguments.packet_root,
            expected_manifest_sha256=arguments.manifest_sha256,
        )
        source_label = "官方 small 预览归档（最长边约 256px）"

    packet_path = arguments.packet_root / "review-packet.jsonl"
    rows = parse_canonical_jsonl(
        packet_path.read_bytes(),
        label="ABO catalog-photo review packet",
    )
    pairs: dict[str, dict[str, object]] = {}
    for row in rows:
        pairs.setdefault(row["item_id"], {})[row["image_role"]] = row
    ordered = [
        {"item_id": item_id, **roles} for item_id, roles in sorted(pairs.items())
    ]
    carried_payload: dict[str, dict[str, object]] = {}
    carry_manifest_sha256 = None
    if arguments.carry_forward_root is not None:
        _, carried = verify_abo_review_decision_carry_forward(
            arguments.carry_forward_root,
            expected_manifest_sha256=(arguments.carry_forward_manifest_sha256),
            target_packet_root=arguments.packet_root,
            expected_target_packet_manifest_sha256=arguments.manifest_sha256,
            source_packet_root=arguments.source_packet_root,
            expected_source_packet_manifest_sha256=(
                arguments.source_packet_manifest_sha256
            ),
            source_human_review_root=arguments.source_human_review_root,
            expected_source_human_review_manifest_sha256=(
                arguments.source_human_review_manifest_sha256
            ),
        )
        carried_payload = {
            _decision_key(decision.item_id, decision.image_id): decision.model_dump(
                mode="json"
            )
            for decision in carried
        }
        carry_manifest_sha256 = arguments.carry_forward_manifest_sha256

    output = arguments.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(output.parent, "ABO review UI output parent")
    relative_images = Path(
        os.path.relpath(arguments.packet_root, output.parent)
    ).as_posix()
    document = _document(
        payload=json.dumps(ordered, ensure_ascii=False, separators=(",", ":")),
        carried_payload=json.dumps(
            carried_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        image_root=relative_images,
        carry_manifest_sha256=carry_manifest_sha256,
        manifest_sha256=arguments.manifest_sha256,
        pair_count=manifest["candidate_pair_count"],
        source_label=source_label,
    )
    content = document.encode("utf-8")
    atomic_create_file(output, content)
    remaining_images = len(rows) - len(carried_payload)
    print(
        json.dumps(
            {
                "carried_decisions": len(carried_payload),
                "carry_forward_manifest_sha256": carry_manifest_sha256,
                "output": str(output),
                "output_sha256": sha256_bytes(content),
                "pairs": len(ordered),
                "remaining_images": remaining_images,
                "status": "local-review-ui-rendered",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _decision_key(item_id: str, image_id: str) -> str:
    return f"{item_id}\u001f{image_id}"


def _review_storage_key(
    manifest_sha256: str,
    carry_manifest_sha256: str | None,
) -> str:
    key = f"abo-catalog-photo-review:{manifest_sha256}"
    if carry_manifest_sha256 is not None:
        key += f":carry:{carry_manifest_sha256}"
    return key


def _require_real_directory(path: Path, label: str) -> None:
    try:
        snapshot = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} cannot be inspected") from error
    is_junction = getattr(path, "is_junction", None)
    if (
        stat.S_ISLNK(snapshot.st_mode)
        or (is_junction is not None and is_junction())
        or not stat.S_ISDIR(snapshot.st_mode)
    ):
        raise ValueError(f"{label} must be a real non-symlink directory")


def _document(
    *,
    payload: str,
    carried_payload: str,
    image_root: str,
    carry_manifest_sha256: str | None,
    manifest_sha256: str,
    pair_count: int,
    source_label: str,
) -> str:
    safe_root = html.escape(image_root, quote=True)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ABO Catalog Photo 审核</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, "Segoe UI", sans-serif; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #0b0e12; color: #eef2f7; }}
header {{ position: sticky; top: 0; z-index: 2; display: flex; gap: 18px;
  align-items: center; padding: 14px 22px; background: #121821ee;
  border-bottom: 1px solid #2a3544; backdrop-filter: blur(10px); }}
h1 {{ margin: 0; font-size: 18px; }}
.progress {{ flex: 1; height: 8px; background: #27303b; border-radius: 9px;
  overflow: hidden; }}
.progress > div {{ height: 100%; width: 0; background: #58d6a2;
  transition: width .2s; }}
button, input {{ font: inherit; }}
button {{ border: 1px solid #3a4757; background: #1b2430; color: #eef2f7;
  border-radius: 8px; padding: 8px 12px; cursor: pointer; }}
button:hover {{ border-color: #77a8ff; }}
button:disabled {{ cursor: default; opacity: .55; }}
main {{ max-width: 1380px; margin: 0 auto; padding: 22px; }}
.meta {{ display: flex; justify-content: space-between; color: #9fb0c3;
  margin-bottom: 14px; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
.card {{ background: #131922; border: 1px solid #2a3544; border-radius: 14px;
  padding: 14px; }}
.card h2 {{ margin: 0 0 10px; font-size: 15px; color: #b9c9da; }}
.imagebox {{ height: min(54vh, 560px); background: #f5f5f2; border-radius: 10px;
  overflow: hidden; }}
.imagebox a {{ width: 100%; height: 100%; display: block; }}
.imagebox img {{ width: 100%; height: 100%; display: block; object-fit: contain;
  object-position: center; image-rendering: auto; }}
.image-meta {{ margin-top: 8px; color: #9fb0c3; font-size: 12px; }}
.image-meta a {{ color: #77a8ff; }}
.choices {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
  margin-top: 12px; }}
.choice {{ text-align: left; min-height: 48px; }}
.choice.selected {{ outline: 2px solid #58d6a2; background: #17362e; }}
.choice.reject.selected {{ outline-color: #ff7c86; background: #3b1e24; }}
.carried {{ margin-top: 8px; color: #58d6a2; font-size: 12px; }}
.id {{ margin-top: 10px; color: #8294a8; font: 12px ui-monospace, monospace;
  overflow-wrap: anywhere; }}
.toolbar {{ display: flex; gap: 10px; margin-top: 18px; align-items: center; }}
.toolbar .spacer {{ flex: 1; }}
.approve-both {{ background: #16513f; border-color: #3cbf8d; }}
.export {{ background: #224d86; border-color: #4d89d8; }}
.reviewer {{ width: 220px; background: #0f141b; color: white;
  border: 1px solid #3a4757; border-radius: 8px; padding: 9px; }}
.notice {{ margin-top: 14px; padding: 12px; border-left: 3px solid #d9a441;
  background: #2a2417; color: #e8d4a5; }}
.result {{ width: 100%; min-height: 180px; margin-top: 14px; padding: 10px;
  background: #0f141b; color: #d7e3f0; border: 1px solid #3a4757;
  border-radius: 8px; font: 11px ui-monospace, monospace; }}
@media (max-width: 850px) {{
  .pair {{ grid-template-columns: 1fr; }}
  .imagebox {{ height: 42vh; }}
}}
</style>
</head>
<body>
<header>
  <h1>ABO Catalog Photo 审核</h1>
  <span id="counter">0 / {pair_count}</span>
  <div class="progress"><div id="bar"></div></div>
  <span id="done">0 对完成</span>
</header>
<main>
  <div class="meta"><span id="item"></span><span>{html.escape(source_label)}
    · manifest {manifest_sha256[:12]}…</span></div>
  <section class="pair" id="pair"></section>
  <div class="toolbar">
    <button id="prev">← 上一对</button>
    <button id="approveBoth" class="approve-both">两张均为真实商品照片</button>
    <button id="next">下一对 →</button>
    <span class="spacer"></span>
    <input id="reviewer" class="reviewer" placeholder="你的 reviewer_id">
    <button id="showResult">显示完整 JSONL</button>
    <button id="export" class="export">导出完整 JSONL</button>
  </div>
  <div class="notice">绿色“沿用”决定来自已完成的人类审核，且已按旧 packet、
  旧 human-review 和当前图片内容逐项校验，不能在这里修改。页面只展示仍需审核的
  pair；未选择的新图片不会被导出为批准。图片使用 contain 展示，不做裁剪，点击可按
  原始尺寸打开。</div>
  <textarea id="result" class="result" aria-label="审核结果 JSONL"
    readonly hidden></textarea>
</main>
<script>
const allPairs = {payload};
const carried = {carried_payload};
const imageRoot = {json.dumps(safe_root)};
const key = {json.dumps(_review_storage_key(manifest_sha256, carry_manifest_sha256))};
const decisions = JSON.parse(localStorage.getItem(key) || "{{}}");
const decisionKey = row => row.item_id + "\\u001f" + row.image_id;
const isCarried = row => Object.hasOwn(carried, decisionKey(row));
const valueFor = row => isCarried(row)
  ? carried[decisionKey(row)].decision : decisions[decisionKey(row)];
const pairs = allPairs.filter(p => !isCarried(p.main) || !isCarried(p.other));
let index = Number(localStorage.getItem(key + ":index") || 0);
const options = [
  ["approve_catalog_product_photo","批准：真实商品照片",false],
  ["reject_auxiliary_graphic","拒绝：说明/尺寸/包装/拼图",true],
  ["reject_non_product","拒绝：非商品主体",true],
  ["reject_uncertain","拒绝：不确定",true]
];
const carriedReviewers = [...new Set(
  Object.values(carried).map(row => row.reviewer_id)
)];
if (carriedReviewers.length > 1) throw new Error("carried reviewer mismatch");
const reviewerInput = document.querySelector("#reviewer");
if (carriedReviewers.length === 1) {{
  reviewerInput.value = carriedReviewers[0];
  reviewerInput.readOnly = true;
}}
function save() {{
  localStorage.setItem(key, JSON.stringify(decisions));
  localStorage.setItem(key + ":index", String(index));
}}
function completedPairs() {{
  return pairs.filter(p => valueFor(p.main) && valueFor(p.other)).length;
}}
function choose(row, value) {{
  if (isCarried(row)) return;
  decisions[decisionKey(row)] = value;
  save();
  render();
}}
function card(role, row) {{
  const selected = valueFor(row);
  const immutable = isCarried(row);
  const src = `${{imageRoot}}/${{row.local_path}}`;
  return `<article class="card"><h2>${{role === "main" ? "Main image" : "Other image"}}</h2>
    <div class="imagebox"><a href="${{src}}" target="_blank" rel="noopener"><img
      src="${{src}}" alt="${{row.item_id}} ${{role}}"></a></div>
    <div class="image-meta">${{row.width}} × ${{row.height}} px ·
      <a href="${{src}}" target="_blank" rel="noopener">按原始尺寸打开</a></div>
    ${{immutable ? '<div class="carried">已沿用 v1 人工决定（身份与内容校验通过）</div>' : ''}}
    <div class="choices">${{options.map(([value,label,reject]) =>
      `<button class="choice ${{reject ? "reject" : ""}}
       ${{selected === value ? "selected" : ""}}" data-role="${{role}}"
       data-value="${{value}}" ${{immutable ? "disabled" : ""}}>${{label}}</button>`
    ).join("")}}</div>
    <div class="id">${{row.image_id}} · ${{row.source_image_sha256.slice(0,16)}}…</div>
    </article>`;
}}
function render() {{
  if (!pairs.length) throw new Error("no new pairs to review");
  index = Math.max(0, Math.min(index, pairs.length - 1));
  const p = pairs[index];
  document.querySelector("#counter").textContent = `${{index + 1}} / ${{pairs.length}}`;
  document.querySelector("#item").textContent = `item_id: ${{p.item_id}}`;
  document.querySelector("#pair").innerHTML = card("main", p.main)
    + card("other", p.other);
  document.querySelectorAll(".choice").forEach(button => {{
    button.onclick = () => choose(p[button.dataset.role], button.dataset.value);
  }});
  const done = completedPairs();
  document.querySelector("#done").textContent = `${{done}} 对完成`;
  document.querySelector("#bar").style.width = `${{100 * done / pairs.length}}%`;
  document.querySelector("#prev").disabled = index === 0;
  document.querySelector("#next").disabled = index === pairs.length - 1;
  const bothImmutable = isCarried(p.main) && isCarried(p.other);
  document.querySelector("#approveBoth").disabled = bothImmutable;
  save();
}}
document.querySelector("#prev").onclick = () => {{ index--; render(); }};
document.querySelector("#next").onclick = () => {{ index++; render(); }};
document.querySelector("#approveBoth").onclick = () => {{
  const p = pairs[index];
  for (const row of [p.main, p.other]) {{
    if (!isCarried(row)) {{
      decisions[decisionKey(row)] = "approve_catalog_product_photo";
    }}
  }}
  save();
  if (index < pairs.length - 1) index++;
  render();
}};
function completedJsonl() {{
  const reviewer = reviewerInput.value.trim();
  if (!reviewer) {{
    alert("请先填写 reviewer_id");
    return null;
  }}
  if (carriedReviewers.length === 1 && reviewer !== carriedReviewers[0]) {{
    alert("新增决定必须由与既有 ledger 相同的 reviewer 完成");
    return null;
  }}
  const incomplete = pairs.filter(p => !valueFor(p.main) || !valueFor(p.other));
  if (incomplete.length) {{
    alert(`仍有 ${{incomplete.length}} 个新增 pair 未完成`);
    return null;
  }}
  const reviewedAt = new Date().toISOString().replace(/\\.\\d{{3}}Z$/, "Z");
  const rows = allPairs.flatMap(p => [p.main, p.other]).map(row => {{
    if (isCarried(row)) return carried[decisionKey(row)];
    return {{
      schema_version: 1,
      review_policy_version: "abo-catalog-photo-human-review-v1",
      item_id: row.item_id,
      image_id: row.image_id,
      listing_record_sha256: row.listing_record_sha256,
      image_record_sha256: row.image_record_sha256,
      source_image_sha256: row.source_image_sha256,
      decision: decisions[decisionKey(row)],
      reviewer_kind: "human",
      reviewer_id: reviewer,
      reviewed_at: reviewedAt
    }};
  }});
  return rows.map(row => JSON.stringify(row)).join("\\n") + "\\n";
}}
document.querySelector("#showResult").onclick = () => {{
  const content = completedJsonl();
  if (content === null) return;
  const result = document.querySelector("#result");
  result.value = content;
  result.hidden = false;
}};
document.querySelector("#export").onclick = () => {{
  const content = completedJsonl();
  if (content === null) return;
  const blob = new Blob([content], {{type: "application/x-ndjson"}});
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = "abo-image-review-decisions-v2.jsonl";
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}};
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
