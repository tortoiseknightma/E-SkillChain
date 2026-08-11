---
name: generate-phase3-corpus
description: Use when generating, reviewing, accepting, rejecting, or explicitly rolling forward formal Phase 3 visual e-commerce query corpus seeds, user utterances, dialogue trajectories, or boundary samples in this repository.
---

# Generate Phase 3 Corpus

## Purpose

Run the human-gated Phase 3 corpus workflow. The deterministic Python pipeline owns plans, validation, hashes, revisions, and promotion; this Skill is the only place where formal seed text or query trajectories may be composed.

Read [references/corpus-contract.md](references/corpus-contract.md) before preparing a seed or query draft. Do not copy its mechanical placeholders into formal output.

## Non-negotiable controls

First classify the requested action as exactly one of `seed`, `generate`, `accept`, `reject`, `status`, or `roll-forward`.

For `seed`, `generate`, or the generation phase of `roll-forward`, require the user to state that the current Codex session is running **5.6 Sol Ultra**. Before inspecting an image or writing a corpus file, run:

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries guard-model --confirmed-model "5.6 Sol Ultra"
```

Only provide `--confirmed-model` when the user explicitly confirmed that display name for the current session. If confirmation is absent, run `guard-model` without the flag, report the model gate, and stop: **不得生成任何正式语料**.

After the gate succeeds, `view_image` may be used and draft files may be placed under `data/queries/inbox/`. Never perform either action before the gate.

**禁止派发子代理**创作、改写、补全或审核正式种子、用户话术、轨迹或边界样本. Keep all formal synthesis in the confirmed main Codex session.

All generated content **只写 staging**. **每批恰好 25 条**. **不得自动 accept**.

`roll-forward` is the only combined workflow. It is allowed only when the
current user turn names the exact staged query batch ID, reports the complete
human-review result as `通过=25，待修改=0，拒绝=0`, includes a nonblank
`reviewer_id` and positive integer `review_minutes`, and explicitly authorizes
both accepting that exact batch and generating the next batch. A review summary
that says it is not corpus acceptance cannot trigger `roll-forward`.

## Workflow

### Status

Status is read-only and does not require the model gate:

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries status --json
uv run python scripts/synth_queries.py --queries-root data/queries stats --json
```

Report accepted and staged counts, seed state, active plan, and the next batch. Do not synthesize content during a status action.

### Seed

1. Pass the model gate above.
2. Inspect workflow state with `status`. Prepare seeds only when no accepted seed set exists.
3. Compose exactly three distinct examples for each paper-aligned canonical intent:
   `exact_match`, `multi_product`, `divergent_rec`, `encyclopedia`, and
   `utility`. These IDs must remain identical to the frozen project taxonomy;
   interaction-pattern or query-relation labels must not replace them. The
   draft itself must declare the current UTC `generated_at`, `provider: codex`,
   `model_display_name: 5.6 Sol Ultra`, and
   `model_claim_source: user_confirmation`.
4. Write the JSON draft to `data/queries/inbox/seeds/<seed_batch_id>.json` and stage it:

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries stage-seeds --input data/queries/inbox/seeds/<seed_batch_id>.json --seed-batch-id <seed_batch_id>
```

5. Link the staged seed files and summarize coverage. Stop for human review; do not accept them.

### Generate

1. Pass the model gate above.
2. Run `status` and require `seed_status` to be `accepted`. If it is not accepted, stop before image inspection or composition and ask the user to review/accept a staged seed set. Then run `show-next`; use only the returned active plan and next `base_batch_id`.

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries status --json
uv run python scripts/synth_queries.py --queries-root data/queries show-next --json
```

3. Read `data/queries/seeds/accepted/seed_examples.json` and its adjacent manifest. Use the accepted examples only as intent/style anchors, do not copy them, and do not use any inbox or rejected seed draft.
4. For every planned item, resolve the plan's relative `image_path` beneath `data/clean/` and pass the resulting absolute local path to `view_image`; keep the stored plan path unchanged. Inspect the actual image and do not infer visual facts from the filename, directory, plan label, or prior examples.
5. Produce exactly 25 JSONL records containing only `plan_id` and valid `turns`. Preserve every plan ID exactly. Create an adjacent manifest declaring the same `base_batch_id`, `data_origin: synthetic_derived`, the current UTC `generated_at`, and the confirmed model provenance. Mock trajectories must never be described as real user logs.
6. Write both files beneath `data/queries/inbox/batches/`, then stage them:

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries stage-batch --draft data/queries/inbox/batches/<base_batch_id>.jsonl --draft-manifest data/queries/inbox/batches/<base_batch_id>.manifest.json --base-batch-id <base_batch_id>
```

7. Read `data/queries/staging/<batch_id>/quality_report.json`. Report the revisioned batch ID, count, intent distribution, boundary count, single-/multi-turn distribution, duplicate count, and clickable local links to the report and results.
8. Stop and wait for review. Never infer acceptance from silence, prior approvals, or a request to generate the next batch.

### Accept

Accept only when the user names the exact staged seed/batch ID and gives an unambiguous **明确接受** instruction in the current turn. Translate that authorization to the literal CLI confirmation `ACCEPT`.

For a seed set:

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries accept-seeds --seed-batch-id <seed_batch_id> --confirmation ACCEPT
```

For a query batch:

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries accept-batch --batch-id <revisioned_batch_id> --confirmation ACCEPT
```

Report the promoted path and updated status.

Outside the explicitly authorized `roll-forward` workflow below, never combine
generation and acceptance in one action.

### Roll forward after an all-approved review

Use this only for query batches, never seed sets.

1. Verify all `roll-forward` conditions above from the current user turn. Do
   not infer any missing count, reviewer field, batch ID, acceptance, or
   continue-generation authorization from browser state, prior turns, or
   silence.
2. Accept the exact staged batch with literal confirmation `ACCEPT`. Supply the
   same asset catalog bound by its manifest. Then run `status` and verify that
   the accepted ledger advanced by exactly that batch and identifies the next
   base batch.
3. If `status` returns `next_batch_id=null` and `next_revision=null`, verify
   that the active plan is exhausted and stop successfully before the model
   gate, image inspection, or any inbox write. Report that the accepted batch
   closed the plan; never invent a successor.
4. Otherwise, run the complete `Generate` workflow for exactly that next base batch,
   including the model gate, accepted seed read, per-item image inspection,
   25-record draft, manifest, official staging, and quality-report read.
5. Stop with the next batch in staging. Never accept the newly generated batch
   in the same roll-forward action; it requires its own human review.
6. Treat the two state transitions as sequential and auditable, not atomic. If
   acceptance succeeds but next-batch generation fails, report the accepted
   state and the exact generation failure. Never roll back or mutate accepted
   data, and never fabricate a staged successor.

### Reject

Reject only the exact staged ID identified by the user. Preserve their substantive reason; do not replace it with a generic phrase.

```powershell
uv run python scripts/synth_queries.py --queries-root data/queries reject-seeds --seed-batch-id <seed_batch_id> --reason "<user_reason>"
uv run python scripts/synth_queries.py --queries-root data/queries reject-batch --batch-id <revisioned_batch_id> --reason "<user_reason>"
```

Report the rejected path and use a new revision for any later retry.

## Quality boundary

The frozen project taxonomy is authoritative for the top-level intent axis.
Keep interaction patterns, recommendation relations, and capability subtypes
on separate fields; never promote them into `canonical_intent` or use them to
replace the paper's five evaluation strata.

The active plan is authoritative for image path, intent, boundary flag, split, ordering, and IDs. Do not change those fields in model output. Never mutate accepted directories or `accepted-ledger.jsonl`; all labels and reviews are derived artifacts.

If any validation, hash, duplicate, plan, seed, or count check fails, stop and report the exact failure. Do not bypass the validator or hand-edit staging output.
