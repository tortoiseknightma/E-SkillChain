# Phase 3 corpus contract

This reference defines file shapes and mechanical checks. Every example below is **测试专用** and uses `机械占位-不可发布`; it is not formal corpus material and must never be promoted.

## Seed draft

The seed draft is one UTF-8 JSON object. It declares model provenance and
exactly three distinct strings for each paper-aligned canonical intent:
`exact_match`, `multi_product`, `divergent_rec`, `encyclopedia`, and `utility`.
These IDs must stay identical to the frozen project taxonomy. Recommendation
relations, interaction patterns, and query forms are orthogonal metadata and
must not replace this top-level intent axis.

```json
{
  "provider": "codex",
  "model_display_name": "5.6 Sol Ultra",
  "model_claim_source": "user_confirmation",
  "generated_at": "2000-01-01T00:00:00Z",
  "examples": {
    "exact_match": [
      "机械占位-不可发布-测试专用-01",
      "机械占位-不可发布-测试专用-02",
      "机械占位-不可发布-测试专用-03"
    ],
    "multi_product": [
      "机械占位-不可发布-测试专用-04",
      "机械占位-不可发布-测试专用-05",
      "机械占位-不可发布-测试专用-06"
    ],
    "divergent_rec": [
      "机械占位-不可发布-测试专用-07",
      "机械占位-不可发布-测试专用-08",
      "机械占位-不可发布-测试专用-09"
    ],
    "encyclopedia": [
      "机械占位-不可发布-测试专用-10",
      "机械占位-不可发布-测试专用-11",
      "机械占位-不可发布-测试专用-12"
    ],
    "utility": [
      "机械占位-不可发布-测试专用-13",
      "机械占位-不可发布-测试专用-14",
      "机械占位-不可发布-测试专用-15"
    ]
  }
}
```

## Query draft

A query draft is UTF-8 JSONL with exactly 25 nonblank lines. Each line contains only `plan_id` and `turns`. Valid turn shapes are:

- one turn: `[user]`
- three turns: `[user, assistant, user]`

The final turn must always be a nonblank user turn. A single-turn mechanical record looks like:

```json
{"plan_id":"机械-plan-001","turns":[{"role":"user","content":"机械占位-不可发布-测试专用-01"}]}
```

A three-turn mechanical record looks like:

```json
{"plan_id":"机械-plan-002","turns":[{"role":"user","content":"机械占位-不可发布-测试专用-02"},{"role":"assistant","content":"机械占位-不可发布-测试专用-03"},{"role":"user","content":"机械占位-不可发布-测试专用-04"}]}
```

Do not add image paths, intents, boundary flags, split fields, query IDs, model claims, or labels to a query record. The deterministic plan supplies them.

## Batch draft manifest

The manifest adjacent to the JSONL draft is one UTF-8 JSON object:

```json
{
  "schema_version": 2,
  "base_batch_id": "机械-batch-001",
  "data_origin": "synthetic_derived",
  "provider": "codex",
  "model_display_name": "5.6 Sol Ultra",
  "model_claim_source": "user_confirmation",
  "generated_at": "2000-01-01T00:00:00Z",
  "plan_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "asset_catalog_sha256": null,
  "leakage_policy_version": "relative-path-plus-plan-groups-v1",
  "seed_set_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "draft_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "generation_input_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
}
```

Use a real timezone-aware current UTC timestamp for formal drafts. The timestamp above is a fixed testing placeholder.
`data_origin` is mandatory and has the single accepted value `synthetic_derived`;
source dialogue corpora contribute interaction patterns only, never real-user provenance
or product/visual facts.

## State and quality gates

1. `guard-model` must succeed before formal seed/query composition, image inspection, or inbox writes.
2. `stage-seeds` validates the seed shape and writes a new immutable seed staging revision.
3. `stage-batch` requires an accepted seed set, an active plan, exactly 25 planned records, valid turn shapes, unique normalized final-user text, and exact plan ID coverage.
4. Staging derives plan-owned metadata and writes `results.jsonl`, `manifest.json`, and `quality_report.json`.
5. `accept-seeds` and `accept-batch` require both explicit user authorization and literal CLI confirmation `ACCEPT`.
6. Rejection records a nonblank reason and preserves the staged material under `rejected/`.
7. Accepted data and the accepted ledger are immutable. Reviews, labels, and split products are derived elsewhere.
8. An explicitly authorized all-approved roll-forward may accept one exact
   staged query batch and then generate the ledger-selected next batch in the
   same agent turn. It requires `25/0/0`, reviewer identity and duration,
   explicit acceptance, and explicit next-generation authorization in that
   turn. The accepted transition is verified before generation begins; the
   successor is written only to staging and still requires independent review.
9. If the verified post-acceptance state returns `next_batch_id=null` and
   `next_revision=null`, the active plan is exhausted. This is a successful
   terminal roll-forward: stop before the model gate or any generation write
   and do not fabricate another batch.

Never repair a failed staged result manually. Correct the inbox draft, preserve the rejection reason when applicable, and let the workflow allocate a new revision.
