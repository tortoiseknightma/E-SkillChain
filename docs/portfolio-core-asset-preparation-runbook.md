# Portfolio core clean-asset preparation

This is the non-formal, resumable preparation step before the 1,500-query
Portfolio core run.  It copies only source files that a source-specific,
declarative inventory already identifies.  It does not crawl a dataset,
extract an archive, create query text, infer product identities, or replace a
formal source/label review.

Each emitted manifest declares `track=portfolio`, `formal_eligible=false`, and
`formal_status=non_formal`.

## Source/pool policy

Create a create-only policy snapshot before writing an inventory:

```powershell
uv run python scripts/portfolio_core_assets.py write-default-policy `
  --output E:\skillchain-data\clean\portfolio-core-source-policy-v2.json
```

The policy is deliberately narrower than raw-download availability.  It
registers the reusable dev-mini sources without broadening their existing
roles:

| Source | Permitted clean-image pool |
| --- | --- |
| `abo` | `exact_match` |
| `rpc` | `multi_product` |
| `fashioniq` | `divergent_rec` |
| `inaturalist` | `encyclopedia` challenge candidates |
| `isia_food500` | `utility.recipe_guidance` only; 116 fresh regular images fill the recipe side of utility |
| `wikimedia_commons_documents` | `utility` challenge candidates |
| `products_10k` | `exact_match` only, after its archive is explicitly materialized |
| `polyvore` | `divergent_rec` only, from the owner-selected image-bearing snapshot |
| `recipe1m_plus` | `utility` only, from the bounded mapped selection |

`DeepFashion` remains `pending_extraction` and exposes no generic positive
query pool: its registered role is candidate gallery / same-item exclusion.
`ISIA Food-500` is intentionally narrower than the generic utility pool: its
selected images bind only to `utility.recipe_guidance`; the source ceiling
rejects document-reading or product-retrieval use.  The regular-source core
inventory combines 116 fresh ISIA images with the 30 already verified
dev-mini Wikimedia document images, preserving both utility capabilities when
the live Wikimedia expansion cannot supply enough additional files.
`SKU-110K` and `MEP-3M` are `restricted` challenge/negative sources, not
generic retrieval-positive pools.  `SIMMC 2.1` and `CSDS` are metadata/pattern
sources for this layer; `U-NEED` remains permanently unavailable.

The loader applies code-level pool ceilings as well as the policy snapshot, so
a hand-edited policy cannot silently turn Products-10K into multi-product,
Polyvore into a generic source, or restricted sources into positives.  A
different permitted state still needs a new policy file and a new clean-output
root; existing output manifests are create-only.

## Build the regular-source inventory

The regular-file builder does not open ABO/RPC archives.  It verifies the
already accepted dev-mini manifest, carries forward its 30 Wikimedia document
images, excludes prior mini identities from new selections, and deterministically
selects 130 FashionIQ, 122 iNaturalist and 116 ISIA Food-500 images:

```powershell
uv run python scripts/portfolio_core_inventory.py build-regular `
  --dev-mini-selection <dev-mini-selection-manifest> `
  --fashioniq-adapter-root <verified-fashioniq-adapter> `
  --fashioniq-manifest-sha256 <verified-fashioniq-manifest-sha256> `
  --fashioniq-asset-root <fashioniq-regular-image-root> `
  --inaturalist-manifest <core-inaturalist-manifest-jsonl> `
  --inaturalist-asset-root <core-inaturalist-image-root> `
  --commons-document-manifest <dev-mini-commons-document-manifest-jsonl> `
  --commons-document-asset-root <dev-mini-commons-document-image-root> `
  --isia-food-manifest <core-isia-food500-manifest-jsonl> `
  --isia-food-asset-root <core-isia-food500-image-root> `
  --output <regular-candidates-jsonl>
```

The result has 398 included rows and unique pool counts of 130 divergent,
122 encyclopedia and 146 utility.  FashionIQ intentionally exceeds its raw
122 preflight floor because the core planner consumes a fresh dev prefix and
allocates by catalog leakage component.  The verified iNaturalist
122-row staging already leaves exactly the required 95 tail components after
the fresh dev prefix.  Utility likewise has zero component margin, but the
frozen deterministic `prepare` run verified exactly 86 usable tail components
for 86 required; it is not expanded merely for unused margin.  Revisit these
two exact-capacity pools only if the source policy, selection seed, catalog
component policy, or plan changes.  An
approved bounded Recipe1M+ inventory
may be supplied with the optional Recipe1M+ arguments, but those rows are
`reserve` only and do not inflate capacity.  The current core build does not
need them.

Fresh-source quotas are evaluated **after** excluding every dev-mini source
record and exact content hash.  Therefore an unfiltered ISIA materialization
that contains the 30 dev-mini recipe records must supply at least 146
content-unique rows to leave 116 fresh candidates; an already prefiltered
ISIA root needs 116.  The same rule applies to iNaturalist: a prefiltered root
needs 122 rows, while an unfiltered root with all 27 mini overlaps needs at
least 149.  A smaller unfiltered staging root
fails closed and must not be described as core-ready.

## Build the selected-only ABO/RPC adapters

ABO and RPC are read directly from their locked archives, but only the bounded
selected members are materialized.  Their output roots must be new clean roots
that are disjoint from the RAW root:

```powershell
uv run python scripts/portfolio_core_source_adapters.py abo `
  --raw-root E:\skillchain-data\raw `
  --source-policy E:\skillchain-data\clean\portfolio-core-source-policy-v2.json `
  --dev-mini-selection-manifest <dev-mini-selection-manifest> `
  --output-root <new-clean-abo-adapter-root> `
  --include-count 340 `
  --reserve-count 8 `
  --cross-intent-count 25

uv run python scripts/portfolio_core_source_adapters.py rpc `
  --raw-root E:\skillchain-data\raw `
  --source-policy E:\skillchain-data\clean\portfolio-core-source-policy-v2.json `
  --dev-mini-selection-manifest <dev-mini-selection-manifest> `
  --output-root <new-clean-rpc-adapter-root>
```

The ABO defaults are 340 included rows, 8 reserves, and 25 cross-intent-
eligible rows.  ABO contributes about two image rows per product, so the old
157-row floor collapsed to only 78 product/leakage components; after the
35-component dev prefix it left 43 components for a tail that needs at least
122 groups.  The larger default supplies a safe product-component margin.
These are still only conservative candidate counts: the create-only
AssetCatalog and the later `prepare` command remain the authoritative checks
for independent batch groups and at least eight triplet components.  Explicit
CLI count flags remain supported for a new immutable output root.

After the selected-only ABO and RPC adapters have published their own strict
candidate files, merge all three inputs.  Merge order does not affect bytes;
the default command fails unless every core pool floor is met:

```powershell
uv run python scripts/portfolio_core_inventory.py merge `
  --inventory <regular-candidates-jsonl> `
  --inventory <abo-adapter-root>\candidates.jsonl `
  --inventory <rpc-adapter-root>\candidates.jsonl `
  --output <core-candidates-v2-jsonl>
```

## Candidate inventory contract

An adapter or a reviewed local inventory writes canonical JSONL.  This module
does not generate the rows.  Each row includes:

- `candidate_id`, `source_id`, `selection` (`include` or `reserve`), and a
  policy-allowed pool;
- a source-root-relative `source_local_path`, clean-relative
  `destination_path`, exact source byte length and SHA-256;
- a complete `DatasetAssetDraft` whose original `local_path` equals the source
  path and whose `source_dataset` exactly equals the candidate `source_id`; and
- one or more frozen-taxonomy `capability_bindings`.

Included rows are checked against both the source/pool ceiling and the
source/capability ceiling.  In particular, ISIA Food-500 can bind only to
`utility.recipe_guidance`, even though its files live in the shared `utility`
pool.

`destination_path` must be below `query_images/<pool>/`.  The later published
draft replaces only its local path with that clean-relative destination; source
provenance and permission fields are retained.

## Preflight and checkpointed materialization

Preflight is read-only.  It reports source state, exact-byte readiness,
selected counts by pool and capability, and capacity shortfalls:

```powershell
uv run python scripts/portfolio_core_assets.py preflight `
  --inventory <core-candidates-jsonl> `
  --source-policy E:\skillchain-data\clean\portfolio-core-source-policy-v2.json `
  --source-root abo=<clean-abo-adapter-root> `
  --source-root rpc=<clean-rpc-adapter-root> `
  --source-root fashioniq=<verified-fashioniq-asset-root> `
  --source-root inaturalist=<fresh-core-inaturalist-root> `
  --source-root isia_food500=<fresh-core-isia-food500-root> `
  --source-root wikimedia_commons_documents=<dev-mini-commons-document-image-root> `
  --source-root recipe1m_plus=E:\skillchain-data\raw\recipe1m_plus\selected
```

To avoid describing a one-item or dev-mini-sized inventory as core-ready, the
default preflight uses conservative **pool** floors for the fresh 1,500-query /
60-batch plan: 157 exact, 105 multi-product, 122 divergent, 122 encyclopedia,
and 146 utility images.  It counts unique declared content SHA-256 values
within each pool.  These account for the 200-query dev prefix plus the core
tail and still assume distinct leakage components.  They are not proof of
planner feasibility, source-positive semantics, or a formal review result.
In particular, the 157 exact-image preflight floor is lower than the current
340-row ABO adapter default because preflight counts unique bytes while the
planner isolates product/leakage components across 52 tail batches.
After catalog publication, run the core planning dry-run/`prepare` check; it
remains the authoritative end-to-end capacity check.

For a full candidate inventory that satisfies the floor, materialize in bounded
calls.  `--max-items` limits only newly checkpointed images and makes the
command safe to interrupt and resume:

```powershell
uv run python scripts/portfolio_core_assets.py materialize `
  --inventory <core-candidates-jsonl> `
  --source-policy E:\skillchain-data\clean\portfolio-core-source-policy-v2.json `
  --source-root abo=<clean-abo-adapter-root> `
  --source-root rpc=<clean-rpc-adapter-root> `
  --source-root fashioniq=<verified-fashioniq-asset-root> `
  --source-root inaturalist=<fresh-core-inaturalist-root> `
  --source-root isia_food500=<fresh-core-isia-food500-root> `
  --source-root wikimedia_commons_documents=<dev-mini-commons-document-image-root> `
  --source-root recipe1m_plus=E:\skillchain-data\raw\recipe1m_plus\selected `
  --output-root <new-clean-portfolio-core-root> `
  --max-items 250
```

Each file is atomically created, verified against the declared byte count and
SHA-256, then followed by an immutable checkpoint.  If interruption happens
between those writes, the next invocation verifies the image and publishes its
missing checkpoint without overwriting it.  A completed output has
`dataset-assets.jsonl` and `selection-manifest.json`; RAW input is never
deleted, renamed, extracted, or modified.

Before it writes a run manifest, materialization resolves the output root and
all selected source roots and rejects equality or either ancestor/descendant
overlap.  This turns the RAW/source non-mutation promise into a fail-closed
runtime boundary rather than relying on the operator to notice a bad path.

`--allow-incomplete-capacity` exists only for a bounded smoke/trial selection.
It must not be used to claim a core-ready clean pool.

## Catalog, assignments, and core plan

Build the generic catalog from the emitted drafts and clean root:

```powershell
uv run python -m skillchain.data.asset_catalog_cli build `
  --drafts E:\skillchain-data\clean\portfolio-core-v1\dataset-assets.jsonl `
  --asset-root E:\skillchain-data\clean\portfolio-core-v1 `
  --output E:\skillchain-data\clean\portfolio-core-asset-catalog-v1 `
  --coverage-root query_images
```

Then bind cataloged assets to the inventory-declared capability mappings:

```powershell
uv run python scripts/portfolio_core_assets.py build-assignments `
  --selection-manifest E:\skillchain-data\clean\portfolio-core-v1\selection-manifest.json `
  --asset-catalog E:\skillchain-data\clean\portfolio-core-asset-catalog-v1 `
  --asset-root E:\skillchain-data\clean\portfolio-core-v1 `
  --output E:\skillchain-data\clean\portfolio-core-capability-assignments-v1.jsonl
```

Use the SHA-256 of that JSONL in the existing
[`portfolio core synthesis runbook`](portfolio-core-synthesis-runbook.md) when
running `prepare`.  The `prepare` command rebuilds the fresh dev-mini prefix
and core plan against this catalog, which is the direct end-to-end feasibility
check before any query text is composed.
