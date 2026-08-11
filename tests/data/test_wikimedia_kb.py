from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from pydantic import ValidationError
import pytest

import skillchain.data.wikimedia_kb as wikimedia_kb
from skillchain.data.kb_catalog import KBEntryV2
from skillchain.data.wikimedia_kb import (
    LICENSE_ID,
    LICENSE_URI,
    REVIEW_ATTESTATION,
    TERMS_REVISION_URI,
    VALIDATOR_ABSENT_REASON,
    WikimediaKBError,
    WikimediaLicenseEvidence,
    WikimediaPermissionMatrix,
    WikimediaReviewDecision,
    WikimediaReviewLedger,
    WikimediaSelectionEntry,
    WikimediaSelectionLock,
    build_wikimedia_kb_bundle,
    canonical_json_document,
    canonicalize_revision_api_response,
    load_verified_selection_lock,
    load_verified_wikimedia_kb_bundle,
    page_history_uri,
    revision_api_uri,
    revision_source_uri,
    write_revision_api_snapshot,
)

UTC_NOW = datetime(2026, 7, 21, 1, 2, 3, tzinfo=timezone.utc)
REVISION_TIME = datetime(2026, 7, 20, tzinfo=timezone.utc)
HTTP_DATE = "Tue, 21 Jul 2026 01:02:03 GMT"
REQUEST_ID = "1eedc8da-0a0a-403d-8d42-b3337c361572"


@dataclass(frozen=True)
class FormalCase:
    snapshots: Path
    evidence: Path
    lock_path: Path
    lock: WikimediaSelectionLock
    lock_sha256: str
    license_sha256: dict[str, str]
    review_sha256: str


def _raw_api_response(
    *, page_id: int, revision_id: int, title: str, content: str
) -> bytes:
    return json.dumps(
        {
            "batchcomplete": True,
            "query": {
                "pages": [
                    {
                        "pageid": page_id,
                        "ns": 0,
                        "title": title,
                        "revisions": [
                            {
                                "revid": revision_id,
                                "parentid": revision_id - 1,
                                "timestamp": "2026-07-20T00:00:00Z",
                                "sha1": hashlib.sha1(
                                    content.encode("utf-8")
                                ).hexdigest(),
                                "slots": {
                                    "main": {
                                        "contentmodel": "wikitext",
                                        "contentformat": "text/x-wiki",
                                        "content": content,
                                    }
                                },
                            }
                        ],
                    }
                ]
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _permission_matrix() -> WikimediaPermissionMatrix:
    return WikimediaPermissionMatrix(
        local_research_allowed=True,
        cloud_processing_allowed=True,
        redistribution_allowed=True,
        public_demo_allowed=True,
        attribution_required=True,
        license_notice_required=True,
        indicate_changes_required=True,
        share_alike_on_adaptations=True,
        preserve_additional_attribution_notices=True,
        no_endorsement=True,
    )


def _license_evidence(evidence_root: Path) -> tuple[WikimediaLicenseEvidence, ...]:
    result: list[WikimediaLicenseEvidence] = []
    for project in ("zh.wikibooks.org", "zh.wikipedia.org"):
        name = project.replace(".", "-") + "-terms.html"
        text = (
            "Reviewed Wikimedia policy capture: Attribution-ShareAlike 4.0; "
            "CC BY-SA 4.0; https://creativecommons.org/licenses/by-sa/4.0/; "
            f"immutable policy URI {TERMS_REVISION_URI}. "
            "This fixture represents the exact externally archived evidence bytes."
        ).encode("utf-8")
        (evidence_root / name).write_bytes(text)
        result.append(
            WikimediaLicenseEvidence(
                applicable_project=project,
                license_id=LICENSE_ID,
                license_uri=LICENSE_URI,
                policy_revision_uri=TERMS_REVISION_URI,
                evidence_file=name,
                evidence_sha256=hashlib.sha256(text).hexdigest(),
                acquired_at=UTC_NOW,
                http_status=200,
                final_uri=TERMS_REVISION_URI,
                etag=f'"{project}-policy-revision"',
            )
        )
    return tuple(result)


def _decision(entry: WikimediaSelectionEntry) -> WikimediaReviewDecision:
    return WikimediaReviewDecision(
        kind=entry.kind,
        project_domain=entry.project_domain,
        page_id=entry.page_id,
        revision_id=entry.revision_id,
        raw_response_sha256=entry.raw_response_sha256,
        acquisition_receipt_sha256=entry.acquisition_receipt_sha256,
        response_sha256=entry.response_sha256,
        evidence_start_char=entry.evidence_start_char,
        evidence_end_char=entry.evidence_end_char,
        evidence_sha256=entry.evidence_sha256,
        decision="approved",
        reviewer_kind="human",
        reviewer_id="fixture-reviewer",
        reviewed_at=UTC_NOW,
        evidence_reviewed_in_revision_context=True,
        page_rights_and_attribution_reviewed=True,
        automated_approval_delegated=False,
        decision_basis="Exact revision span and page rights notices reviewed.",
        attestation=REVIEW_ATTESTATION,
    )


def _write_lock(path: Path, lock: WikimediaSelectionLock) -> str:
    content = canonical_json_document(lock)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _make_formal_case(tmp_path: Path) -> FormalCase:
    snapshots = tmp_path / "snapshots"
    evidence = tmp_path / "evidence"
    snapshots.mkdir(parents=True)
    evidence.mkdir()
    license_evidence = _license_evidence(evidence)
    entries: list[WikimediaSelectionEntry] = []
    for kind in ("encyclopedia", "recipe"):
        domain = "zh.wikipedia.org" if kind == "encyclopedia" else "zh.wikibooks.org"
        base = 1_000 if kind == "encyclopedia" else 2_000
        for index in range(10):
            page_id = base + index
            revision_id = 10_000 + base + index
            title = (
                f"Encyclopedia sample {index:02d}"
                if kind == "encyclopedia"
                else f"\u98df\u8c31/Sample recipe {index:02d}"
            )
            content = (
                f"{title} fixed revision evidence. "
                + "evidence words ingredients method temperature safety " * 10
            )
            raw = _raw_api_response(
                page_id=page_id,
                revision_id=revision_id,
                title=title,
                content=content,
            )
            prefix = f"{kind}-{index:02d}"
            raw_name = f"{prefix}.raw.json"
            receipt_name = f"{prefix}.receipt.json"
            response_name = f"{prefix}.snapshot.json"
            request_uri = revision_api_uri(domain, revision_id)
            report = write_revision_api_snapshot(
                raw,
                snapshots / response_name,
                raw_response_destination=snapshots / raw_name,
                acquisition_receipt_destination=snapshots / receipt_name,
                request_uri=request_uri,
                final_uri=request_uri,
                http_status=200,
                http_date=HTTP_DATE,
                x_request_id=(f"00000000-0000-4000-8000-{revision_id:012d}"),
                acquired_at=UTC_NOW,
                etag=f'"revision-{revision_id}"' if index else None,
                validator_absent_reason=(
                    VALIDATOR_ABSENT_REASON if index == 0 else None
                ),
            )
            source_uri = revision_source_uri(domain, revision_id)
            entries.append(
                WikimediaSelectionEntry(
                    kind=kind,
                    project_domain=domain,
                    raw_response_file=raw_name,
                    raw_response_sha256=report.raw_response_sha256,
                    acquisition_receipt_file=receipt_name,
                    acquisition_receipt_sha256=report.acquisition_receipt_sha256,
                    response_file=response_name,
                    response_sha256=report.response_sha256,
                    page_id=page_id,
                    revision_id=revision_id,
                    revision_timestamp=REVISION_TIME,
                    revision_sha1=hashlib.sha1(content.encode("utf-8")).hexdigest(),
                    canonical_title=title,
                    evidence_start_char=0,
                    evidence_end_char=len(content),
                    evidence_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    source_uri=source_uri,
                    history_uri=page_history_uri(domain, page_id),
                    license_id=LICENSE_ID,
                    license_uri=LICENSE_URI,
                    attribution=(
                        f'{domain} contributors, "{title}", revision '
                        f"{revision_id}; {source_uri}"
                    ),
                    page_footer_history_and_talk_reviewed=True,
                    unresolved_rights_notice=False,
                )
            )
    entries.sort(key=lambda item: (item.kind, item.page_id, item.revision_id))
    ledger = WikimediaReviewLedger(
        selection_id="fixture-wikimedia-mini-v1",
        decisions=tuple(_decision(entry) for entry in entries),
    )
    review_name = "human-review-ledger.json"
    review_bytes = canonical_json_document(ledger)
    (evidence / review_name).write_bytes(review_bytes)
    review_sha = hashlib.sha256(review_bytes).hexdigest()
    lock = WikimediaSelectionLock(
        selection_id=ledger.selection_id,
        created_at=UTC_NOW,
        permission_matrix=_permission_matrix(),
        license_evidence=license_evidence,
        entries=tuple(entries),
        reviewer_kind="human",
        reviewer_id="fixture-reviewer",
        reviewed_at=UTC_NOW,
        review_record_file=review_name,
        review_record_uri=f"urn:sha256:{review_sha}",
        review_record_sha256=review_sha,
    )
    lock_path = tmp_path / "selection-lock.json"
    lock_sha = _write_lock(lock_path, lock)
    return FormalCase(
        snapshots=snapshots,
        evidence=evidence,
        lock_path=lock_path,
        lock=lock,
        lock_sha256=lock_sha,
        license_sha256={
            item.applicable_project: item.evidence_sha256 for item in license_evidence
        },
        review_sha256=review_sha,
    )


def _build(case: FormalCase, output: Path):
    return build_wikimedia_kb_bundle(
        case.snapshots,
        case.evidence,
        output,
        selection_lock_path=case.lock_path,
        expected_selection_lock_sha256=case.lock_sha256,
        expected_license_evidence_sha256=case.license_sha256,
        expected_review_ledger_sha256=case.review_sha256,
    )


def _replace_lock(case: FormalCase, **updates) -> FormalCase:
    payload = case.lock.model_dump(mode="json")
    payload.update(updates)
    lock = WikimediaSelectionLock.model_validate(payload)
    lock_sha = _write_lock(case.lock_path, lock)
    return FormalCase(
        snapshots=case.snapshots,
        evidence=case.evidence,
        lock_path=case.lock_path,
        lock=lock,
        lock_sha256=lock_sha,
        license_sha256=case.license_sha256,
        review_sha256=case.review_sha256,
    )


def test_acquisition_records_raw_response_and_http_receipt(tmp_path: Path) -> None:
    content = "fixed evidence words " * 10
    raw = _raw_api_response(page_id=12, revision_id=34, title="Sample", content=content)
    uri = revision_api_uri("zh.wikipedia.org", 34)
    root = tmp_path / "capture"
    root.mkdir()
    report = write_revision_api_snapshot(
        raw,
        root / "sample.snapshot.json",
        raw_response_destination=root / "sample.raw.json",
        acquisition_receipt_destination=root / "sample.receipt.json",
        request_uri=uri,
        final_uri=uri,
        http_status=200,
        http_date=HTTP_DATE,
        x_request_id=REQUEST_ID,
        acquired_at=UTC_NOW,
        last_modified="Tue, 21 Jul 2026 01:02:03 GMT",
    )
    receipt = json.loads((root / "sample.receipt.json").read_bytes())
    assert receipt["raw_response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["http_status"] == 200
    assert receipt["final_uri"] == uri
    assert receipt["last_modified"]
    assert (
        report.response_sha256
        == hashlib.sha256((root / "sample.snapshot.json").read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        write_revision_api_snapshot(
            raw,
            root / "sample.snapshot.json",
            raw_response_destination=root / "sample.raw.json",
            acquisition_receipt_destination=root / "sample.receipt.json",
            request_uri=uri,
            final_uri=uri,
            http_status=200,
            http_date=HTTP_DATE,
            x_request_id=REQUEST_ID,
            acquired_at=UTC_NOW,
            etag='"same"',
        )


def test_acquisition_receipt_audits_absent_validator_and_rejects_bad_uri(
    tmp_path: Path,
) -> None:
    raw = _raw_api_response(
        page_id=12,
        revision_id=34,
        title="Sample",
        content="fixed evidence words " * 10,
    )
    uri = revision_api_uri("zh.wikipedia.org", 34)
    for final_uri, etag, absence_reason in (
        (uri, None, None),
        ("https://example.com/redirect", '"x"', None),
    ):
        root = (
            tmp_path / hashlib.sha256((final_uri + str(etag)).encode()).hexdigest()[:8]
        )
        root.mkdir()
        with pytest.raises(WikimediaKBError, match="receipt"):
            write_revision_api_snapshot(
                raw,
                root / "sample.snapshot.json",
                raw_response_destination=root / "sample.raw.json",
                acquisition_receipt_destination=root / "sample.receipt.json",
                request_uri=uri,
                final_uri=final_uri,
                http_status=200,
                http_date=HTTP_DATE,
                x_request_id=REQUEST_ID,
                acquired_at=UTC_NOW,
                etag=etag,
                validator_absent_reason=absence_reason,
            )

    root = tmp_path / "no-validator"
    root.mkdir()
    report = write_revision_api_snapshot(
        raw,
        root / "sample.snapshot.json",
        raw_response_destination=root / "sample.raw.json",
        acquisition_receipt_destination=root / "sample.receipt.json",
        request_uri=uri,
        final_uri=uri,
        http_status=200,
        http_date=HTTP_DATE,
        x_request_id=REQUEST_ID,
        acquired_at=UTC_NOW,
        validator_absent_reason=VALIDATOR_ABSENT_REASON,
    )
    receipt = json.loads((root / "sample.receipt.json").read_bytes())
    assert (
        report.acquisition_receipt_sha256
        == hashlib.sha256((root / "sample.receipt.json").read_bytes()).hexdigest()
    )
    assert receipt["validator_kind"] == "none"
    assert receipt["http_date"] == HTTP_DATE
    assert receipt["x_request_id"] == REQUEST_ID
    assert receipt["validator_absent_reason"] == VALIDATOR_ABSENT_REASON


def test_canonicalizer_rejects_duplicate_keys_and_wrong_revision() -> None:
    with pytest.raises(WikimediaKBError, match="strict UTF-8 JSON"):
        canonicalize_revision_api_response(
            b'{"batchcomplete":true,"batchcomplete":true,"query":{}}',
            request_uri=revision_api_uri("zh.wikipedia.org", 1),
            acquired_at=UTC_NOW,
        )
    raw = _raw_api_response(
        page_id=1,
        revision_id=2,
        title="Sample",
        content="long enough evidence words " * 5,
    )
    with pytest.raises(WikimediaKBError, match="does not match"):
        canonicalize_revision_api_response(
            raw,
            request_uri=revision_api_uri("zh.wikipedia.org", 3),
            acquired_at=UTC_NOW,
        )


def test_candidate_manifest_is_explicitly_nonformal_and_requires_receipts() -> None:
    path = (
        Path(__file__).parents[2]
        / "specs"
        / "data_sources"
        / "wikimedia-mini-acquisition-v0.json"
    )
    candidate = json.loads(path.read_bytes())
    assert candidate["formal_eligible"] is False
    assert candidate["api_contract"]["formal_build_network_allowed"] is False
    assert candidate["api_contract"]["raw_response_receipt_required"] is True
    assert candidate["formalization_required"]


def test_formal_bundle_reparses_and_loads_through_external_digest(
    tmp_path: Path,
) -> None:
    case = _make_formal_case(tmp_path)
    output = tmp_path / "bundle"
    report = _build(case, output)
    assert report.encyclopedia_entries == report.recipe_entries == 10
    assert set(path.name for path in output.iterdir()) == {
        "encyclopedia.jsonl",
        "recipes.jsonl",
        "manifest.json",
    }
    verified = load_verified_wikimedia_kb_bundle(
        output, expected_bundle_sha256=report.manifest.bundle_sha256
    )
    assert verified.manifest.review_ledger_sha256 == case.review_sha256
    assert verified.manifest.license_evidence_sha256 == case.license_sha256
    entries = [
        KBEntryV2.model_validate_json(line)
        for name in ("encyclopedia.jsonl", "recipes.jsonl")
        for line in (output / name).read_bytes().splitlines()
    ]
    assert len(entries) == 20
    assert all(entry.formally_verified and entry.origin == "dump" for entry in entries)


def test_every_external_digest_is_required_and_independent(tmp_path: Path) -> None:
    case = _make_formal_case(tmp_path)
    load_verified_selection_lock(
        case.lock_path, expected_selection_lock_sha256=case.lock_sha256
    )
    with pytest.raises(WikimediaKBError, match="external expected digest"):
        load_verified_selection_lock(
            case.lock_path, expected_selection_lock_sha256="0" * 64
        )
    bad_license = dict(case.license_sha256)
    bad_license["zh.wikipedia.org"] = "0" * 64
    with pytest.raises(WikimediaKBError, match="license evidence"):
        build_wikimedia_kb_bundle(
            case.snapshots,
            case.evidence,
            tmp_path / "bad-license",
            selection_lock_path=case.lock_path,
            expected_selection_lock_sha256=case.lock_sha256,
            expected_license_evidence_sha256=bad_license,
            expected_review_ledger_sha256=case.review_sha256,
        )
    with pytest.raises(WikimediaKBError, match="review ledger"):
        build_wikimedia_kb_bundle(
            case.snapshots,
            case.evidence,
            tmp_path / "bad-review",
            selection_lock_path=case.lock_path,
            expected_selection_lock_sha256=case.lock_sha256,
            expected_license_evidence_sha256=case.license_sha256,
            expected_review_ledger_sha256="0" * 64,
        )


def test_snapshot_cannot_be_fabricated_independently_of_raw_receipt(
    tmp_path: Path,
) -> None:
    case = _make_formal_case(tmp_path)
    selected = case.lock.entries[0]
    response_path = case.snapshots / selected.response_file
    changed = response_path.read_bytes().replace(b"evidence", b"fabricate", 1)
    response_path.write_bytes(changed)
    entry = selected.model_copy(
        update={"response_sha256": hashlib.sha256(changed).hexdigest()}
    )
    entries = (entry,) + case.lock.entries[1:]
    case = _replace_lock(
        case, entries=[item.model_dump(mode="json") for item in entries]
    )
    with pytest.raises(
        WikimediaKBError,
        match="acquisition receipt does not match|receipt does not bind",
    ):
        _build(case, tmp_path / "output")


def test_license_evidence_is_read_not_merely_named(tmp_path: Path) -> None:
    case = _make_formal_case(tmp_path)
    reference = case.lock.license_evidence[0]
    invalid = (
        f"unrelated text that happens to cite {TERMS_REVISION_URI} " * 4
    ).encode()
    (case.evidence / reference.evidence_file).write_bytes(invalid)
    digest = hashlib.sha256(invalid).hexdigest()
    changed_reference = reference.model_copy(update={"evidence_sha256": digest})
    references = (changed_reference,) + case.lock.license_evidence[1:]
    license_digests = dict(case.license_sha256)
    license_digests[reference.applicable_project] = digest
    case = _replace_lock(
        case,
        license_evidence=[item.model_dump(mode="json") for item in references],
    )
    case = FormalCase(
        **{
            **case.__dict__,
            "license_sha256": license_digests,
        }
    )
    with pytest.raises(WikimediaKBError, match="CC BY-SA"):
        _build(case, tmp_path / "output")


def test_review_ledger_row_must_bind_exact_span_and_reviewer(tmp_path: Path) -> None:
    case = _make_formal_case(tmp_path)
    review_path = case.evidence / case.lock.review_record_file
    ledger = WikimediaReviewLedger.model_validate(json.loads(review_path.read_bytes()))
    changed = ledger.decisions[0].model_copy(
        update={"evidence_start_char": 1, "evidence_end_char": 5}
    )
    changed_ledger = WikimediaReviewLedger(
        selection_id=ledger.selection_id,
        decisions=(changed,) + ledger.decisions[1:],
    )
    content = canonical_json_document(changed_ledger)
    review_path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    case = _replace_lock(
        case,
        review_record_uri=f"urn:sha256:{digest}",
        review_record_sha256=digest,
    )
    case = FormalCase(**{**case.__dict__, "review_sha256": digest})
    with pytest.raises(WikimediaKBError, match="row does not match"):
        _build(case, tmp_path / "output")

    payload = ledger.decisions[0].model_dump(mode="json")
    payload["reviewer_id"] = "gpt-reviewer"
    with pytest.raises(ValidationError, match="LLM cannot approve"):
        WikimediaReviewDecision.model_validate(payload)


def test_exact_input_sets_and_all_link_types_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra_case = _make_formal_case(tmp_path / "extra")
    (extra_case.evidence / "unreviewed.txt").write_text("extra")
    with pytest.raises(WikimediaKBError, match="artifact set"):
        _build(extra_case, tmp_path / "extra-output")

    hardlink_case = _make_formal_case(tmp_path / "hardlink")
    selected = hardlink_case.lock.entries[0]
    alias = tmp_path / "raw-hardlink-alias"
    try:
        os.link(hardlink_case.snapshots / selected.raw_response_file, alias)
    except OSError:
        pytest.skip("hardlink creation unavailable")
    with pytest.raises(WikimediaKBError, match="hard-linked"):
        _build(hardlink_case, tmp_path / "hardlink-output")

    reparse_case = _make_formal_case(tmp_path / "reparse")
    original = wikimedia_kb._path_is_link_or_junction
    target = (reparse_case.evidence / reparse_case.lock.review_record_file).absolute()

    def report_reparse(path: Path, metadata: os.stat_result) -> bool:
        return path.absolute() == target or original(path, metadata)

    monkeypatch.setattr(wikimedia_kb, "_path_is_link_or_junction", report_reparse)
    with pytest.raises(WikimediaKBError, match="regular non-reparse|regular files"):
        _build(reparse_case, tmp_path / "reparse-output")


def test_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    case = _make_formal_case(tmp_path / "case")
    link = tmp_path / "snapshots-link"
    try:
        link.symlink_to(case.snapshots, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation unavailable")
    with pytest.raises(WikimediaKBError, match="symlink or junction"):
        build_wikimedia_kb_bundle(
            link,
            case.evidence,
            tmp_path / "symlink-output",
            selection_lock_path=case.lock_path,
            expected_selection_lock_sha256=case.lock_sha256,
            expected_license_evidence_sha256=case.license_sha256,
            expected_review_ledger_sha256=case.review_sha256,
        )


def test_ancestor_junction_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_formal_case(tmp_path / "case")
    original = getattr(Path, "is_junction", lambda self: False)
    expected = case.evidence.absolute()

    def fake_is_junction(path: Path) -> bool:
        return path.absolute() == expected or original(path)

    monkeypatch.setattr(Path, "is_junction", fake_is_junction, raising=False)
    with pytest.raises(WikimediaKBError, match="symlink or junction"):
        _build(case, tmp_path / "junction-output")


def test_toctou_failure_prevents_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _make_formal_case(tmp_path)
    original = wikimedia_kb._verify_secure_snapshot

    def reject_receipt(snapshot, label: str):
        if "acquisition input" in label:
            raise WikimediaKBError("changed before publication")
        return original(snapshot, label)

    monkeypatch.setattr(wikimedia_kb, "_verify_secure_snapshot", reject_receipt)
    output = tmp_path / "output"
    with pytest.raises(WikimediaKBError, match="changed before publication"):
        _build(case, output)
    assert not output.exists()


def test_published_bundle_rejects_extra_noncanonical_and_wrong_digest(
    tmp_path: Path,
) -> None:
    case = _make_formal_case(tmp_path)
    output = tmp_path / "bundle"
    report = _build(case, output)
    with pytest.raises(WikimediaKBError, match="external expected digest"):
        load_verified_wikimedia_kb_bundle(output, expected_bundle_sha256="0" * 64)
    (output / "extra.json").write_text("{}")
    with pytest.raises(WikimediaKBError, match="artifact set"):
        load_verified_wikimedia_kb_bundle(
            output, expected_bundle_sha256=report.manifest.bundle_sha256
        )

    canonical_case = _make_formal_case(tmp_path / "canonical")
    canonical_output = tmp_path / "canonical-bundle"
    _build(canonical_case, canonical_output)
    artifact_path = canonical_output / "encyclopedia.jsonl"
    rows = [json.loads(line) for line in artifact_path.read_bytes().splitlines()]
    noncanonical = (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + "\n"
    ).encode("utf-8")
    artifact_path.write_bytes(noncanonical)
    manifest_path = canonical_output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"]["encyclopedia.jsonl"]["bytes"] = len(noncanonical)
    manifest["artifacts"]["encyclopedia.jsonl"]["sha256"] = hashlib.sha256(
        noncanonical
    ).hexdigest()
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    manifest["bundle_sha256"] = hashlib.sha256(
        canonical_json_document(unsigned)
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_document(manifest))
    with pytest.raises(WikimediaKBError, match="canonical JSONL"):
        load_verified_wikimedia_kb_bundle(
            canonical_output,
            expected_bundle_sha256=manifest["bundle_sha256"],
        )


def test_noncanonical_selection_and_review_files_fail_closed(tmp_path: Path) -> None:
    case = _make_formal_case(tmp_path)
    pretty_lock = json.dumps(case.lock.model_dump(mode="json"), indent=2).encode()
    case.lock_path.write_bytes(pretty_lock)
    with pytest.raises(WikimediaKBError, match="canonical JSON"):
        load_verified_selection_lock(
            case.lock_path,
            expected_selection_lock_sha256=hashlib.sha256(pretty_lock).hexdigest(),
        )

    review_case = _make_formal_case(tmp_path / "review")
    review_path = review_case.evidence / review_case.lock.review_record_file
    parsed = json.loads(review_path.read_bytes())
    pretty_review = json.dumps(parsed, indent=2).encode()
    review_path.write_bytes(pretty_review)
    digest = hashlib.sha256(pretty_review).hexdigest()
    review_case = _replace_lock(
        review_case,
        review_record_uri=f"urn:sha256:{digest}",
        review_record_sha256=digest,
    )
    review_case = FormalCase(**{**review_case.__dict__, "review_sha256": digest})
    with pytest.raises(WikimediaKBError, match="canonical JSON"):
        _build(review_case, tmp_path / "review-output")
