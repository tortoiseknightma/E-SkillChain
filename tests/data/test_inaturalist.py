import io
import json

from PIL import Image, ImageDraw

import requests

from skillchain.data.inaturalist import (
    _get_json,
    extract_candidates,
    materialize,
    normalize_candidate,
    normalize_candidates,
    repair_existing_collection,
    refresh_existing_manifest,
)


def _photo_bytes(pattern="left"):
    image = Image.new("RGB", (500, 400), "green")
    draw = ImageDraw.Draw(image)
    if pattern == "left":
        draw.rectangle((0, 0, 150, 400), fill="black")
    else:
        draw.rectangle((0, 0, 500, 150), fill="white")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_extract_candidates_keeps_licensed_photo_and_taxon_metadata():
    payload = {
        "results": [
            {
                "id": 55,
                "quality_grade": "research",
                "taxon": {
                    "name": "Ribes californicum",
                    "preferred_common_name": "hillside gooseberry",
                    "iconic_taxon_name": "Plantae",
                },
                "photos": [
                    {
                        "id": 16,
                        "license_code": "cc-by",
                        "original_dimensions": {"width": 2048, "height": 2048},
                        "url": (
                            "https://inaturalist-open-data.s3.amazonaws.com/"
                            "photos/16/square.jpg"
                        ),
                        "attribution": "(c) Ken, CC BY",
                    }
                ],
            },
            {
                "id": 56,
                "quality_grade": "research",
                "taxon": {"name": "Hidden species", "iconic_taxon_name": "Animalia"},
                "photos": [
                    {
                        "id": 17,
                        "license_code": "cc-by-nc",
                        "original_dimensions": {"width": 1000, "height": 1000},
                        "url": "https://example.test/photos/17/square.jpg",
                        "attribution": "non-commercial",
                    }
                ],
            },
            {
                "id": 57,
                "quality_grade": "research",
                "taxon": {"name": "Unknown dimensions", "iconic_taxon_name": "Fungi"},
                "photos": [
                    {
                        "id": 18,
                        "license_code": "cc0",
                        "original_dimensions": {"width": None, "height": None},
                        "url": "https://example.test/photos/18/square.jpg",
                        "attribution": "public domain",
                    }
                ],
            },
            {
                "id": 58,
                "quality_grade": "research",
                "taxon": {"name": "Wrong host", "iconic_taxon_name": "Fungi"},
                "photos": [
                    {
                        "id": 19,
                        "license_code": "cc-by",
                        "original_dimensions": {"width": 1000, "height": 1000},
                        "url": "https://static.inaturalist.org/photos/19/square.jpg",
                        "attribution": "claimed open license on a non-open-data host",
                    }
                ],
            },
            {
                "id": 59,
                "quality_grade": "research",
                "taxon": {"name": "Missing attribution", "iconic_taxon_name": "Plantae"},
                "photos": [
                    {
                        "id": 20,
                        "license_code": "cc-by",
                        "original_dimensions": {"width": 1000, "height": 1000},
                        "url": (
                            "https://inaturalist-open-data.s3.amazonaws.com/"
                            "photos/20/square.jpg"
                        ),
                        "attribution": "",
                    }
                ],
            },
        ]
    }

    candidates = extract_candidates(payload)

    assert candidates == [
        {
            "observation_id": 55,
            "photo_id": 16,
            "scientific_name": "Ribes californicum",
            "common_name": "hillside gooseberry",
            "iconic_taxon": "Plantae",
            "license": "cc-by",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "(c) Ken, CC BY",
            "observation_url": "https://www.inaturalist.org/observations/55",
            "api_photo_url": (
                "https://inaturalist-open-data.s3.amazonaws.com/photos/16/square.jpg"
            ),
            "original_url": (
                "https://inaturalist-open-data.s3.amazonaws.com/photos/16/original.jpg"
            ),
            "source_url": (
                "https://inaturalist-open-data.s3.amazonaws.com/photos/16/medium.jpg"
            ),
        }
    ]


def test_materialize_writes_images_and_license_manifest_atomically(tmp_path):
    candidates = [
        {
            "observation_id": 1,
            "photo_id": 10,
            "scientific_name": "Species one",
            "common_name": "one",
            "iconic_taxon": "Plantae",
            "license": "cc0",
            "attribution": "public domain",
            "source_url": "https://example.test/10.jpg",
        },
        {
            "observation_id": 2,
            "photo_id": 20,
            "scientific_name": "Species two",
            "common_name": None,
            "iconic_taxon": "Animalia",
            "license": "cc-by",
            "attribution": "author, CC BY",
            "source_url": "https://example.test/20.jpg",
        },
    ]
    payloads = {10: _photo_bytes("left"), 20: _photo_bytes("top")}

    report = materialize(
        candidates,
        tmp_path / "encyclopedia",
        limit=2,
        workers=1,
        fetcher=lambda candidate: payloads[candidate["photo_id"]],
    )

    assert report.kept == 2
    assert (tmp_path / "encyclopedia" / "inat-10.jpg").is_file()
    manifest = [
        json.loads(line)
        for line in (tmp_path / "encyclopedia" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["photo_id"] for row in manifest] == [10, 20]
    assert manifest[1]["license"] == "cc-by"


def test_get_json_retries_transient_transport_failures():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    class Session:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise requests.exceptions.SSLError("transient EOF")
            return Response()

    session = Session()
    sleeps = []

    payload = _get_json(
        session,
        "https://example.test/api",
        params={},
        attempts=3,
        sleep=lambda seconds: sleeps.append(seconds),
    )

    assert payload == {"results": []}
    assert session.calls == 3
    assert sleeps == [1, 2]


def test_materialize_stops_fetching_after_limit_plus_current_worker_batch(tmp_path):
    candidates = [
        {
            "observation_id": index,
            "photo_id": index,
            "scientific_name": f"Species {index}",
            "common_name": None,
            "iconic_taxon": "Plantae",
            "license": "cc0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "attribution": "public domain",
            "observation_url": f"https://www.inaturalist.org/observations/{index}",
            "api_photo_url": (
                "https://inaturalist-open-data.s3.amazonaws.com/"
                f"photos/{index}/square.jpg"
            ),
            "original_url": (
                "https://inaturalist-open-data.s3.amazonaws.com/"
                f"photos/{index}/original.jpg"
            ),
            "source_url": (
                "https://inaturalist-open-data.s3.amazonaws.com/"
                f"photos/{index}/medium.jpg"
            ),
        }
        for index in range(1, 11)
    ]
    calls = []

    def fetch(candidate):
        calls.append(candidate["photo_id"])
        return _photo_bytes("left" if candidate["photo_id"] % 2 else "top")

    report = materialize(
        candidates,
        tmp_path / "encyclopedia",
        limit=2,
        workers=2,
        fetcher=fetch,
    )

    assert report.kept == 2
    assert calls == [1, 2]


def test_normalize_candidate_upgrades_existing_open_data_metadata():
    upgraded = normalize_candidate(
        {
            "observation_id": 55,
            "photo_id": 16,
            "license": "cc-by",
            "attribution": "author, CC BY",
            "source_url": (
                "https://inaturalist-open-data.s3.amazonaws.com/photos/16/medium.jpg"
            ),
        }
    )

    assert upgraded["observation_url"].endswith("/observations/55")
    assert upgraded["api_photo_url"].endswith("/photos/16/square.jpg")
    assert upgraded["original_url"].endswith("/photos/16/original.jpg")
    assert upgraded["license_url"] == "https://creativecommons.org/licenses/by/4.0/"


def test_normalize_candidate_rejects_attribution_license_without_attribution():
    candidate = {
        "observation_id": 55,
        "photo_id": 16,
        "license": "cc-by-sa",
        "attribution": "  ",
        "source_url": (
            "https://inaturalist-open-data.s3.amazonaws.com/photos/16/medium.jpg"
        ),
    }

    try:
        normalize_candidate(candidate)
    except ValueError as error:
        assert "署名" in str(error)
    else:
        raise AssertionError("CC BY-SA 缺少署名时应拒绝")


def test_normalize_candidates_rejects_non_open_data_rows_without_aborting_batch():
    valid = {
        "observation_id": 55,
        "photo_id": 16,
        "license": "cc-by",
        "attribution": "author, CC BY",
        "source_url": (
            "https://inaturalist-open-data.s3.amazonaws.com/photos/16/medium.jpg"
        ),
    }
    invalid = {
        "observation_id": 56,
        "photo_id": 17,
        "license": "cc0",
        "source_url": "https://static.inaturalist.org/photos/17/medium.jpg",
    }

    accepted, rejected = normalize_candidates([valid, invalid])

    assert [row["photo_id"] for row in accepted] == [16]
    assert rejected == 1


def test_refresh_existing_manifest_adds_audit_fields_without_network(tmp_path):
    destination = tmp_path / "encyclopedia"
    destination.mkdir()
    (destination / "inat-16.jpg").write_bytes(_photo_bytes())
    old_row = {
        "observation_id": 55,
        "photo_id": 16,
        "license": "cc-by",
        "attribution": "author, CC BY",
        "source_url": (
            "https://inaturalist-open-data.s3.amazonaws.com/photos/16/medium.jpg"
        ),
        "image": "inat-16.jpg",
    }
    (destination / "manifest.jsonl").write_text(
        json.dumps(old_row) + "\n", encoding="utf-8"
    )

    refreshed = refresh_existing_manifest(destination, [old_row])

    assert refreshed == 1
    row = json.loads((destination / "manifest.jsonl").read_text(encoding="utf-8"))
    assert row["observation_url"].endswith("/observations/55")
    assert row["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert (destination / "inat-16.jpg").is_file()


def test_repair_existing_collection_drops_invalid_host_and_backfills(tmp_path):
    destination = tmp_path / "encyclopedia"
    destination.mkdir()
    (destination / "inat-16.jpg").write_bytes(_photo_bytes("left"))
    (destination / "inat-17.jpg").write_bytes(_photo_bytes("top"))
    existing = [
        {
            "observation_id": 55,
            "photo_id": 16,
            "license": "cc-by",
            "attribution": "author, CC BY",
            "source_url": (
                "https://inaturalist-open-data.s3.amazonaws.com/photos/16/medium.jpg"
            ),
            "image": "inat-16.jpg",
        },
        {
            "observation_id": 56,
            "photo_id": 17,
            "license": "cc0",
            "source_url": "https://static.inaturalist.org/photos/17/medium.jpg",
            "image": "inat-17.jpg",
        },
    ]
    (destination / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in existing), encoding="utf-8"
    )
    fallback = {
        "observation_id": 57,
        "photo_id": 18,
        "license": "cc0",
        "source_url": (
            "https://inaturalist-open-data.s3.amazonaws.com/photos/18/medium.jpg"
        ),
    }

    report = repair_existing_collection(
        destination,
        [*existing, fallback],
        workers=1,
        fetcher=lambda candidate: _photo_bytes("top"),
    )

    assert report.kept == 2
    rows = [
        json.loads(line)
        for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["photo_id"] for row in rows] == [16, 18]
    assert all("license_url" in row for row in rows)
