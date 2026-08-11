import io
import json

from PIL import Image, ImageDraw

from skillchain.data.wikimedia_documents import (
    extract_candidates,
    materialize,
    repair_existing_collection,
)


def _document_bytes(index: int) -> bytes:
    image = Image.new("RGB", (500, 700), "white")
    draw = ImageDraw.Draw(image)
    for line in range(10 + index):
        y = 30 + line * 35
        draw.rectangle((30 + index * 3, y, 300 + line * 5, y + 8), fill="black")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_extract_candidates_keeps_open_license_and_rejects_gpl():
    payload = {
        "query": {
            "pages": [
                {
                    "pageid": 7,
                    "title": "File:Receipt.jpg",
                    "imageinfo": [
                        {
                            "width": 1200,
                            "height": 1800,
                            "thumbwidth": 800,
                            "thumbheight": 1200,
                            "url": "https://upload.wikimedia.org/original.jpg",
                            "thumburl": "https://upload.wikimedia.org/thumb.jpg",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Receipt.jpg",
                            "mime": "image/jpeg",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY 4.0"},
                                "LicenseUrl": {
                                    "value": "https://creativecommons.org/licenses/by/4.0/"
                                },
                                "Artist": {"value": '<a href="/wiki/User:A">Alice</a>'},
                            },
                        }
                    ],
                },
                {
                    "pageid": 8,
                    "title": "File:Scanner.png",
                    "imageinfo": [
                        {
                            "width": 1200,
                            "height": 800,
                            "thumburl": "https://upload.wikimedia.org/scanner.png",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Scanner.png",
                            "mime": "image/png",
                            "extmetadata": {
                                "LicenseShortName": {"value": "GPLv3"},
                                "Artist": {"value": "Bob"},
                            },
                        }
                    ],
                },
            ]
        }
    }

    assert extract_candidates(payload, "Receipts") == [
        {
            "page_id": 7,
            "title": "File:Receipt.jpg",
            "document_category": "Receipts",
            "width": 800,
            "height": 1200,
            "mime": "image/jpeg",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Alice",
            "description_url": "https://commons.wikimedia.org/wiki/File:Receipt.jpg",
            "original_url": "https://upload.wikimedia.org/original.jpg",
            "source_url": "https://upload.wikimedia.org/thumb.jpg",
        }
    ]


def test_materialize_balances_categories_and_writes_license_manifest(tmp_path):
    candidates = []
    payloads = {}
    page_id = 1
    for category in ("Receipts", "Letters"):
        for item in range(2):
            candidates.append(
                {
                    "page_id": page_id,
                    "title": f"File:{page_id}.jpg",
                    "document_category": category,
                    "width": 500,
                    "height": 700,
                    "mime": "image/jpeg",
                    "license": "CC0",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "attribution": "author",
                    "description_url": f"https://commons.wikimedia.org/wiki/File:{page_id}.jpg",
                    "original_url": f"https://upload.wikimedia.org/{page_id}.jpg",
                    "source_url": f"https://upload.wikimedia.org/thumb/{page_id}.jpg",
                }
            )
            payloads[page_id] = _document_bytes(page_id)
            page_id += 1

    report = materialize(
        candidates,
        tmp_path / "utility_docs",
        limit=4,
        workers=2,
        fetcher=lambda candidate: payloads[candidate["page_id"]],
    )

    assert report.kept == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "utility_docs" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["document_category"] for row in rows] == [
        "Letters",
        "Receipts",
        "Letters",
        "Receipts",
    ]
    assert all(row["license"] == "CC0" for row in rows)
    assert len(list((tmp_path / "utility_docs").glob("commons-*.jpg"))) == 4


def test_extract_candidates_rejects_fake_license_wrong_uri_and_blank_cc_attribution():
    pages = []
    for page_id, license_name, license_url, artist in (
        (1, "CC0foo", "https://creativecommons.org/publicdomain/zero/1.0/", "Alice"),
        (2, "CC BY 4.0", "https://example.test/not-a-license", "Alice"),
        (3, "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/", " "),
    ):
        pages.append(
            {
                "pageid": page_id,
                "title": f"File:{page_id}.jpg",
                "imageinfo": [
                    {
                        "width": 800,
                        "height": 1000,
                        "url": f"https://upload.wikimedia.org/{page_id}.jpg",
                        "descriptionurl": (
                            f"https://commons.wikimedia.org/wiki/File:{page_id}.jpg"
                        ),
                        "mime": "image/jpeg",
                        "extmetadata": {
                            "LicenseShortName": {"value": license_name},
                            "LicenseUrl": {"value": license_url},
                            "Artist": {"value": artist},
                        },
                    }
                ],
            }
        )

    assert extract_candidates({"query": {"pages": pages}}, "Documents") == []


def test_repair_existing_collection_backfills_invalid_provenance(tmp_path):
    destination = tmp_path / "utility_docs"
    destination.mkdir()

    def row(page_id, *, license_name="CC0", attribution="author"):
        return {
            "page_id": page_id,
            "title": f"File:{page_id}.jpg",
            "document_category": "Documents",
            "width": 500,
            "height": 700,
            "mime": "image/jpeg",
            "license": license_name,
            "license_url": (
                "https://creativecommons.org/publicdomain/zero/1.0/"
                if license_name == "CC0"
                else "https://creativecommons.org/licenses/by/4.0/"
            ),
            "attribution": attribution,
            "description_url": f"https://commons.wikimedia.org/wiki/File:{page_id}.jpg",
            "original_url": f"https://upload.wikimedia.org/{page_id}.jpg",
            "source_url": f"https://upload.wikimedia.org/thumb/{page_id}.jpg",
            "image": f"commons-{page_id}.jpg",
            "source": "wikimedia-commons",
        }

    existing = [row(1), row(2, license_name="CC BY 4.0", attribution="")]
    for item in existing:
        (destination / item["image"]).write_bytes(_document_bytes(item["page_id"]))
    (destination / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in existing), encoding="utf-8"
    )
    fallback = row(3)

    report = repair_existing_collection(
        destination,
        [*existing, fallback],
        workers=1,
        fetcher=lambda candidate: _document_bytes(candidate["page_id"]),
    )

    assert report.kept == 2
    rows = [
        json.loads(line)
        for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["page_id"] for item in rows] == [1, 3]
