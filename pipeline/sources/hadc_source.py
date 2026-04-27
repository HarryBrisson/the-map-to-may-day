from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from utils.ids import stable_source_id
from utils.s3_storage import JsonStorage


TOC_URL = "https://www.chicagohistoryresources.org/hadc/transcript/trialtoc.htm#OUTLINE"
REQUEST_HEADERS = {
    "User-Agent": "the-map-to-may-day/0.1 (+https://github.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TEST_URLS = [
    TOC_URL,
    "https://www.chicagohistoryresources.org/hadc/transcript/volumei/000-050/I019-052.htm",
    "https://www.chicagohistoryresources.org/hadc/transcript/exhibits/X000-050/X0010.htm",
    "https://www.chicagohistoryresources.org/hadc/transcript/volumen/000-050/N017-105.htm",
]

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("", TEI_NS)


@dataclass(frozen=True)
class SourcePage:
    id: str
    url: str
    title: str
    source_type: str
    fetched_at: str
    raw_html: str
    raw_html_sha256: str
    candidate_text: str
    candidate_text_sha256: str
    tei_xml: str
    transcript_index: dict[str, Any]
    text: str
    links: list[dict[str, str]]
    page_images: list[str]
    page_cues: list[dict[str, Any]]
    toc_entries: list[dict[str, Any]]
    transcript_metadata: dict[str, Any]
    source_stats: dict[str, Any]

    def to_dict(self, paths: dict[str, str]) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "source_type": self.source_type,
            "fetched_at": self.fetched_at,
            "raw_html_sha256": self.raw_html_sha256,
            "candidate_text_sha256": self.candidate_text_sha256,
            "raw_html_path": paths["raw_html"],
            "candidate_text_path": paths["candidate_text"],
            "tei_path": paths["tei"],
            "transcript_json_path": paths["transcript_json"],
            "text": self.candidate_text,
            "links": self.links,
            "page_images": self.page_images,
            "page_cues": self.page_cues,
            "toc_entries": self.toc_entries,
            "transcript_metadata": self.transcript_metadata,
            "source_stats": self.source_stats,
        }


def pull_corpus(corpus: str, storage: JsonStorage, run_id: str) -> list[dict[str, Any]]:
    urls = TEST_URLS if corpus == "test" else discover_full_corpus_urls()
    pages = [fetch_and_normalize(url) for url in dedupe(urls)]
    page_dicts = []
    for page in pages:
        paths = transcript_artifact_paths(run_id, page.id)
        storage.write_text(paths["raw_html"], page.raw_html, "text/html; charset=utf-8")
        storage.write_text(paths["candidate_text"], page.candidate_text)
        storage.write_text(paths["tei"], page.tei_xml, "application/tei+xml; charset=utf-8")
        storage.write_json(paths["transcript_json"], page.transcript_index)
        page_dicts.append(page.to_dict(paths))
        print(
            "Pulled "
            f"{page.id}: raw_html={len(page.raw_html)} chars, "
            f"candidate_text={len(page.candidate_text)} chars, "
            f"lines={page.source_stats['lines']}, "
            f"page_markers={page.source_stats['page_markers']}, "
            f"navigation_removed={page.source_stats.get('navigation_removed', 0)}"
        )

    storage.write_json(f"raw/haymarket/hadc/{run_id}/pages.json", page_dicts)
    storage.write_json(
        "raw/haymarket/hadc/latest_run.json",
        {
            "run_id": run_id,
            "corpus": corpus,
            "page_count": len(page_dicts),
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return page_dicts


def transcript_artifact_paths(run_id: str, source_id: str) -> dict[str, str]:
    return {
        "raw_html": f"raw/haymarket/hadc/{run_id}/html/{source_id}.html",
        "candidate_text": f"raw/haymarket/hadc/{run_id}/text/{source_id}.txt",
        "tei": f"raw/haymarket/hadc/{run_id}/tei/{source_id}.xml",
        "transcript_json": f"raw/haymarket/hadc/{run_id}/transcripts/{source_id}.json",
    }


def discover_full_corpus_urls() -> list[str]:
    toc = fetch_and_normalize(TOC_URL)
    urls = [TOC_URL]
    for link in toc.links:
        href = link["url"]
        if "/hadc/transcript/" in href and href.endswith(".htm"):
            urls.append(href)
    return dedupe(urls)


def fetch_and_normalize(url: str) -> SourcePage:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title = extract_title(soup)
    raw_text = normalize_text(soup.get_text("\n"))
    text, candidate_stats = extract_candidate_core_text(soup)
    links = extract_links(soup, url)
    page_images = [link["url"] for link in links if "Image" in link["text"] or re.search(r"/\d+[A-Z]?\d*\.htm$", link["url"])]
    page_markers = extract_page_markers(text)
    page_cues = build_page_cues(page_markers, page_images)
    content_blocks = extract_content_blocks(text)
    toc_entries = extract_toc_entries(links, text, url)
    source_type = classify_source(url, title, f"{raw_text}\n{text}")
    raw_html_sha256 = hashlib.sha256(response.text.encode("utf-8")).hexdigest()
    candidate_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    fetched_at = datetime.now(timezone.utc).isoformat()
    metadata = extract_metadata(title, text, url)
    source_id = stable_source_id(url)
    tei_xml = build_tei_transcript(
        source_id=source_id,
        url=url,
        title=title,
        source_type=source_type,
        fetched_at=fetched_at,
        raw_html_sha256=raw_html_sha256,
        text=text,
        page_images=page_images,
    )
    transcript_index = tei_to_transcript_json(
        source_id=source_id,
        url=url,
        title=title,
        source_type=source_type,
        fetched_at=fetched_at,
        tei_xml=tei_xml,
        transcript_metadata=metadata,
    )

    return SourcePage(
        id=source_id,
        url=url,
        title=title,
        source_type=source_type,
        fetched_at=fetched_at,
        raw_html=response.text,
        raw_html_sha256=raw_html_sha256,
        candidate_text=text,
        candidate_text_sha256=candidate_text_sha256,
        tei_xml=tei_xml,
        transcript_index=transcript_index,
        text=text,
        links=links,
        page_images=page_images,
        page_cues=page_cues,
        toc_entries=toc_entries if source_type == "toc" else [],
        transcript_metadata=metadata,
        source_stats=build_source_stats(text, links, page_markers, content_blocks, transcript_index, candidate_stats),
    )


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return normalize_text(soup.title.string)
    first_text = soup.get_text("\n").strip().splitlines()
    return normalize_text(first_text[0]) if first_text else "Untitled HADC page"


def extract_links(soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        links.append({"text": normalize_text(anchor.get_text(" ")), "url": urljoin(base_url, href)})
    return links


def normalize_text(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_candidate_core_text(soup: BeautifulSoup) -> tuple[str, dict[str, int]]:
    clone = BeautifulSoup(str(soup), "html.parser")
    for element in clone(["script", "style", "noscript"]):
        element.decompose()

    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in clone.get_text("\n").splitlines()]
    lines: list[str] = []
    navigation_removed = 0
    for line in raw_lines:
        if not line:
            continue
        if is_navigation_line(line):
            navigation_removed += 1
            continue
        lines.append(line)

    return "\n".join(lines), {"navigation_removed": navigation_removed, "raw_text_lines": len([line for line in raw_lines if line])}


def is_navigation_line(line: str) -> bool:
    normalized = line.strip().lower()
    if normalized in {"|", "back to top"}:
        return True
    navigation_prefixes = (
        "go to next",
        "return to previous",
        "return to trial toc",
        "return to the hadc table of contents",
        "return to top",
        "return to hadc table of contents",
    )
    return any(normalized.startswith(prefix) for prefix in navigation_prefixes)


def extract_page_markers(text: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for line_index, line in enumerate(text.splitlines()):
        match = re.match(r"\[Image,\s*(.+?)\]", line)
        if not match:
            continue
        markers.append(
            {
                "index": len(markers),
                "line_index": line_index,
                "label": match.group(1),
                "page_ref": extract_page_ref(match.group(1)),
            }
        )
    return markers


def extract_page_ref(label: str) -> str | None:
    page_match = re.search(r"Page\s+([A-Z]?\d+|[ivxlcdm]+)", label, re.IGNORECASE)
    return page_match.group(1) if page_match else None


def build_page_cues(page_markers: list[dict[str, Any]], page_images: list[str]) -> list[dict[str, Any]]:
    cues = []
    for marker in page_markers:
        index = marker["index"]
        cues.append(
            {
                "index": index,
                "line_index": marker["line_index"],
                "label": marker["label"],
                "page_ref": marker.get("page_ref"),
                "facs": page_images[index] if index < len(page_images) else None,
            }
        )
    return cues


def extract_content_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current_page_ref: str | None = None

    for line_index, line in enumerate(text.splitlines()):
        marker_match = re.match(r"\[Image,\s*(.+?)\]", line)
        if marker_match:
            current_page_ref = extract_page_ref(marker_match.group(1))
            blocks.append(
                {
                    "index": len(blocks),
                    "kind": "page_marker",
                    "page_ref": current_page_ref,
                    "line_start": line_index,
                    "line_end": line_index,
                    "text": line,
                }
            )
            continue

        kind = classify_line(line)
        blocks.append(
            {
                "index": len(blocks),
                "kind": kind,
                "page_ref": current_page_ref,
                "line_start": line_index,
                "line_end": line_index,
                "text": line,
            }
        )
    return blocks


def classify_line(line: str) -> str:
    if re.match(r"^Q[\.\s]", line):
        return "question"
    if re.match(r"^A[\.\s]", line):
        return "answer"
    if re.match(r"^(MR|Mr|THE COURT|WITNESS|A JUROR)[:\.\s]", line):
        return "speaker"
    if line.startswith("[") and line.endswith("]"):
        return "reference"
    if re.match(r"^[-*]\s+\[", line):
        return "toc_entry"
    return "text"


def tei_tag(name: str) -> str:
    return f"{{{TEI_NS}}}{name}"


def build_tei_transcript(
    source_id: str,
    url: str,
    title: str,
    source_type: str,
    fetched_at: str,
    raw_html_sha256: str,
    text: str,
    page_images: list[str],
) -> str:
    root = ET.Element(tei_tag("TEI"), {f"{{{XML_NS}}}id": source_id})
    header = ET.SubElement(root, tei_tag("teiHeader"))
    file_desc = ET.SubElement(header, tei_tag("fileDesc"))
    title_stmt = ET.SubElement(file_desc, tei_tag("titleStmt"))
    ET.SubElement(title_stmt, tei_tag("title")).text = title
    publication_stmt = ET.SubElement(file_desc, tei_tag("publicationStmt"))
    ET.SubElement(publication_stmt, tei_tag("p")).text = "Generated by The Map to May Day pipeline."
    source_desc = ET.SubElement(file_desc, tei_tag("sourceDesc"))
    ET.SubElement(source_desc, tei_tag("p")).text = f"Original HADC URL: {url}"

    encoding_desc = ET.SubElement(header, tei_tag("encodingDesc"))
    ET.SubElement(encoding_desc, tei_tag("p")).text = (
        "Small local TEI profile generated from normalized HADC transcript text; "
        f"source_type={source_type}; fetched_at={fetched_at}; raw_html_sha256={raw_html_sha256}."
    )

    text_el = ET.SubElement(root, tei_tag("text"))
    body = ET.SubElement(text_el, tei_tag("body"))
    div = ET.SubElement(body, tei_tag("div"), {"type": source_type})

    image_index = 0
    for line_index, line in enumerate(text.splitlines()):
        marker_match = re.match(r"\[Image,\s*(.+?)\]", line)
        if marker_match:
            attrs = {"n": extract_page_ref(marker_match.group(1)) or marker_match.group(1), "ana": "page_marker"}
            if image_index < len(page_images):
                attrs["facs"] = page_images[image_index]
            image_index += 1
            ET.SubElement(div, tei_tag("pb"), attrs)
            continue

        kind = classify_line(line)
        if kind in {"question", "answer"}:
            add_speech_element(div, line, line_index, kind)
        elif kind == "speaker":
            add_speaker_element(div, line, line_index)
        else:
            attrs = {"n": str(line_index)}
            if kind != "text":
                attrs["type"] = kind
            ET.SubElement(div, tei_tag("p"), attrs).text = line

    return serialize_xml(root)


def add_speech_element(parent: ET.Element, line: str, line_index: int, kind: str) -> None:
    match = re.match(r"^((?:Q|A)\.?)\s*(.*)$", line)
    speaker_text = match.group(1) if match else ("Q." if kind == "question" else "A.")
    body_text = (match.group(2) if match else line).strip()
    sp = ET.SubElement(parent, tei_tag("sp"), {"n": str(line_index), "type": kind})
    ET.SubElement(sp, tei_tag("speaker")).text = speaker_text
    ET.SubElement(sp, tei_tag("p")).text = body_text


def add_speaker_element(parent: ET.Element, line: str, line_index: int) -> None:
    match = re.match(r"^([^:]+:)\s*(.*)$", line)
    if not match:
        ET.SubElement(parent, tei_tag("p"), {"n": str(line_index), "type": "speaker"}).text = line
        return

    sp = ET.SubElement(parent, tei_tag("sp"), {"n": str(line_index), "type": "speaker"})
    ET.SubElement(sp, tei_tag("speaker")).text = match.group(1)
    ET.SubElement(sp, tei_tag("p")).text = match.group(2).strip()


def tei_to_transcript_json(
    source_id: str,
    url: str,
    title: str,
    source_type: str,
    fetched_at: str,
    tei_xml: str,
    transcript_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = ET.fromstring(tei_xml)
    body = root.find(f".//{tei_tag('body')}")
    segments: list[dict[str, Any]] = []
    page_refs: list[dict[str, Any]] = []
    inline_mentions: list[dict[str, Any]] = []
    span_offsets: dict[str, dict[str, Any]] = {}
    text_parts: list[str] = []
    current_page_ref: str | None = None

    def append_segment(kind: str, value: str, attrs: dict[str, str], mentions: list[dict[str, Any]] | None = None) -> None:
        nonlocal current_page_ref
        value = re.sub(r"\s+", " ", value).strip()
        if not value:
            return
        if text_parts:
            text_parts.append("\n")
        start = sum(len(part) for part in text_parts)
        text_parts.append(value)
        end = start + len(value)
        segments.append(
            {
                "index": len(segments),
                "kind": kind,
                "page_ref": current_page_ref,
                "line": int(attrs["n"]) if attrs.get("n", "").isdigit() else None,
                "start": start,
                "end": end,
                "text": value,
                "speaker_id": strip_ref(attrs.get("who")),
            }
        )
        for mention in mentions or []:
            global_mention = dict(mention)
            global_mention["start"] = start + int(mention["start"])
            global_mention["end"] = start + int(mention["end"])
            inline_mentions.append(global_mention)
            if global_mention.get("id"):
                span_offsets[str(global_mention["id"])] = global_mention

    def walk(element: ET.Element) -> None:
        nonlocal current_page_ref
        for child in list(element):
            local_name = child.tag.split("}", 1)[-1]
            if local_name == "pb":
                current_page_ref = child.attrib.get("n")
                page_refs.append({"index": len(page_refs), "page_ref": current_page_ref, "offset": sum(len(part) for part in text_parts), "facs": child.attrib.get("facs")})
            elif local_name == "sp":
                speaker = child.findtext(tei_tag("speaker")) or ""
                speech_parts = []
                speech_mentions: list[dict[str, Any]] = []
                cursor = len(speaker.strip()) + 1 if speaker.strip() else 0
                for part in child.findall(tei_tag("p")):
                    part_text, part_mentions = collect_inline_text(part)
                    if part_text:
                        speech_parts.append(part_text)
                        for mention in part_mentions:
                            adjusted = dict(mention)
                            adjusted["start"] = cursor + int(mention["start"])
                            adjusted["end"] = cursor + int(mention["end"])
                            speech_mentions.append(adjusted)
                        cursor += len(part_text) + 1
                kind = child.attrib.get("type", "speech")
                append_segment(kind, " ".join([speaker, *speech_parts]), child.attrib, speech_mentions)
            elif local_name == "p":
                value, mentions = collect_inline_text(child)
                append_segment(child.attrib.get("type", "text"), value, child.attrib, mentions)
            else:
                walk(child)

    if body is not None:
        walk(body)

    plain_text = "".join(text_parts)
    mentions = merge_mentions(inline_mentions, extract_standoff_mentions(root, span_offsets))
    return {
        "source_id": source_id,
        "url": url,
        "title": title,
        "source_type": source_type,
        "fetched_at": fetched_at,
        "text": plain_text,
        "page_refs": page_refs,
        "segments": segments,
        "mentions": mentions,
        "transcript_metadata": transcript_metadata or {},
        "source_stats": {
            "characters": len(plain_text),
            "segments": len(segments),
            "page_refs": len(page_refs),
            "mentions": len(mentions),
            "questions": sum(1 for segment in segments if segment["kind"] == "question"),
            "answers": sum(1 for segment in segments if segment["kind"] == "answer"),
        },
    }


def collect_inline_text(element: ET.Element) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    mentions: list[dict[str, Any]] = []

    def append_text(value: str | None) -> None:
        if value:
            parts.append(value)

    def walk_inline(node: ET.Element) -> None:
        append_text(node.text)
        for child in list(node):
            local_name = child.tag.split("}", 1)[-1]
            start = len("".join(parts))
            walk_inline(child)
            end = len("".join(parts))
            if local_name == "seg" and child.attrib.get("corresp"):
                mention_id = child.attrib.get(f"{{{XML_NS}}}id") or child.attrib.get("id")
                mentions.append(
                    {
                        "id": mention_id,
                        "kind": child.attrib.get("type", "entity"),
                        "entity_id": strip_ref(child.attrib.get("corresp")),
                        "provisional_entity_id": strip_ref(child.attrib.get("ana")),
                        "start": start,
                        "end": end,
                        "text": "".join(parts)[start:end],
                        "confidence": float(child.attrib["cert"]) if child.attrib.get("cert") else None,
                        "source": "llm_tei",
                    }
                )
            append_text(child.tail)

    walk_inline(element)
    text = re.sub(r"\s+", " ", "".join(parts)).strip()
    if not text:
        return "", []

    # Recalculate mention offsets after whitespace normalization.
    normalized_mentions = []
    for mention in mentions:
        mention_text = re.sub(r"\s+", " ", str(mention.get("text") or "")).strip()
        start = text.lower().find(mention_text.lower()) if mention_text else -1
        if start < 0:
            continue
        updated = dict(mention)
        updated["start"] = start
        updated["end"] = start + len(mention_text)
        updated["text"] = text[updated["start"] : updated["end"]]
        normalized_mentions.append(updated)
    return text, normalized_mentions


def strip_ref(value: str | None) -> str | None:
    if not value:
        return None
    return value.lstrip("#")


def extract_standoff_mentions(root: ET.Element, span_offsets: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    span_offsets = span_offsets or {}
    mentions: list[dict[str, Any]] = []
    for annotation in root.findall(f".//{tei_tag('annotation')}"):
        target = annotation.attrib.get("target", "")
        match = re.match(r"#char-(\d+)-(\d+)$", target)
        span = span_offsets.get(strip_ref(target) or "")
        if not match and not span:
            continue
        start = int(match.group(1)) if match else int(span["start"])
        end = int(match.group(2)) if match else int(span["end"])
        mentions.append(
            {
                "id": annotation.attrib.get(f"{{{XML_NS}}}id") or annotation.attrib.get("id"),
                "kind": annotation.attrib.get("type"),
                "entity_id": strip_ref(annotation.attrib.get("corresp")) or (span or {}).get("entity_id"),
                "speaker_id": strip_ref(annotation.attrib.get("resp")) or strip_ref(annotation.attrib.get("source")),
                "start": start,
                "end": end,
                "text": "".join(annotation.itertext()),
                "confidence": float(annotation.attrib["cert"]) if annotation.attrib.get("cert") else None,
                "source": "llm_tei_annotation",
            }
        )
    return mentions


def merge_mentions(*mention_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, int, int]] = set()
    for group in mention_groups:
        for mention in group:
            key = (mention.get("kind"), mention.get("entity_id"), int(mention["start"]), int(mention["end"]))
            if key in seen:
                continue
            seen.add(key)
            merged.append(mention)
    return sorted(merged, key=lambda item: (int(item["start"]), int(item["end"]), str(item.get("kind") or "")))


def add_standoff_annotations_to_tei(tei_xml: str, mentions: list[dict[str, Any]]) -> str:
    root = ET.fromstring(tei_xml)
    for existing in root.findall(tei_tag("standOff")):
        root.remove(existing)
    if not mentions:
        return serialize_xml(root)

    standoff = ET.SubElement(root, tei_tag("standOff"))
    annotation_list = ET.SubElement(standoff, tei_tag("listAnnotation"))
    for index, mention in enumerate(mentions):
        attrs = {
            f"{{{XML_NS}}}id": mention.get("id") or f"mention_{index}",
            "type": str(mention["kind"]),
            "target": f"#char-{mention['start']}-{mention['end']}",
            "corresp": str(mention["entity_id"]),
            "resp": str(mention.get("source") or "llm_extraction"),
        }
        if mention.get("confidence") is not None:
            attrs["cert"] = str(round(float(mention["confidence"]), 4))
        ET.SubElement(annotation_list, tei_tag("annotation"), attrs).text = str(mention.get("text") or "")
    return serialize_xml(root)


def serialize_xml(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def extract_page_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line_index, line in enumerate(text.splitlines()):
        marker_match = re.match(r"\[Image,\s*(.+?)\]", line)
        if marker_match:
            if current:
                current["line_end"] = line_index - 1
                current["text"] = "\n".join(current.pop("_lines"))
                sections.append(current)
            current = {
                "index": len(sections),
                "label": marker_match.group(1),
                "page_ref": extract_page_ref(marker_match.group(1)),
                "line_start": line_index,
                "_lines": [line],
            }
            continue

        if current:
            current["_lines"].append(line)

    if current:
        current["line_end"] = len(text.splitlines()) - 1
        current["text"] = "\n".join(current.pop("_lines"))
        sections.append(current)

    return sections


def extract_toc_entries(links: list[dict[str, str]], text: str, base_url: str) -> list[dict[str, Any]]:
    del text, base_url
    entries = []
    for index, link in enumerate(links):
        if "/hadc/transcript/" not in link["url"] or not link["url"].endswith(".htm"):
            continue
        entries.append(
            {
                "index": index,
                "label": link["text"],
                "url": link["url"],
                "source_id": stable_source_id(link["url"]),
            }
        )
    return entries


def build_source_stats(
    text: str,
    links: list[dict[str, str]],
    page_markers: list[dict[str, Any]],
    content_blocks: list[dict[str, Any]],
    transcript_index: dict[str, Any] | None = None,
    candidate_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    transcript_stats = (transcript_index or {}).get("source_stats", {})
    stats = {
        "characters": len(text),
        "lines": len(text.splitlines()),
        "links": len(links),
        "page_markers": len(page_markers),
        "segments": transcript_stats.get("segments", len(content_blocks)),
        "questions": sum(1 for block in content_blocks if block["kind"] == "question"),
        "answers": sum(1 for block in content_blocks if block["kind"] == "answer"),
    }
    stats.update(candidate_stats or {})
    return stats


def classify_source(url: str, title: str, text: str) -> str:
    lowered_url = url.lower()
    lowered_title = title.lower()
    lowered = f"{url} {title} {text[:500]}".lower()
    if "trialtoc" in lowered_url or "table of contents" in lowered_title:
        return "toc"
    if "/exhibits/" in lowered_url or "exhibit" in lowered_title:
        return "exhibit"
    if "testimony" in lowered or "direct examination" in lowered or "cross-examination" in lowered:
        return "testimony"
    return "transcript"


def extract_metadata(title: str, text: str, url: str) -> dict[str, Any]:
    volume_match = re.search(r"Volume\s+([A-Z]+|[IVXLCDM]+)", text, re.IGNORECASE)
    page_match = re.search(r"(?:pages?|Volume\s+[A-Z]+,)\s+([ivxlcdm\d -]+)", text, re.IGNORECASE)
    date_match = re.search(r"(1886\s+[A-Z][a-z]+\s+\d{1,2}|[A-Z][a-z]+\s+\d{1,2},\s+1886)", text)
    witness_match = re.search(r"Testimony of ([^,\n]+)", title)

    return {
        "volume": volume_match.group(1) if volume_match else None,
        "pages": page_match.group(1).strip() if page_match else None,
        "date_text": date_match.group(1) if date_match else None,
        "witness_name": witness_match.group(1).strip() if witness_match else None,
        "url_path": url.split("/hadc/")[-1],
    }


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
