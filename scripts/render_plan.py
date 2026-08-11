"""Parse checklist progress from plan Markdown."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
import re
import sys
from typing import Sequence

from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin


_PHASE_HEADING = re.compile(r"^Phase\s+(\d+)\s*[:：]\s*(.*?)\s*$")
_TASK_STATE = re.compile(
    r"^\s*(?:>\s*)*(?:[-+*]|\d+[.)])\s+\[([ xX])\]"
)
_MARKDOWN = MarkdownIt().use(tasklists_plugin)
GENERATOR_VERSION = "1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "docs/plans/2026-07-09-skillchain-reproduction.md"


class PlanRenderError(ValueError):
    """Raised when Markdown does not contain a usable plan summary."""


@dataclass(frozen=True)
class PhaseSummary:
    number: int
    title: str
    completed_tasks: int
    total_tasks: int

    @property
    def anchor(self) -> str:
        return f"phase-{self.number}"


@dataclass(frozen=True)
class PlanSummary:
    phases: tuple[PhaseSummary, ...]
    completed_tasks: int
    total_tasks: int
    current_phase_number: int | None


def parse_plan(markdown: str) -> PlanSummary:
    """Summarize GFM checklist tasks grouped under ``## Phase`` headings."""
    source_lines = markdown.splitlines()
    tokens = _MARKDOWN.parse(markdown)
    phase_headers: list[tuple[int, int, str]] = []
    phase_numbers: set[int] = set()
    task_entries: list[tuple[int, bool]] = []

    for index, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h2" and token.map:
            inline = tokens[index + 1]
            phase_match = _PHASE_HEADING.fullmatch(inline.content)
            if phase_match:
                title = phase_match.group(2).strip()
                if not title:
                    raise PlanRenderError(
                        f"Phase {phase_match.group(1)} must have a non-empty title."
                    )
                number = int(phase_match.group(1))
                if number in phase_numbers:
                    raise PlanRenderError(f"Duplicate Phase {number} heading.")
                phase_numbers.add(number)
                phase_headers.append(
                    (token.map[1], number, title)
                )

        if token.type != "inline" or token.map is None or index < 2:
            continue
        list_item = tokens[index - 2]
        if list_item.type != "list_item_open":
            continue
        if list_item.attrGet("class") != "task-list-item":
            continue
        task_state = _TASK_STATE.match(source_lines[token.map[0]])
        if task_state is None:
            continue
        task_entries.append(
            (token.map[0], task_state.group(1).lower() == "x")
        )

    if not phase_headers:
        raise PlanRenderError("Markdown plan must contain at least one Phase heading.")

    phases: list[PhaseSummary] = []
    for index, (start_line, number, title) in enumerate(phase_headers):
        next_start_line = (
            phase_headers[index + 1][0]
            if index + 1 < len(phase_headers)
            else len(markdown.splitlines())
        )
        phase_tasks = [
            completed
            for line, completed in task_entries
            if start_line <= line < next_start_line
        ]
        if not phase_tasks:
            raise PlanRenderError(
                f"Phase {number} must contain at least one checkbox task."
            )
        phases.append(
            PhaseSummary(
                number=number,
                title=title,
                completed_tasks=sum(phase_tasks),
                total_tasks=len(phase_tasks),
            )
        )

    total_tasks = sum(phase.total_tasks for phase in phases)
    completed_tasks = sum(phase.completed_tasks for phase in phases)
    current_phase_number = next(
        (
            phase.number
            for phase in phases
            if phase.completed_tasks < phase.total_tasks
        ),
        None,
    )
    return PlanSummary(
        phases=tuple(phases),
        completed_tasks=completed_tasks,
        total_tasks=total_tasks,
        current_phase_number=current_phase_number,
    )


def _render_markdown_with_phase_progress(markdown: str, summary: PlanSummary) -> str:
    """Render plan Markdown and annotate each Phase heading with its progress."""
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    renderer.use(tasklists_plugin)
    tokens = renderer.parse(markdown)
    _downgrade_images(tokens)
    phase_by_number = {phase.number: phase for phase in summary.phases}

    phase_for_heading_close: dict[int, PhaseSummary] = {}
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.tag != "h2":
            continue
        inline = tokens[index + 1]
        phase_match = _PHASE_HEADING.fullmatch(inline.content)
        if phase_match is None:
            continue
        phase = phase_by_number[int(phase_match.group(1))]
        token.attrSet("id", phase.anchor)
        phase_for_heading_close[index + 2] = phase

    rendered_tokens = []
    for index, token in enumerate(tokens):
        rendered_tokens.append(token)
        phase = phase_for_heading_close.get(index)
        if phase is None:
            continue
        progress = (
            '<div class="phase-progress" aria-label="Phase progress: '
            f'{phase.completed_tasks} of {phase.total_tasks} tasks completed">'
            f'<span>进度：{phase.completed_tasks} / {phase.total_tasks}</span>'
            f'<progress value="{phase.completed_tasks}" '
            f'max="{phase.total_tasks}">{phase.completed_tasks} / '
            f'{phase.total_tasks}</progress></div>\n'
        )
        rendered_tokens.append(
            token.copy(type="html_block", tag="", nesting=0, content=progress)
        )
    return renderer.renderer.render(rendered_tokens, renderer.options, {})


def _downgrade_images(tokens: list) -> None:
    """Replace image tokens with their alt text for self-contained output."""
    for token in tokens:
        for child in token.children or []:
            if child.type != "image":
                continue
            child.type = "text"
            child.tag = ""
            child.attrs = {}


def _phase_state(phase: PhaseSummary, current_phase_number: int | None) -> tuple[str, str]:
    """Return the stable state and visible label for a Phase."""
    if phase.completed_tasks == phase.total_tasks:
        return "completed", "已完成"
    if phase.number == current_phase_number:
        return "current", "进行中"
    return "pending", "待开始"


def build_html(source: Path, markdown: str) -> str:
    """Build a self-contained, deterministic HTML plan visualization."""
    summary = parse_plan(markdown)
    rendered_markdown = _render_markdown_with_phase_progress(markdown, summary)
    source_name = escape(source.stem)
    source_path = escape(source.as_posix())
    source_hash = sha256(markdown.encode("utf-8")).hexdigest()
    current_phase = (
        f"Phase {summary.current_phase_number}"
        if summary.current_phase_number is not None
        else "全部完成"
    )
    navigation = "\n".join(
        (
            f'<a href="#{phase.anchor}">{escape("Phase " + str(phase.number))}: '
            f'{escape(phase.title)} <span>{phase.completed_tasks} / '
            f'{phase.total_tasks}</span></a>'
        )
        for phase in summary.phases
    )
    phase_progress = "\n".join(
        (
            '<li class="phase-state {state}" data-state="{state}">'
            '<a href="#{anchor}">Phase {number}: {title}</a>'
            '<span class="phase-state-label">{label}</span>'
            '<progress value="{completed}" max="{total}">{completed} / '
            "{total}</progress><span>{completed} / {total}</span></li>"
        ).format(
            anchor=phase.anchor,
            number=phase.number,
            title=escape(phase.title),
            state=state,
            label=label,
            completed=phase.completed_tasks,
            total=phase.total_tasks,
        )
        for phase in summary.phases
        for state, label in [_phase_state(phase, summary.current_phase_number)]
    )
    return f"""<!doctype html>
<!-- Generated by scripts/render_plan.py v{GENERATOR_VERSION}; do not edit manually. -->
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{source_name}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; line-height: 1.6; background: Canvas; color: CanvasText; }}
    .page {{ max-width: 1200px; margin: auto; padding: 1.5rem; }}
    header {{ border-bottom: 1px solid color-mix(in srgb, CanvasText 24%, transparent); }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 1rem; margin: 1rem 0; }}
    .summary div, nav, .phase-list {{ border: 1px solid color-mix(in srgb, CanvasText 24%, transparent); border-radius: .5rem; padding: .75rem; }}
    .summary strong {{ display: block; font-size: 1.35rem; }}
    .layout {{ display: grid; gap: 1.5rem; margin-top: 1.5rem; }}
    nav {{ display: flex; flex-wrap: wrap; align-content: start; gap: .5rem; }}
    nav a {{ padding: .35rem .5rem; border-radius: .3rem; background: color-mix(in srgb, CanvasText 9%, transparent); }}
    nav span, .phase-list span {{ white-space: nowrap; }}
    a {{ color: LinkText; }}
    article {{ min-width: 0; overflow-wrap: anywhere; }}
    article img, article pre {{ max-width: 100%; }}
    article pre {{ padding: 1rem; white-space: pre-wrap; overflow-wrap: anywhere; background: color-mix(in srgb, CanvasText 8%, transparent); border-radius: .5rem; }}
    article table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    article th, article td {{ border: 1px solid color-mix(in srgb, CanvasText 24%, transparent); padding: .4rem; text-align: left; overflow-wrap: anywhere; }}
    .phase-list ul {{ padding-left: 1.25rem; margin-bottom: 0; }}
    .phase-list li {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(7rem, 1fr) auto; gap: .5rem; align-items: center; margin: .5rem 0; }}
    .phase-state-label {{ font-weight: 600; }}
    progress {{ inline-size: 100%; accent-color: Highlight; }}
    .phase-progress {{ display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; margin: -.4rem 0 1rem; }}
    footer {{ margin-top: 2rem; font-size: .9rem; overflow-wrap: anywhere; }}
    @media (min-width: 900px) {{ .layout {{ grid-template-columns: minmax(13rem, 18rem) minmax(0, 1fr); }} nav {{ position: sticky; top: 1rem; }} }}
    @media (max-width: 550px) {{ .phase-list li {{ grid-template-columns: 1fr; gap: .2rem; }} }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>{source_name}</h1>
      <div class="summary" aria-label="计划摘要">
        <div><span>总体任务</span><strong>{summary.completed_tasks} / {summary.total_tasks}</strong></div>
        <div><span>已完成 Phase</span><strong>{sum(phase.completed_tasks == phase.total_tasks for phase in summary.phases)} / {len(summary.phases)}</strong></div>
        <div><span>下一执行 Phase</span><strong>{escape(current_phase)}</strong></div>
      </div>
      <section class="phase-list" aria-labelledby="phase-progress-heading">
        <h2 id="phase-progress-heading">Phase 进度</h2>
        <ul>{phase_progress}</ul>
      </section>
    </header>
    <div class="layout">
      <nav aria-label="Phase 导航">{navigation}</nav>
      <article>{rendered_markdown}</article>
    </div>
    <footer>来源：{source_path}<br>SHA-256：{source_hash}<br>生成器版本：{GENERATOR_VERSION}</footer>
  </div>
</body>
</html>
"""


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for rendering a plan."""
    parser = argparse.ArgumentParser(
        description="Render a Markdown implementation plan as self-contained HTML."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Markdown plan to render (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML output path (default: source path with an .html suffix)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return nonzero when the output is missing or stale without writing it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render a plan, or check whether its HTML output is current."""
    arguments = _arguments(argv)
    source = arguments.source
    output = arguments.output or source.with_suffix(".html")

    try:
        markdown = source.read_text(encoding="utf-8")
        rendered_source = (
            source.relative_to(REPOSITORY_ROOT)
            if source == DEFAULT_SOURCE
            else source
        )
        rendered = build_html(rendered_source, markdown)
    except (OSError, UnicodeError, PlanRenderError) as error:
        print(f"render-plan: {error}", file=sys.stderr)
        return 2

    expected = rendered.encode("utf-8")
    if arguments.check:
        try:
            return 0 if output.read_bytes() == expected else 1
        except FileNotFoundError:
            return 1
        except OSError as error:
            print(f"render-plan: {error}", file=sys.stderr)
            return 2

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as file:
            file.write(rendered)
    except OSError as error:
        print(f"render-plan: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
