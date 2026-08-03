# 04 — Conventions

## Language rule (the exact wording matters)

`default ENGLISH, then English/French when possible.`

This wording was deliberately KEPT (decision of 12 July 2026); a proposal to
replace it with a STRICT bilingual mandate was **refused**. Full repo
bilingualism is a "when possible" goal, not a blocking requirement.

Per-artifact guide for the "when possible" moments — a GUIDE, not a mandate:

| Artifact | Convention |
|---|---|
| README | sections doubled, EN then FR |
| Docstrings | blocks EN then FR |
| Error messages | one line EN then FR |
| Notebooks | stacked bilingual cells |
| pyproject / GitHub description | bilingual |
| Issues | body EN + FR summary |
| Identifiers, trivial inline comments | English only |

`scripts/check_bilingual.py` reports the gaps. It is an **advisory dashboard,
permanently non-blocking** — never wire it into CI as a gate. Slash command:
`/bilingual-audit`.

Still open (author's call, not a requirement): bilingualising `_PARAM_HINTS` and
the ABI warning, both user-facing and therefore typical "possible" cases.

## Communication channels (outside the repo)

English-only: Hacker News, Reddit, dev.to, Manning pitches.
LinkedIn / Facebook: sequential bilingual (FR paragraph, then EN).
Live talks: French, English on request.

## Documents

Word/PDF deliverables use the author's "Executive Shift" design system.
Do not regenerate a full deliverable for a small correction — describe the
correction in text, and only regenerate at an explicit checkpoint.

## Numbers

Verify every figure programmatically before writing it. See `00-pitfalls.md`,
items P3 and P8.
