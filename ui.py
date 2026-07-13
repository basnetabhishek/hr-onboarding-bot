"""Original styling and small presentation helpers for the assistant.

Kept intentionally light: the app uses Streamlit's native chat components, so
this only adds a header and small source-citation chips.
"""

PAGE_TITLE = "HR Onboarding Assistant"
PAGE_ICON = "🧭"

CUSTOM_CSS = """
<style>
:root { --hr-accent: #2f5d62; }
.block-container { padding-top: 2.5rem; max-width: 820px; }
.app-header {
    display: flex; align-items: center; gap: 0.8rem;
    padding-bottom: 0.8rem; margin-bottom: 0.4rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.25);
}
.app-header .mark { font-size: 1.9rem; line-height: 1; }
.app-header h1 { font-size: 1.5rem; margin: 0; }
.app-tagline { color: #6b7280; font-size: 0.9rem; margin-top: 0.2rem; }
.source-chip {
    display: inline-block; font-size: 0.72rem; line-height: 1.6;
    padding: 0.04rem 0.6rem; margin: 0.2rem 0.3rem 0 0;
    border-radius: 999px;
    background: rgba(47, 93, 98, 0.12);
    color: var(--hr-accent);
    border: 1px solid rgba(47, 93, 98, 0.28);
}
.source-label { font-size: 0.72rem; color: #6b7280; margin-top: 0.35rem; }
</style>
"""


def header_html() -> str:
    return (
        '<div class="app-header">'
        f'<span class="mark">{PAGE_ICON}</span>'
        f"<div><h1>{PAGE_TITLE}</h1>"
        '<div class="app-tagline">Ask about policies, benefits, and onboarding '
        "&mdash; answered from your HR documents.</div>"
        "</div></div>"
    )


def sources_html(sources) -> str:
    chips = "".join(f'<span class="source-chip">{s}</span>' for s in sources)
    return (
        '<div class="source-label">Sources</div>'
        f'<div style="margin-top:0.1rem">{chips}</div>'
    )
