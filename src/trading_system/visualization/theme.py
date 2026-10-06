"""Shared dashboard tokens, sourced from the project's web interface."""

from pathlib import Path
import re


SYSTEM_FONT = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
PALETTE = {
    "ink": "#101318", "muted": "#626975", "line": "#dfe3e8",
    "paper": "#f7f8fa", "white": "#ffffff", "blue": "#165dff",
    "blue-dark": "#1048d1", "sell": "#d25562", "sell-ink": "#b93c4b",
    "hold": "#e4a332", "buy": "#165dff",
}
_stylesheet = Path(__file__).resolve().parents[3] / "web" / "frontend" / "styles.css"
if _stylesheet.is_file():
    for _block in re.findall(r":root\s*\{([^}]+)\}", _stylesheet.read_text()):
        for _name, _value in re.findall(r"--([\w-]+)\s*:\s*(#[\da-fA-F]{3,8})\s*[;}]?", _block):
            if _name in PALETTE:
                PALETTE[_name] = _value


def dashboard_css() -> str:
    """Light, legible chart surfaces with restrained structural translucency."""
    p = PALETTE
    return f"""<style>
    :root {{ color-scheme: light; }}
    .stApp {{ background: {p['paper']}; color: {p['ink']}; font-family: {SYSTEM_FONT}; }}
    .stApp h1, .stApp h2, .stApp h3 {{ color: {p['ink']}; font-family: {SYSTEM_FONT};
        letter-spacing: -.035em; font-weight: 650; }}
    .stApp p, .stApp label, .stApp [data-testid="stMetricLabel"] {{ color: {p['ink']}; }}
    .stApp [data-testid="stCaptionContainer"] p {{ color: {p['muted']}; }}
    .stApp [data-testid="stHeader"] {{ background: rgba(247,248,250,.88);
        backdrop-filter: blur(16px); border-bottom: 1px solid {p['line']}; }}
    .stApp [data-testid="stSidebar"] {{ background: {p['white']}; border-right: 1px solid {p['line']}; }}
    .stApp [data-testid="stMainBlockContainer"] {{ max-width: 1500px; padding-top: 3rem; }}
    .stApp [data-testid="stMetric"] {{ background: {p['white']}; border: 1px solid {p['line']};
        border-radius: 14px; padding: 1rem 1.15rem; min-height: 112px; }}
    .stApp [data-testid="stMetricValue"] {{ color: {p['ink']}; font-size: clamp(1.3rem,2vw,2rem); letter-spacing: -.04em; }}
    .stApp [data-testid="stPlotlyChart"] {{ background: {p['white']}; border: 1px solid {p['line']};
        border-radius: 16px; padding: .3rem; overflow: hidden; }}
    .stApp [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {p['line']}; }}
    .stApp [data-baseweb="tab"] {{ color: {p['muted']}; }}
    .stApp [data-baseweb="tab"][aria-selected="true"] {{ color: {p['blue']}; }}
    .stApp [data-baseweb="tab-highlight"] {{ background: {p['blue']}; }}
    .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"],
    .stApp [data-baseweb="base-input"] {{ background: {p['white']}; color: {p['ink']};
        border-color: {p['line']}; }}
    .stApp input {{ color: {p['ink']}; caret-color: {p['blue']}; }}
    .stApp [data-baseweb="tag"] {{ background: #e9efff; color: {p['blue-dark']}; }}
    .stApp [data-baseweb="tag"] span {{ color: {p['blue-dark']}; }}
    .stApp button {{ transition: transform 120ms ease-out, background-color 120ms ease-out; }}
    .stApp button:active {{ transform: scale(.98); }}
    .stApp [data-testid="stBaseButton-secondary"] {{ background: {p['white']}; color: {p['ink']};
        border: 1px solid {p['line']}; border-radius: 10px; }}
    .stApp [data-testid="stBaseButton-secondary"]:hover {{ border-color: {p['blue']}; color: {p['blue']}; }}
    .stApp [data-testid="stExpander"] {{ border-color: {p['line']}; background: {p['white']}; border-radius: 12px; }}
    .stApp [data-testid="stExpander"] details summary {{ color: {p['ink']}; }}
    .stApp [data-testid="stAppDeployButton"] {{ display: none; }}
    @media (max-width: 700px) {{
        .stApp [data-testid="stMainBlockContainer"] {{ padding: 2.4rem 1rem 1rem; }}
        .stApp [data-testid="stMetric"] {{ min-height: auto; padding: .75rem 1rem; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ .stApp button {{ transition: none; }} }}
    @media (prefers-reduced-transparency: reduce) {{
        .stApp [data-testid="stHeader"] {{ background: {p['paper']}; backdrop-filter: none; }}
    }}
    @media (prefers-contrast: more) {{
        .stApp [data-testid="stCaptionContainer"] p, .stApp [data-baseweb="tab"] {{ color: {p['ink']}; }}
        .stApp [data-testid="stMetric"], .stApp [data-testid="stPlotlyChart"] {{ border-color: {p['muted']}; }}
    }}
    </style>"""
