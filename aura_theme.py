"""
aura_theme.py — AURA India · Art-Inspired Design System
=========================================================
Vibrant paint-palette colours, watercolour CSS patterns,
art-themed UI components, Unsplash image helpers.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── PAINT PALETTE ─────────────────────────────────────────────
BG       = "#0a0608"
SURFACE  = "#130d10"
SURFACE2 = "#1c1218"
SURFACE3 = "#261820"
INK      = "#f5ede8"
MUTED    = "#8a7070"

GOLD     = "#f0c040"
TEAL     = "#3ecfa8"
ORANGE   = "#f07840"
ROSE     = "#e8507a"
INDIGO   = "#8878f0"
LAVENDER = "#c098e8"
SAGE     = "#78c8a0"
AMBER    = "#f0a030"
COBALT   = "#4898e8"
CORAL    = "#f07860"

PALETTE = [GOLD, TEAL, ORANGE, ROSE, INDIGO, SAGE, LAVENDER, AMBER, COBALT, CORAL]
CLUSTER_COLORS = [GOLD, TEAL, ORANGE, ROSE, INDIGO, SAGE, LAVENDER, AMBER]

# ── UNSPLASH IMAGE URLS ───────────────────────────────────────
IMAGES = {
    "hero_paint":    "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=1400&q=80",
    "watercolour":   "https://images.unsplash.com/photo-1513364776144-60967b0f800f?w=1200&q=80",
    "art_supplies":  "https://images.unsplash.com/photo-1499781350541-7783f6c6a0c8?w=1200&q=80",
    "mandala":       "https://images.unsplash.com/photo-1605106702734-205df224ecce?w=800&q=80",
    "pottery":       "https://images.unsplash.com/photo-1565193566173-7a0ee3dbe261?w=800&q=80",
    "sketching":     "https://images.unsplash.com/photo-1452860606245-08befc0ff44b?w=800&q=80",
    "calligraphy":   "https://images.unsplash.com/photo-1455885661740-29cbf08a42fa?w=800&q=80",
    "paint_brushes": "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=800&q=80",
    "colour_palette":"https://images.unsplash.com/photo-1587654780291-39c9404d746b?w=800&q=80",
    "urban_art":     "https://images.unsplash.com/photo-1561214115-f2f134cc4912?w=1200&q=80",
    "studio":        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=1200&q=80",
    "india_art":     "https://images.unsplash.com/photo-1524492412937-b28074a5d7da?w=1200&q=80",
    "wellness":      "https://images.unsplash.com/photo-1506126613408-eca07ce68773?w=800&q=80",
    "acrylic":       "https://images.unsplash.com/photo-1618005198919-d3d4b5a92ead?w=800&q=80",
    "street_art":    "https://images.unsplash.com/photo-1531913223931-b0d3198229ee?w=800&q=80",
}

# ── BRUSH STROKE SVG ─────────────────────────────────────────
BRUSH_STROKE_SVG = """<svg viewBox="0 0 300 10" xmlns="http://www.w3.org/2000/svg" style="width:200px;display:block;">
  <path d="M0,5 Q75,1 150,5 Q225,9 300,5" stroke="url(#bgrad)" stroke-width="3.5" fill="none" stroke-linecap="round"/>
  <defs><linearGradient id="bgrad" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" style="stop-color:#f0c040;stop-opacity:1"/>
    <stop offset="50%" style="stop-color:#3ecfa8;stop-opacity:1"/>
    <stop offset="100%" style="stop-color:#8878f0;stop-opacity:1"/>
  </linearGradient></defs>
</svg>"""

# ── PLOTLY TEMPLATE ───────────────────────────────────────────
_t = go.layout.Template()
_t.layout = go.Layout(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE2,
    font=dict(family="Inter, system-ui, sans-serif", color=INK, size=12),
    title=dict(font=dict(color=INK, size=16), x=0.01),
    colorway=PALETTE,
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
               linecolor="rgba(255,255,255,0.06)", tickcolor=MUTED),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
               linecolor="rgba(255,255,255,0.06)", tickcolor=MUTED),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)", borderwidth=1),
    margin=dict(l=40, r=20, t=50, b=40),
)
pio.templates["aura"] = _t
pio.templates.default = "aura"

# ── GLOBAL CSS ─────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: {BG} !important;
    color: {INK} !important;
}}
.stApp {{ background-color: {BG} !important; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SURFACE} 0%, #1a0e14 60%, #0f080b 100%) !important;
    border-right: 1px solid rgba(240,192,64,0.18) !important;
}}
[data-testid="stSidebar"] * {{ color: {INK} !important; }}

.stRadio > div > label {{ color: {INK} !important; }}
.stRadio > div > div[data-checked="true"] > label {{ color: {GOLD} !important; font-weight:600 !important; }}

.stSelectbox > div > div {{
    background-color: {SURFACE2} !important; color: {INK} !important;
    border-color: rgba(240,192,64,0.2) !important; border-radius: 8px !important;
}}

div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, {SURFACE2}, {SURFACE3}) !important;
    border: 1px solid rgba(240,192,64,0.15) !important;
    border-radius: 10px !important; padding: 18px !important;
}}
div[data-testid="metric-container"] label {{
    color: {MUTED} !important; font-size: 11px !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
}}
div[data-testid="metric-container"] [data-testid="metric-value"] {{
    color: {GOLD} !important; font-weight: 700 !important; font-size: 26px !important;
}}

.stDataFrame {{ border: 1px solid rgba(240,192,64,0.15) !important; border-radius: 10px !important; }}

.stDownloadButton > button {{
    background: linear-gradient(135deg,rgba(240,192,64,0.15),rgba(62,207,168,0.15)) !important;
    color: {GOLD} !important; border: 1px solid rgba(240,192,64,0.4) !important;
    border-radius: 8px !important; font-weight: 600 !important; font-size: 12px !important;
}}
.stDownloadButton > button:hover {{
    background: linear-gradient(135deg,rgba(240,192,64,0.3),rgba(62,207,168,0.3)) !important;
    border-color: {GOLD} !important;
}}

.stButton > button {{
    background: linear-gradient(135deg, {GOLD}, {ORANGE}) !important;
    color: {BG} !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; letter-spacing: 0.04em !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; box-shadow: 0 4px 20px rgba(240,192,64,0.3) !important; }}

.stExpander {{
    border: 1px solid rgba(240,192,64,0.15) !important;
    border-radius: 10px !important; background: {SURFACE2} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: {SURFACE2} !important; border-radius: 10px !important; padding: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{ border-radius: 7px !important; color: {MUTED} !important; font-weight: 500 !important; }}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg,rgba(240,192,64,0.2),rgba(62,207,168,0.2)) !important;
    color: {GOLD} !important; font-weight: 700 !important;
}}

hr {{ border-color: rgba(240,192,64,0.1) !important; }}
h1, h2, h3 {{ color: {INK} !important; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient({GOLD},{TEAL}); border-radius: 3px; }}
</style>
"""

# ── UI COMPONENTS ─────────────────────────────────────────────
def page_header(title: str, subtitle: str = "", badge: str = ""):
    badge_html = (
        f'<span style="background:linear-gradient(135deg,rgba(240,192,64,0.2),rgba(62,207,168,0.2));'
        f'border:1px solid rgba(240,192,64,0.3);color:{GOLD};font-size:10px;letter-spacing:0.15em;'
        f'text-transform:uppercase;padding:5px 14px;border-radius:20px;font-weight:600;">{badge}</span>'
    ) if badge else ""
    st.markdown(f"""
    <div style="padding:32px 0 8px;">
        {badge_html}
        <div style="font-size:34px;font-weight:900;color:{INK};margin-top:{'14px' if badge else '0'};
        line-height:1.1;font-family:'Playfair Display',serif;">{title}</div>
        {f'<div style="font-size:14px;color:{MUTED};margin-top:10px;line-height:1.7;max-width:740px;">{subtitle}</div>' if subtitle else ""}
        <div style="margin-top:14px;">{BRUSH_STROKE_SVG}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, color: str = GOLD):
    st.markdown(f"""
    <div style="margin:28px 0 14px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:4px;height:20px;background:linear-gradient({color},{TEAL});
            border-radius:2px;box-shadow:0 0 8px {color}44;"></div>
            <span style="font-size:14px;font-weight:700;color:{INK};letter-spacing:0.04em;">{title}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def info_card(title: str, body: str, color: str = TEAL):
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{SURFACE2},{SURFACE3});
    border:1px solid rgba(255,255,255,0.06);border-left:4px solid {color};
    border-radius:10px;padding:18px 22px;margin:16px 0;box-shadow:0 2px 12px rgba(0,0,0,0.3);">
        <div style="color:{color};font-size:11px;letter-spacing:0.12em;
        text-transform:uppercase;font-weight:700;margin-bottom:8px;">🎨 {title}</div>
        <div style="color:{MUTED};font-size:13px;line-height:1.7;">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", color: str = GOLD):
    delta_html = f'<div style="color:{TEAL};font-size:11px;margin-top:4px;">{delta}</div>' if delta else ""
    return f"""
    <div style="background:linear-gradient(135deg,{SURFACE2},{SURFACE3});
    border:1px solid rgba(255,255,255,0.06);border-top:3px solid {color};
    border-radius:10px;padding:20px;text-align:center;
    box-shadow:0 4px 16px rgba(0,0,0,0.25);">
        <div style="color:{MUTED};font-size:10px;letter-spacing:0.15em;
        text-transform:uppercase;font-weight:600;margin-bottom:8px;">{label}</div>
        <div style="color:{color};font-size:28px;font-weight:800;
        line-height:1;text-shadow:0 0 20px {color}44;">{value}</div>
        {delta_html}
    </div>
    """


def art_image_banner(url: str, height: int = 220, overlay_text: str = "", overlay_sub: str = ""):
    """Full-width Unsplash art banner with text overlay."""
    text_block = f"""
        <div style="position:absolute;bottom:24px;left:32px;z-index:2;">
            <div style="color:white;font-size:26px;font-weight:900;
            font-family:'Playfair Display',serif;text-shadow:0 2px 12px rgba(0,0,0,0.9);">
                {overlay_text}</div>
            <div style="color:rgba(255,255,255,0.85);font-size:13px;margin-top:4px;
            text-shadow:0 1px 6px rgba(0,0,0,0.8);">{overlay_sub}</div>
        </div>""" if overlay_text else ""
    st.markdown(f"""
    <div style="position:relative;width:100%;height:{height}px;
    border-radius:14px;overflow:hidden;margin-bottom:24px;
    box-shadow:0 8px 32px rgba(0,0,0,0.5);">
        <img src="{url}" style="width:100%;height:100%;object-fit:cover;display:block;"
             onerror="this.style.display='none'"/>
        <div style="position:absolute;inset:0;
        background:linear-gradient(to bottom,rgba(0,0,0,0.05),rgba(0,0,0,0.6));
        z-index:1;border-radius:14px;"></div>
        {text_block}
    </div>
    """, unsafe_allow_html=True)


def art_image_grid(images: list, captions: list = None, height: int = 150):
    """Row of art images with captions."""
    cols = st.columns(len(images))
    for i, (col, url) in enumerate(zip(cols, images)):
        cap = captions[i] if captions and i < len(captions) else ""
        with col:
            st.markdown(f"""
            <div style="border-radius:10px;overflow:hidden;
            box-shadow:0 4px 16px rgba(0,0,0,0.35);margin-bottom:6px;">
                <img src="{url}" style="width:100%;height:{height}px;object-fit:cover;display:block;"
                     onerror="this.style.display='none'"/>
            </div>
            <div style="color:{MUTED};font-size:11px;text-align:center;
            letter-spacing:0.05em;padding:2px 0;">{cap}</div>
            """, unsafe_allow_html=True)


def colour_palette_strip(colors: list = None):
    """Paint swatch strip."""
    colors = colors or PALETTE[:8]
    swatches = "".join([
        f'<div style="flex:1;height:28px;background:{c};border-radius:4px;'
        f'box-shadow:inset 0 -2px 4px rgba(0,0,0,0.2);"></div>'
        for c in colors
    ])
    st.markdown(f"""
    <div style="display:flex;gap:4px;border-radius:8px;overflow:hidden;
    margin:12px 0;box-shadow:0 2px 10px rgba(0,0,0,0.4);">{swatches}</div>
    """, unsafe_allow_html=True)


def persona_avatar(emoji: str, name: str, desc: str, color: str,
                   stat_label: str = "", stat_val: str = ""):
    stat_block = f"""
    <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.06);">
        <span style="color:{MUTED};font-size:10px;">{stat_label}</span>
        <span style="color:{color};font-size:13px;font-weight:700;margin-left:8px;">{stat_val}</span>
    </div>""" if stat_label else ""
    return f"""
    <div style="background:linear-gradient(135deg,{SURFACE2},{SURFACE3});
    border:1px solid rgba(255,255,255,0.06);border-top:3px solid {color};
    border-radius:12px;padding:18px;text-align:center;">
        <div style="font-size:36px;margin-bottom:8px;">{emoji}</div>
        <div style="color:{color};font-size:12px;font-weight:700;
        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">{name}</div>
        <div style="color:{MUTED};font-size:11px;line-height:1.6;">{desc}</div>
        {stat_block}
    </div>"""
