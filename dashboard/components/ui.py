"""
CollideX Dashboard — Reusable UI Components
=============================================
All HTML/CSS component builders using INLINE STYLES only.
Strings are stripped to avoid Streamlit treating leading whitespace as code blocks.
"""

from config import COLORS, RISK_COLORS, RISK_BG_COLORS


# ---------------------------------------------------------------------------
# Core Cards
# ---------------------------------------------------------------------------

def metric_card(value: str, label: str, sublabel: str = "",
                color: str = "#00D4FF", icon: str = "") -> str:
    """Glassmorphism metric card — inline styles, no leading whitespace."""
    icon_html = f'<div style="font-size:26px;margin-bottom:8px;">{icon}</div>' if icon else ""
    sub_html  = (f'<div style="font-size:12px;color:#8BA3C7;margin-top:4px;">{sublabel}</div>'
                 if sublabel else "")
    return (
        f'<div style="background:rgba(13,31,60,0.85);border:1px solid #1A3A5C;'
        f'border-radius:18px;padding:24px 20px;text-align:center;'
        f'backdrop-filter:blur(10px);border-top:2px solid {color}22;">'
        f'{icon_html}'
        f'<div style="font-size:34px;font-weight:800;color:{color};line-height:1;'
        f'font-family:\'JetBrains Mono\',monospace;">{value}</div>'
        f'<div style="font-size:11px;color:#4A6A8A;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-top:8px;font-weight:600;">{label}</div>'
        f'{sub_html}'
        f'</div>'
    )


def kpi_card(value: str, label: str, icon: str = "",
             color: str = "#00D4FF", sub: str = "") -> str:
    """KPI card with icon — no leading whitespace."""
    sub_html = (f'<div style="font-size:12px;color:#4A6A8A;margin-top:4px;">{sub}</div>'
                if sub else "")
    return (
        f'<div style="background:rgba(13,31,60,0.9);border:1px solid #1A3A5C;'
        f'border-radius:12px;padding:18px;backdrop-filter:blur(12px);">'
        f'<div style="font-size:26px;margin-bottom:6px;">{icon}</div>'
        f'<div style="font-size:26px;font-weight:700;color:{color};'
        f'font-family:\'JetBrains Mono\',monospace;">{value}</div>'
        f'<div style="font-size:11px;color:#4A6A8A;text-transform:uppercase;'
        f'letter-spacing:1px;margin-top:4px;font-weight:600;">{label}</div>'
        f'{sub_html}'
        f'</div>'
    )


def risk_badge(risk_label: str) -> str:
    colors  = {"High": "#FF3860", "Medium": "#FFD700", "Low": "#00FF9F"}
    bgs     = {"High": "rgba(255,56,96,0.15)", "Medium": "rgba(255,215,0,0.15)",
               "Low": "rgba(0,255,159,0.15)"}
    borders = {"High": "rgba(255,56,96,0.35)", "Medium": "rgba(255,215,0,0.35)",
               "Low": "rgba(0,255,159,0.35)"}
    c  = colors.get(risk_label, "#8BA3C7")
    bg = bgs.get(risk_label, "rgba(139,163,199,0.1)")
    bo = borders.get(risk_label, "rgba(139,163,199,0.2)")
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
        f'font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;'
        f'background:{bg};color:{c};border:1px solid {bo};">{risk_label}</span>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    sub_html = (f'<div style="font-size:13px;color:#4A6A8A;margin-top:4px;">{subtitle}</div>'
                if subtitle else "")
    return (
        f'<div style="display:flex;align-items:center;gap:12px;'
        f'margin:32px 0 20px 0;padding-bottom:12px;border-bottom:1px solid #1A3A5C;">'
        f'<div style="width:4px;height:24px;background:linear-gradient(180deg,#00D4FF,#1E6FFF);'
        f'border-radius:2px;flex-shrink:0;"></div>'
        f'<div>'
        f'<div style="font-size:18px;font-weight:700;color:#E8F4FD;letter-spacing:0.5px;">{title}</div>'
        f'{sub_html}'
        f'</div>'
        f'</div>'
    )


def hero_section(title: str, subtitle: str,
                 tagline: str = "AI-POWERED SPACE DEBRIS MONITORING") -> str:
    return (
        f'<div style="text-align:center;padding:80px 40px 60px;">'
        f'<div style="font-size:11px;font-weight:600;letter-spacing:4px;color:#00D4FF;'
        f'text-transform:uppercase;margin-bottom:16px;">{tagline}</div>'
        f'<div style="font-size:clamp(40px,6vw,72px);font-weight:900;letter-spacing:-1px;'
        f'line-height:1.05;background:linear-gradient(135deg,#E8F4FD 0%,#00D4FF 50%,#1E6FFF 100%);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        f'background-clip:text;margin-bottom:12px;">{title}</div>'
        f'<div style="font-size:18px;color:#8BA3C7;max-width:600px;margin:0 auto 32px;'
        f'line-height:1.6;font-weight:400;">{subtitle}</div>'
        f'</div>'
    )


def sidebar_logo() -> str:
    return (
        '<div style="padding:24px 20px 16px;border-bottom:1px solid #1A3A5C;margin-bottom:8px;">'
        '<div style="font-size:22px;font-weight:800;letter-spacing:2px;color:#00D4FF;'
        'text-transform:uppercase;">CollideX</div>'
        '<div style="font-size:10px;color:#4A6A8A;letter-spacing:1px;'
        'text-transform:uppercase;margin-top:2px;">Collision Prediction System</div>'
        '<div style="font-size:10px;color:#1E6FFF;margin-top:4px;'
        'font-family:\'JetBrains Mono\',monospace;">v1.0 Production Build</div>'
        '</div>'
    )


def status_indicator(label: str, status: str = "online") -> str:
    dot_c = {"online": "#00FF9F", "warning": "#FFD700", "offline": "#FF3860"}.get(status, "#8BA3C7")
    return (
        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 0;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{dot_c};'
        f'display:inline-block;box-shadow:0 0 6px {dot_c};"></span>'
        f'<span style="font-size:12px;color:#8BA3C7;">{label}</span>'
        f'</div>'
    )


def info_box(text: str, kind: str = "info") -> str:
    if kind == "warning":
        bg, border, color, icon = "rgba(255,107,53,0.06)", "rgba(255,107,53,0.2)", "#FF6B35", "&#9888;"
    else:
        bg, border, color, icon = "rgba(0,212,255,0.06)", "rgba(0,212,255,0.2)", "#00D4FF", "&#8505;"
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;'
        f'padding:14px 18px;margin:10px 0;display:flex;align-items:flex-start;gap:10px;">'
        f'<span style="color:{color};font-size:14px;flex-shrink:0;">{icon}</span>'
        f'<span style="font-size:13px;color:#8BA3C7;line-height:1.5;">{text}</span>'
        f'</div>'
    )


def pipeline_step(num: int, label: str, detail: str = "") -> str:
    detail_html = (f'<span style="font-size:11px;color:#4A6A8A;margin-left:4px;">&#8212; {detail}</span>'
                   if detail else "")
    return (
        f'<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;'
        f'background:rgba(13,31,60,0.6);border:1px solid #1A3A5C;'
        f'border-radius:8px;margin-bottom:6px;font-size:13px;color:#8BA3C7;">'
        f'<span style="background:linear-gradient(135deg,#1E6FFF,#00D4FF);color:#050A14;'
        f'font-weight:700;font-size:11px;width:22px;height:22px;border-radius:50%;'
        f'display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;">{num}</span>'
        f'<span style="color:#E8F4FD;">{label}</span>'
        f'{detail_html}'
        f'</div>'
    )


def stat_row(label: str, value: str, highlight: bool = False) -> str:
    color = "#00D4FF" if highlight else "#E8F4FD"
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 0;border-bottom:1px solid #1A3A5C;">'
        f'<span style="font-size:13px;color:#8BA3C7;">{label}</span>'
        f'<span style="font-size:13px;font-weight:600;color:{color};'
        f'font-family:\'JetBrains Mono\',monospace;">{value}</span>'
        f'</div>'
    )


def divider() -> str:
    return '<hr style="border:none;border-top:1px solid #1A3A5C;margin:24px 0;">'


def page_title(title: str, subtitle: str = "") -> str:
    sub = (f'<p style="color:#4A6A8A;font-size:13px;margin:0;letter-spacing:1px;">{subtitle}</p>'
           if subtitle else "")
    return (
        f'<div style="padding:0 0 24px 0;">'
        f'<h1 style="font-size:28px;font-weight:800;color:#E8F4FD;'
        f'margin:0 0 4px 0;letter-spacing:-0.5px;">{title}</h1>'
        f'{sub}'
        f'</div>'
    )


def progress_bar_html(value: float, max_val: float = 1.0, color: str = "#00D4FF") -> str:
    pct = min(100, round(value / max_val * 100, 1))
    return (
        f'<div style="background:#1A3A5C;border-radius:4px;height:6px;overflow:hidden;margin-top:4px;">'
        f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;"></div>'
        f'</div>'
        f'<div style="text-align:right;font-size:11px;color:#4A6A8A;margin-top:2px;">{pct}%</div>'
    )
