import streamlit as st


def inject_global_css():
    st.markdown(
        """
<style>
:root{
  --bg1:#0a1020;
  --bg2:#0d1728;
  --text:#eaf2ff;
  --border:rgba(255,255,255,0.12);
  --accent1:#2af598;
  --accent2:#00d8ff;
  --accent3:#a78bfa;
  --muted:#b6c2ff;
}
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&display=swap');
html, body, [data-testid="stAppViewContainer"] * {
  font-family: "Poppins", sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
  background:
    radial-gradient(1200px 600px at 10% -10%, #0c214b 0%, transparent 70%),
    radial-gradient(1200px 600px at 90% -10%, #231548 0%, transparent 70%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
  color: var(--text);
}
h1, h2, h3, .gradient-heading{
  font-weight:900 !important;
  background: linear-gradient(90deg, var(--accent1), var(--accent2), var(--accent3));
  -webkit-background-clip:text;
  background-clip:text;
  -webkit-text-fill-color:transparent;
  letter-spacing:.4px;
}
.dashboard-wrap{
  width:min(1200px, 94vw);
  margin: 0 auto;
}
.card{
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 16px 40px rgba(0,0,0,0.35),
              inset 0 0 0 1px rgba(255,255,255,0.04);
  backdrop-filter: blur(8px);
}
.hero-card{
  text-align:center;
  padding: 26px 20px;
  margin: 14px 0 18px;
}
.live-card{
  margin: 10px 0 14px;
  padding: 14px;
}
.gallery-card{
  margin: 10px 0 14px;
}
.card-marker{
  display:block;
  height:0;
  margin:0;
}
.card-marker + div{
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 16px 40px rgba(0,0,0,0.35),
              inset 0 0 0 1px rgba(255,255,255,0.04);
  backdrop-filter: blur(8px);
  margin: 10px 0 14px;
}
div.stButton > button{
  background: linear-gradient(135deg, var(--accent1), var(--accent2));
  color:#0a0f1c !important;
  font-weight:900;
  letter-spacing:.3px;
  border:0 !important;
  border-radius:12px !important;
  padding:10px 16px !important;
  box-shadow: 0 10px 24px rgba(0,216,255,0.25);
  transition: transform .08s ease, filter .2s ease, box-shadow .2s ease;
}
div.stButton > button:hover{
  transform: translateY(-1px);
  filter: brightness(1.05);
  box-shadow: 0 12px 28px rgba(0,216,255,0.35);
}
div.stButton > button:active{
  transform: translateY(0px);
  filter: brightness(0.98);
}
input, textarea{
  background:#0f172a !important;
  color:#e2e8f0 !important;
  border:1px solid var(--border) !important;
  border-radius:10px !important;
}
[data-baseweb="slider"] > div{
  background: linear-gradient(90deg,
      rgba(42,245,152,.35),
      rgba(0,216,255,.35)) !important;
}
.navbar{
  display:flex;
  justify-content:flex-end;
  gap:12px;
  margin: 6px 0 8px 0;
}
.status-row{
  display:flex;
  justify-content:center;
  margin-top:10px;
}
.status-chip{
  padding:8px 14px;
  border-radius:999px;
  font-weight:900;
  letter-spacing:.3px;
  border:1px solid;
  font-size:.95rem;
}
.status-chip.on{
  background:rgba(16,185,129,.12);
  color:#34d399;
  border-color:rgba(16,185,129,.35);
}
.status-chip.off{
  background:rgba(239,68,68,.12);
  color:#f87171;
  border-color:rgba(239,68,68,.35);
}
.live-wrap{
  display:flex;
  justify-content:center;
}
.live-frame{
  position:relative;
  border:0;
  padding:0;
  background:transparent;
}
.live-inner{
  overflow:hidden;
  border-radius:18px;
  background:#0f172a;
}
.live-inner img, .live-img{
  width:100% !important;
  height:auto !important;
  display:block;
  object-fit:contain;
  background:#0f172a;
  border-radius:18px;
  image-rendering:auto;
  will-change:transform, opacity;
  transform: translateZ(0);
  backface-visibility:hidden;
  contain:content;
}
.shot-grid{
  display:grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap:18px;
  padding:8px 0 2px;
}
.shot-card{
  width:100%;
  border-radius:16px;
  overflow:hidden;
  background: rgba(255,255,255,0.06);
  border:1px solid var(--border);
  position:relative;
}
.shot-card img{
  width:100%;
  height:auto;
  display:block;
}
.shot-badge{
  position:absolute;
  top:12px;
  left:12px;
  background:linear-gradient(135deg,#06b6d4,#8b5cf6);
  color:white;
  font-weight:800;
  font-size:.85rem;
  padding:6px 10px;
  border-radius:999px;
}
.shot-footer{
  padding:10px 12px;
  color:#eaf2ff;
  font-weight:700;
  border-top:1px solid rgba(255,255,255,0.1);
}
.table-like { width: 100%; }
.table-like .th {
  padding: 10px 12px;
  border-bottom:1px solid rgba(255,255,255,0.08);
  color:#b6c2ff;
  font-weight:900;
  letter-spacing:.25px;
}
.table-like .td {
  padding: 10px 12px;
  border-bottom:1px solid rgba(255,255,255,0.08);
}
.table-like .td-url {
  display:block;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
.pagination{
  display:flex;
  gap:8px;
  align-items:center;
  justify-content:flex-end;
  margin-top:8px;
}
.page-chip{
  padding:6px 10px;
  border-radius:999px;
  border:1px solid var(--border);
  color:#e2e8f0;
  background: rgba(255,255,255,0.05);
}
.page-chip.active{
  background:#0ea5e9;
  border-color:#38bdf8;
  color:white;
  font-weight:900;
}
</style>
        """,
        unsafe_allow_html=True,
    )