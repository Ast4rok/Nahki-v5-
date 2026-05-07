import streamlit as st
import json
import os
import uuid
import base64
import zipfile
import io
import hashlib
from datetime import datetime
from groq import Groq

HISTORIES_DIR = os.path.join(os.path.dirname(__file__), "chat_histories")
COVERS_DIR    = os.path.join(os.path.dirname(__file__), "covers")
os.makedirs(HISTORIES_DIR, exist_ok=True)
os.makedirs(COVERS_DIR,    exist_ok=True)

IMG_EXTS = {"png", "jpg", "jpeg", "webp", "gif"}

# ── SVG icons ─────────────────────────────────────────────────────────────────
SVG_NEW = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>'
    '</svg>'
)
SVG_GEAR = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="3"/>'
    '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06'
    'a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09'
    'A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06'
    'A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09'
    'A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06'
    'A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09'
    'a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06'
    'A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09'
    'a1.65 1.65 0 0 0-1.51 1z"/></svg>'
)
SVG_DOTS = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">'
    '<circle cx="12" cy="5" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="19" r="2"/>'
    '</svg>'
)
SVG_PIN = (
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>'
    '<circle cx="12" cy="10" r="3"/></svg>'
)
SVG_UNPIN = (
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round">'
    '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
)
SVG_TRASH = (
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="3 6 5 6 21 6"/>'
    '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>'
    '</svg>'
)
SVG_CLIP = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66'
    'l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>'
)


# ── AI Providers ───────────────────────────────────────────────────────────────
def get_providers():
    providers = []
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        providers.append({
            "name": "Groq",
            "client": Groq(api_key=groq_key),
            "models": [
                "llama-3.3-70b-versatile",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.1-8b-instant",
            ],
        })
    return providers


_SKIP_ERRORS = [
    "429", "503", "404",
    "rate", "overloaded", "unavailable", "timeout",
    "no endpoints", "temporarily", "try again",
]


def call_with_fallback(messages):
    providers = get_providers()
    if not providers:
        return (
            "⚠️ Nenhuma chave de API configurada. "
            "Adicione GROQ_API_KEY nos Secrets.",
            "—",
        )
    last_err = ""
    for provider in providers:
        for model in provider["models"]:
            try:
                resp = provider["client"].chat.completions.create(
                    model=model, messages=messages,
                )
                return resp.choices[0].message.content or "", f"{provider['name']} · {model}"
            except Exception as e:
                err = str(e).lower()
                last_err = str(e)
                if any(kw in err for kw in _SKIP_ERRORS):
                    continue
                return f"❌ Erro inesperado: {e}", f"{provider['name']} · {model}"
    return f"⚠️ Todos os modelos estão indisponíveis. Último erro: {last_err[:120]}", "—"


# ── Chat helpers ───────────────────────────────────────────────────────────────
def text_from_content(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part["text"].strip()
    return ""


def make_title(messages):
    for msg in messages:
        if msg["role"] == "user":
            txt = text_from_content(msg["content"])
            if not txt:
                txt = "[Imagem]"
            return txt[:44] + ("..." if len(txt) > 44 else "")
    return "Nova conversa"


def migrate_chat(data, filename):
    changed = False
    for key, default in [
        ("id",            filename.replace(".json", "")),
        ("title",         make_title(data.get("messages", []))),
        ("pinned",        False),
        ("cover_image",   None),
        ("system_prompt", ""),
    ]:
        if key not in data:
            data[key] = default
            changed = True
    if changed:
        save_chat(filename, data)
    return data


def list_chats():
    try:
        files = sorted(
            [f for f in os.listdir(HISTORIES_DIR) if f.endswith(".json")],
            key=lambda f: os.path.getmtime(os.path.join(HISTORIES_DIR, f)),
            reverse=True,
        )
    except Exception:
        return []
    result = []
    for fname in files:
        try:
            result.append((fname, migrate_chat(load_chat(fname), fname)))
        except Exception:
            pass
    return [(f, d) for f, d in result if d.get("pinned")] + \
           [(f, d) for f, d in result if not d.get("pinned")]


def load_chat(filename):
    with open(os.path.join(HISTORIES_DIR, filename), "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_chat(filename, data):
    with open(os.path.join(HISTORIES_DIR, filename), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def new_chat_filename():
    return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json"


def delete_chat(filename, cover_path=None):
    for p in [os.path.join(HISTORIES_DIR, filename), cover_path]:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def toggle_pin(filename, data):
    data["pinned"] = not data.get("pinned", False)
    save_chat(filename, data)


def save_cover(uploaded_file, chat_id):
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext not in IMG_EXTS:
        ext = "png"
    for f in os.listdir(COVERS_DIR):
        if f.rsplit(".", 1)[0] == chat_id:
            try:
                os.remove(os.path.join(COVERS_DIR, f))
            except Exception:
                pass
    path = os.path.join(COVERS_DIR, f"{chat_id}.{ext}")
    with open(path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return path


def avatar_b64_src(cover_path):
    try:
        with open(cover_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        ext  = cover_path.rsplit(".", 1)[-1].lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{b64}"
    except Exception:
        return None


def avatar_html(title, cover=None, size=28):
    letter = (title.strip() or "N")[0].upper()
    if cover and os.path.exists(cover):
        src = avatar_b64_src(cover)
        if src:
            return (
                f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
                f'overflow:hidden;flex-shrink:0;">'
                f'<img src="{src}" style="width:100%;height:100%;object-fit:cover;"/></div>'
            )
    hue_map = {"A":210,"B":190,"C":150,"D":270,"E":30,"F":0,
               "G":120,"H":200,"I":60,"J":180,"K":300,"L":240,
               "M":330,"N":20,"O":160,"P":280,"Q":10,"R":350,
               "S":80,"T":170,"U":220,"V":310,"W":40,"X":260,
               "Y":100,"Z":140}
    hue = hue_map.get(letter, 220)
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:hsl({hue},18%,14%);display:flex;align-items:center;'
        f'justify-content:center;font-size:{size//2-1}px;'
        f'color:hsl({hue},38%,48%);font-weight:700;flex-shrink:0;">'
        f'{letter}</div>'
    )


def export_chat_markdown(data):
    title    = data.get("title", "Conversa")
    messages = data.get("messages", [])
    sp       = data.get("system_prompt", "")
    lines    = [f"# {title}\n", f"_Exportado em {datetime.now().strftime('%d/%m/%Y %H:%M')}_\n"]
    if sp:
        lines.append(f"\n> **System prompt:** {sp}\n")
    lines.append("\n---\n")
    for msg in messages:
        role    = "**Você**" if msg["role"] == "user" else "**Nahki**"
        content = msg["content"]
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        parts.append(p["text"])
                    elif p.get("type") == "image_url":
                        parts.append("_[imagem]_")
            content = "\n".join(parts) if parts else "[sem texto]"
        lines.append(f"{role}\n\n{content}\n\n---\n")
    return "\n".join(lines)


def build_messages_for_api(messages, system_prompt=""):
    api_msgs = []
    if system_prompt and system_prompt.strip():
        api_msgs.append({"role": "system", "content": system_prompt.strip()})
    for msg in messages:
        api_msgs.append({"role": msg["role"], "content": msg["content"]})
    return api_msgs


def export_all_backup():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(HISTORIES_DIR):
            if fname.endswith(".json"):
                zf.write(os.path.join(HISTORIES_DIR, fname), f"chat_histories/{fname}")
        for fname in os.listdir(COVERS_DIR):
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext in IMG_EXTS:
                zf.write(os.path.join(COVERS_DIR, fname), f"covers/{fname}")
    buf.seek(0)
    return buf.getvalue()


def import_chat_from_json(data):
    if not isinstance(data, dict) or "messages" not in data:
        return False, "Arquivo inválido: campo 'messages' não encontrado."
    if not isinstance(data["messages"], list):
        return False, "Arquivo inválido: 'messages' não é uma lista."
    new_fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json"
    save_chat(new_fname, {
        "id":            new_fname.replace(".json", ""),
        "title":         data.get("title") or make_title(data["messages"]),
        "pinned":        False,
        "cover_image":   None,
        "system_prompt": data.get("system_prompt", ""),
        "messages":      data["messages"],
    })
    return True, new_fname


def file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()[:16]


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nahki", page_icon="◈",
    layout="centered", initial_sidebar_state="auto",
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("current_chat_file",  None),
    ("messages",           []),
    ("last_model_used",    None),
    ("show_settings",      False),
    ("actions_open",       None),
    ("system_prompt",      ""),
    ("modal_editing_name", False),
    ("_imported_hashes",   set()),
    ("_last_media_hash",   None),
    # Pending attachment — cleared after send or cancel
    ("_pending_img",       None),  # dict: {bytes, ext, mime, data_url} or None
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Query-param navigation ─────────────────────────────────────────────────────
_action = st.query_params.to_dict().get("click", "").strip()
if _action:
    st.query_params.clear()

    if _action == "new_chat":
        st.session_state.current_chat_file  = None
        st.session_state.messages           = []
        st.session_state.last_model_used    = None
        st.session_state.actions_open       = None
        st.session_state.system_prompt      = ""
        st.session_state.modal_editing_name = False
        st.session_state._pending_img       = None

    elif _action == "toggle_settings":
        st.session_state.show_settings = not st.session_state.show_settings

    elif _action.startswith("open_"):
        fname = _action[5:]
        fpath = os.path.join(HISTORIES_DIR, fname)
        if os.path.exists(fpath):
            data = load_chat(fname)
            sp   = data.get("system_prompt", "")
            st.session_state.current_chat_file  = fname
            st.session_state.messages           = data.get("messages", [])
            st.session_state.last_model_used    = None
            st.session_state.actions_open       = None
            st.session_state.system_prompt      = sp
            st.session_state.modal_editing_name = False
            st.session_state._pending_img       = None

    elif _action.startswith("menu_"):
        fname = _action[5:]
        st.session_state.actions_open = (
            fname if st.session_state.actions_open != fname else None
        )

    elif _action == "close_menu":
        st.session_state.actions_open = None

    elif _action.startswith("pin_"):
        fname = _action[4:]
        if os.path.exists(os.path.join(HISTORIES_DIR, fname)):
            toggle_pin(fname, load_chat(fname))
        st.session_state.actions_open = None

    elif _action.startswith("del_"):
        fname = _action[4:]
        fpath = os.path.join(HISTORIES_DIR, fname)
        if os.path.exists(fpath):
            delete_chat(fname, load_chat(fname).get("cover_image"))
        if st.session_state.current_chat_file == fname:
            st.session_state.current_chat_file = None
            st.session_state.messages          = []
            st.session_state.last_model_used   = None
            st.session_state.system_prompt     = ""
            st.session_state._pending_img      = None
        st.session_state.actions_open = None

    st.rerun()


# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
    background:#0a0a0a!important;color:#e8e8e8!important;
}
[data-testid="stHeader"]{background:#0a0a0a!important;border-bottom:none!important;}
footer,[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}

[data-testid="stSidebar"],
[data-testid="stSidebar"]>div:first-child,
[data-testid="stSidebarContent"]{
    background:#000!important;border-right:none!important;
    padding-left:12px!important;padding-right:12px!important;
}
[data-testid="stSidebar"] *{color:#c0c0c0!important;}
[data-testid="stSidebar"] button{
    background:transparent!important;border:none!important;
    box-shadow:none!important;outline:none!important;
}
[data-testid="stSidebar"] button:focus,[data-testid="stSidebar"] button:active{
    background:transparent!important;border:none!important;
    box-shadow:none!important;outline:none!important;
}
[data-testid="stSidebar"] a{text-decoration:none!important;color:inherit!important;}
[data-testid="stSidebar"] a:hover{text-decoration:none!important;}

.nhk-icon-btn{
    display:inline-flex;align-items:center;justify-content:center;
    width:26px;height:26px;border-radius:6px;
    color:#666;transition:background 0.15s,color 0.15s;cursor:pointer;flex-shrink:0;
}
.nhk-icon-btn:hover{background:rgba(255,255,255,0.07)!important;color:#ccc!important;}

.nhk-row{
    display:flex;align-items:center;justify-content:space-between;
    padding:6px 4px;border-radius:7px;cursor:pointer;
    transition:background 0.12s;-webkit-tap-highlight-color:transparent;
    gap:8px;position:relative;
}
.nhk-row:hover{background:rgba(255,255,255,0.05)!important;}
.nhk-row.active{background:rgba(255,255,255,0.06)!important;}
.nhk-row-overlay{position:absolute;inset:0;z-index:1;border-radius:7px;}
.nhk-row-left{
    display:flex;align-items:center;gap:9px;
    min-width:0;flex:1;overflow:hidden;
    position:relative;z-index:0;pointer-events:none;
}
.nhk-row-name{
    font-size:0.76rem;color:#aaa;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;
}
.nhk-row-name.active{color:#fff;font-weight:600;}
.nhk-menu-link{
    display:inline-flex;align-items:center;justify-content:center;
    width:22px;height:22px;border-radius:5px;flex-shrink:0;
    color:#3a3a3a;position:relative;z-index:2;
}
.nhk-menu-link:hover{color:#888!important;}

.nhk-actions{display:flex;align-items:center;gap:4px;padding:3px 4px 5px 4px;}
.nhk-action-btn{
    display:inline-flex;align-items:center;gap:5px;
    font-size:0.71rem;color:#666;padding:4px 8px;
    border-radius:6px;border:1px solid #1e1e1e;
    transition:background 0.15s,color 0.15s;cursor:pointer;white-space:nowrap;
}
.nhk-action-btn:hover{background:#1a1a1a!important;color:#ccc!important;}
.nhk-action-btn.danger:hover{color:#e05555!important;border-color:#3a1a1a!important;background:#1a0a0a!important;}

.nhk-sep{height:1px;background:#0f0f0f;margin:8px 0;}
.nhk-settings{
    background:#080808;border:1px solid #181818;border-radius:9px;
    padding:10px 12px;margin:4px 0 8px 0;
}
.nhk-section-label{
    font-size:0.6rem;color:#2a2a2a;text-transform:uppercase;
    letter-spacing:0.12em;padding:4px 2px 2px 2px;margin-bottom:4px;
}

/* Sidebar search input */
[data-testid="stSidebar"] [data-testid="stTextInput"] label{display:none!important;}
[data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="input"]{
    background:#080808!important;border:1px solid #191919!important;
    border-radius:999px!important;box-shadow:none!important;min-height:32px!important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within{
    border:1px solid #252525!important;box-shadow:none!important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input{
    background:transparent!important;border:none!important;
    color:#b8b8b8!important;caret-color:#b8b8b8!important;
    font-size:0.78rem!important;padding:5px 14px!important;
    height:32px!important;box-shadow:none!important;outline:none!important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder{color:#2e2e2e!important;}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus{
    outline:none!important;box-shadow:none!important;border:none!important;
}
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within *{
    box-shadow:none!important;outline:none!important;
}

*:focus,*:focus-visible,*:focus-within{outline:none!important;box-shadow:none!important;}

.block-container{max-width:760px!important;padding-top:0!important;padding-bottom:5rem!important;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
    background:#161616!important;border-radius:16px!important;
    padding:12px 16px!important;margin-bottom:6px!important;border:1px solid #222!important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
    background:transparent!important;border-radius:16px!important;
    padding:12px 16px!important;margin-bottom:6px!important;border:1px solid transparent!important;
}
[data-testid="stChatMessageAvatarUser"]{background:#222!important;}
[data-testid="stChatMessageAvatarAssistant"]{background:#0d1b2a!important;}
[data-testid="stChatInput"]{
    border-radius:999px!important;background:#111!important;
    border:1px solid #242424!important;box-shadow:none!important;
}
[data-testid="stChatInput"]:focus-within{border:1px solid #2a2a2a!important;box-shadow:none!important;}
[data-testid="stChatInput"] textarea{background:transparent!important;color:#e8e8e8!important;caret-color:#e8e8e8!important;}
[data-testid="stChatInput"] textarea::placeholder{color:#444!important;}
[data-testid="baseButton-secondary"]{
    background:#161616!important;color:#bbb!important;
    border:1px solid #242424!important;border-radius:8px!important;
}
[data-testid="baseButton-secondary"]:hover{
    background:#1e1e1e!important;border-color:#333!important;color:#fff!important;
}
hr{border-color:#111!important;}
.stCaption,[data-testid="stCaptionContainer"]{color:#383838!important;}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:#000;}
::-webkit-scrollbar-thumb{background:#1a1a1a;border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:#252525;}
[data-testid="stExpander"]{
    background:#0f0f0f!important;border:1px solid #1a1a1a!important;border-radius:10px!important;
}

/* ── Minimal attachment uploader — scoped to main content (not sidebar) ── */
.block-container [data-testid="stFileUploader"]{
    border:none!important;background:transparent!important;
    padding:0!important;margin:0!important;
}
.block-container [data-testid="stFileUploaderDropzone"]{
    border:none!important;padding:0!important;
    min-height:0!important;background:transparent!important;
    display:flex!important;align-items:center!important;
    gap:0!important;
}
/* Hide "Drag and drop / limit" instruction text */
.block-container [data-testid="stFileUploaderDropzoneInstructions"]{
    display:none!important;
}
/* Hide file name row after selection (we show our own thumbnail) */
.block-container [data-testid="stFileUploaderFileName"],
.block-container [data-testid="stFileUploaderDeleteBtn"],
.block-container [data-testid="stFileUploader"] small,
.block-container [data-testid="stFileUploader"] section > div:last-child{
    display:none!important;
}
/* Browse button → tiny clip-icon button */
.block-container [data-testid="stFileUploader"] button{
    background:transparent!important;
    border:1px solid #1e1e1e!important;
    border-radius:8px!important;
    color:#3a3a3a!important;
    padding:0!important;
    font-size:0!important;
    line-height:0!important;
    min-height:0!important;
    height:30px!important;
    width:30px!important;
    display:inline-flex!important;
    align-items:center!important;
    justify-content:center!important;
    transition:color 0.15s,border-color 0.15s!important;
    flex-shrink:0!important;
}
/* Show clip icon via pseudo-element */
.block-container [data-testid="stFileUploader"] button::before{
    content:"📎";
    font-size:14px!important;
    line-height:1!important;
}
.block-container [data-testid="stFileUploader"] button:hover{
    color:#888!important;border-color:#2a2a2a!important;background:transparent!important;
}
/* Hide any text/span inside the button */
.block-container [data-testid="stFileUploader"] button > *{
    display:none!important;
}

/* ── Thumbnail preview strip ── */
.nhk-thumb-strip{
    display:flex;align-items:center;gap:8px;
    padding:6px 2px 2px 2px;
}
.nhk-thumb{
    width:52px;height:52px;border-radius:8px;
    object-fit:cover;border:1px solid #2a2a2a;
    display:block;flex-shrink:0;
}
.nhk-thumb-cancel{
    display:inline-flex;align-items:center;justify-content:center;
    width:18px;height:18px;border-radius:50%;
    background:#1e1e1e;border:1px solid #2a2a2a;
    color:#555;font-size:0.6rem;cursor:pointer;
    line-height:1;flex-shrink:0;
    transition:background 0.15s,color 0.15s;
}
.nhk-thumb-cancel:hover{background:#2a1a1a!important;color:#e05555!important;}
.nhk-thumb-name{
    font-size:0.7rem;color:#444;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px;
}

/* ── Sidebar download button ── */
[data-testid="stDownloadButton"] button{
    background:#111!important;color:#888!important;
    border:1px solid #1e1e1e!important;border-radius:8px!important;
    font-size:0.78rem!important;
}
[data-testid="stDownloadButton"] button:hover{
    background:#181818!important;color:#ccc!important;border-color:#2a2a2a!important;
}

/* ── Dialog / modal ── */
[data-testid="stDialog"]{
    background:#0d0d0d!important;border:1px solid #1e1e1e!important;border-radius:14px!important;
}
[data-testid="stDialog"] *{color:#c0c0c0!important;}
[data-testid="stDialog"] button{
    background:#141414!important;border:1px solid #222!important;
    border-radius:8px!important;color:#999!important;
}
[data-testid="stDialog"] button:hover{color:#fff!important;border-color:#333!important;}
/* System prompt textarea inside modal */
[data-testid="stDialog"] textarea{
    background:#080808!important;border:1px solid #1e1e1e!important;
    border-radius:8px!important;color:#aaa!important;
    font-size:0.78rem!important;caret-color:#aaa!important;
    resize:vertical!important;
}
[data-testid="stDialog"] textarea:focus{
    border:1px solid #2a2a2a!important;box-shadow:none!important;
}

/* ── Sidebar import uploader ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"]{
    background:#0a0a0a!important;border:1px dashed #1a1a1a!important;border-radius:8px!important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:transparent!important;}
</style>
""", unsafe_allow_html=True)


# ── Modal: Perfil do Chat ──────────────────────────────────────────────────────
@st.dialog("Perfil do Chat", width="small")
def chat_profile_modal():
    fname = st.session_state.current_chat_file
    if not fname:
        st.write("Nenhum chat aberto.")
        return
    fpath = os.path.join(HISTORIES_DIR, fname)
    if not os.path.exists(fpath):
        st.write("Chat não encontrado.")
        return

    data  = load_chat(fname)
    title = data.get("title", "Nova conversa")
    cover = data.get("cover_image")

    # ── Avatar ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;justify-content:center;margin-bottom:8px;">'
        f'{avatar_html(title, cover, size=72)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="text-align:center;font-size:0.92rem;font-weight:600;'
        f'color:#ddd;margin:0 0 10px 0;">{title}</p>',
        unsafe_allow_html=True,
    )

    # ── Cover upload ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Foto de capa",
        type=["png","jpg","jpeg","webp"],
        key="modal_cover_upload",
        label_visibility="visible",
    )
    if uploaded is not None:
        h = file_hash(uploaded)
        if st.session_state.get("_modal_cover_hash") != h:
            st.session_state["_modal_cover_hash"] = h
            cover_path = save_cover(uploaded, data.get("id", fname.replace(".json","")))
            data["cover_image"] = cover_path
            save_chat(fname, data)
            st.success("Foto atualizada!")
            st.rerun()

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        pin_label = "📌 Desafixar" if data.get("pinned") else "📌 Fixar"
        if st.button(pin_label, use_container_width=True):
            toggle_pin(fname, data)
            st.rerun()
    with col2:
        if st.button("✏️ Renomear", use_container_width=True):
            st.session_state.modal_editing_name = True
            st.rerun()
    with col3:
        st.download_button(
            "📤 Exportar",
            data=export_chat_markdown(data).encode("utf-8"),
            file_name=f"{title[:30].strip()}.md",
            mime="text/markdown",
            use_container_width=True,
            key="modal_export_btn",
        )

    # ── Rename inline ─────────────────────────────────────────────────────────
    if st.session_state.modal_editing_name:
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
        new_name = st.text_input("Novo nome", value=title, key="modal_new_name")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✓ Salvar", use_container_width=True, key="modal_save_name"):
                data["title"] = new_name.strip() or title
                save_chat(fname, data)
                st.session_state.modal_editing_name = False
                st.rerun()
        with c2:
            if st.button("✕ Cancelar", use_container_width=True, key="modal_cancel_name"):
                st.session_state.modal_editing_name = False
                st.rerun()

    # ── Remove cover ──────────────────────────────────────────────────────────
    if cover and os.path.exists(cover):
        st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)
        if st.button("🗑 Remover foto de capa", use_container_width=True):
            try:
                os.remove(cover)
            except Exception:
                pass
            data["cover_image"] = None
            save_chat(fname, data)
            st.session_state["_modal_cover_hash"] = None
            st.rerun()

    # ── System Prompt ─────────────────────────────────────────────────────────
    st.markdown(
        '<hr style="border-color:#1a1a1a;margin:12px 0 10px 0;"/>'
        '<p style="font-size:0.62rem;color:#333;text-transform:uppercase;'
        'letter-spacing:0.12em;margin:0 0 6px 0;">Personalidade da IA</p>',
        unsafe_allow_html=True,
    )
    sp_current = st.session_state.system_prompt
    new_sp = st.text_area(
        "Personalidade",
        value=sp_current,
        placeholder="Ex: Você é uma assistente direta e objetiva. Responda sempre em português...",
        height=100,
        label_visibility="collapsed",
        key="modal_system_prompt",
    )
    if st.button("💾 Salvar personalidade", use_container_width=True):
        st.session_state.system_prompt = new_sp.strip()
        # Persist to current chat file
        data_fresh = load_chat(fname)
        data_fresh["system_prompt"] = new_sp.strip()
        save_chat(fname, data_fresh)
        st.success("Personalidade salva!")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Header
    gear_style = "color:#4a9eff!important;" if st.session_state.show_settings else ""
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:10px 0 8px 0;">'
        f'  <span style="font-size:0.92rem;font-weight:600;color:#fff;'
        f'letter-spacing:0.01em;">Chats</span>'
        f'  <div style="display:flex;align-items:center;gap:2px;">'
        f'    <a href="?click=new_chat" target="_self">'
        f'      <div class="nhk-icon-btn">{SVG_NEW}</div></a>'
        f'    <a href="?click=toggle_settings" target="_self">'
        f'      <div class="nhk-icon-btn" style="{gear_style}">{SVG_GEAR}</div></a>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Settings panel
    if st.session_state.show_settings:
        names = [p["name"] for p in get_providers()]
        st.markdown(
            f'<div class="nhk-settings">'
            f'<p style="font-size:0.65rem;color:#333;margin:0 0 4px 0;'
            f'text-transform:uppercase;letter-spacing:0.1em;">Provedores ativos</p>'
            f'<p style="font-size:0.78rem;color:#777;margin:0;">'
            f'{", ".join(names) if names else "⚠️ Nenhum — configure as chaves nos Secrets"}'
            f'</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="nhk-sep"></div>', unsafe_allow_html=True)

    # Search
    search_query = st.text_input(
        "search", placeholder="Pesquisar...",
        label_visibility="collapsed", key="sidebar_search",
    )

    st.markdown('<div class="nhk-sep"></div>', unsafe_allow_html=True)

    # Chat list
    all_chats = list_chats()
    query     = search_query.strip().lower()
    filtered  = [
        (f, d) for f, d in all_chats
        if not query or query in d.get("title", "").lower()
    ]

    if not filtered:
        st.markdown(
            f'<p style="font-size:0.73rem;color:#2a2a2a;padding:6px 2px;">'
            f'{"Nenhum resultado." if query else "Nenhuma conversa ainda."}</p>',
            unsafe_allow_html=True,
        )
    else:
        rows_html = []
        for fname, data in filtered:
            title     = data.get("title", "Nova conversa")
            is_pinned = data.get("pinned", False)
            is_active = st.session_state.current_chat_file == fname
            cover     = data.get("cover_image")
            menu_open = st.session_state.actions_open == fname

            active_cls = " active" if is_active else ""
            name_cls   = "nhk-row-name active" if is_active else "nhk-row-name"
            pin_dot    = (
                f'<span style="font-size:0.55rem;color:#4a9eff;margin-right:3px;'
                f'vertical-align:middle;">●</span>' if is_pinned else ""
            )

            rows_html.append(
                f'<div class="nhk-row{active_cls}">'
                f'  <a href="?click=open_{fname}" target="_self" class="nhk-row-overlay"></a>'
                f'  <div class="nhk-row-left">'
                f'    {avatar_html(title, cover, size=28)}'
                f'    <span class="{name_cls}">{pin_dot}{title}</span>'
                f'  </div>'
                f'  <a href="?click=menu_{fname}" target="_self" class="nhk-menu-link">'
                f'    {SVG_DOTS}</a>'
                f'</div>'
            )

            if menu_open:
                pin_svg   = SVG_UNPIN if is_pinned else SVG_PIN
                pin_label = "Desafixar" if is_pinned else "Fixar"
                rows_html.append(
                    f'<div class="nhk-actions">'
                    f'  <a href="?click=pin_{fname}" target="_self">'
                    f'    <div class="nhk-action-btn">{pin_svg} {pin_label}</div></a>'
                    f'  <a href="?click=del_{fname}" target="_self">'
                    f'    <div class="nhk-action-btn danger">{SVG_TRASH} Excluir</div></a>'
                    f'  <a href="?click=close_menu" target="_self">'
                    f'    <div class="nhk-action-btn" style="padding:4px 6px;">✕</div></a>'
                    f'</div>'
                )

        st.markdown("\n".join(rows_html), unsafe_allow_html=True)

    st.markdown('<div class="nhk-sep" style="margin-top:16px;"></div>', unsafe_allow_html=True)

    # Backup
    st.markdown('<div class="nhk-section-label">Backup</div>', unsafe_allow_html=True)
    st.download_button(
        "⬇ Exportar todos os chats (.zip)",
        data=export_all_backup(),
        file_name=f"nahki_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True,
        key="export_backup_btn",
    )
    st.markdown(
        '<p style="font-size:0.7rem;color:#2a2a2a;margin:6px 0 2px 0;">'
        'Importar conversa (.json):</p>',
        unsafe_allow_html=True,
    )
    import_file = st.file_uploader(
        "Importar JSON", type=["json"],
        key="import_chat_upload", label_visibility="collapsed",
    )
    if import_file is not None:
        h = file_hash(import_file)
        if h not in st.session_state._imported_hashes:
            try:
                raw = json.loads(import_file.getvalue().decode("utf-8"))
                ok, info = import_chat_from_json(raw)
                if ok:
                    st.session_state._imported_hashes.add(h)
                    st.success("Conversa importada!")
                    st.rerun()
                else:
                    st.error(info)
            except json.JSONDecodeError:
                st.error("Arquivo JSON inválido.")
            except Exception as e:
                st.error(f"Erro: {e}")
        else:
            st.caption("✓ Já importado")


# ── Main area ──────────────────────────────────────────────────────────────────

# Chat header
if st.session_state.current_chat_file:
    fname = st.session_state.current_chat_file
    fpath = os.path.join(HISTORIES_DIR, fname)
    hdata  = load_chat(fname) if os.path.exists(fpath) else {}
    htitle = hdata.get("title", "Nova conversa")
    hcover = hdata.get("cover_image")

    col_av, col_name, col_btn = st.columns([1, 8, 2])
    with col_av:
        st.markdown(avatar_html(htitle, hcover, size=32), unsafe_allow_html=True)
    with col_name:
        st.markdown(
            f'<div style="padding-top:6px;font-size:0.88rem;font-weight:600;'
            f'color:#ddd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
            f'{htitle}</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("⋯ Perfil", key="open_profile_modal"):
            st.session_state.modal_editing_name = False
            st.session_state["_modal_cover_hash"] = None
            chat_profile_modal()

    st.markdown(
        '<div style="height:1px;background:#111;margin:8px 0 12px 0;"></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div style='text-align:center;padding:2.5rem 0 1rem 0;'>"
        "<div style='font-size:2.2rem;margin-bottom:0.75rem;color:#1f1f1f;'>◈</div>"
        "<div style='font-size:1rem;color:#333;margin-bottom:0.25rem;'>Olá, sou a Nahki.</div>"
        "<div style='font-size:0.8rem;color:#2a2a2a;'>Como posso ajudar você hoje?</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# Watermark + model
st.markdown(
    "<div style='text-align:center;padding:0.4rem 0 0.2rem 0;'>"
    "<span style='font-size:0.88rem;font-weight:500;letter-spacing:0.22em;"
    "color:#1e1e1e;text-transform:uppercase;'>Nahki</span></div>",
    unsafe_allow_html=True,
)
if st.session_state.last_model_used:
    st.markdown(
        f"<div style='text-align:center;margin-bottom:0.4rem;'>"
        f"<span style='font-size:0.67rem;color:#282828;'>"
        f"{st.session_state.last_model_used}</span></div>",
        unsafe_allow_html=True,
    )

# Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    txt = part["text"]
                    if txt and txt != "[Imagem enviada]":
                        st.markdown(txt)
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        st.image(url)
        else:
            st.markdown(content)

# ── Attachment area ────────────────────────────────────────────────────────────
# Thumbnail preview row (shown while an image is pending, before send)
if st.session_state._pending_img is not None:
    pending = st.session_state._pending_img
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'padding:4px 2px 6px 2px;">'
        f'  <img src="{pending["data_url"]}" '
        f'       style="width:48px;height:48px;border-radius:8px;'
        f'              object-fit:cover;border:1px solid #2a2a2a;'
        f'              display:block;flex-shrink:0;"/>'
        f'  <span style="font-size:0.7rem;color:#444;overflow:hidden;'
        f'               text-overflow:ellipsis;white-space:nowrap;'
        f'               max-width:200px;flex:1;">{pending["name"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("✕  Remover imagem", key="cancel_attachment"):
        st.session_state._pending_img = None
        st.rerun()

# File uploader — rendered in a narrow column so CSS can shrink it to clip-icon size
_attach_col, _ = st.columns([1, 14])
with _attach_col:
    uploaded_media = st.file_uploader(
        "📎",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="chat_media_upload",
        label_visibility="collapsed",
    )

# Detect new file → store as pending, trigger rerun to show thumbnail
if uploaded_media is not None:
    new_hash = file_hash(uploaded_media)
    if new_hash != st.session_state._last_media_hash:
        img_bytes = uploaded_media.getvalue()
        ext  = uploaded_media.name.rsplit(".", 1)[-1].lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        b64  = base64.b64encode(img_bytes).decode()
        st.session_state._pending_img = {
            "bytes":    img_bytes,
            "ext":      ext,
            "mime":     mime,
            "data_url": f"data:image/{mime};base64,{b64}",
            "name":     uploaded_media.name,
        }
        st.session_state._last_media_hash = new_hash
        st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Mensagem para Nahki...")

# Send when user submits (Enter): requires text OR pending image
if user_input is not None and (user_input.strip() or st.session_state._pending_img is not None):
    pending = st.session_state._pending_img

    # Build user content
    if pending is not None:
        user_content = [
            {"type": "image_url", "image_url": {"url": pending["data_url"]}},
        ]
        text = user_input.strip() if user_input else ""
        user_content.insert(0, {"type": "text", "text": text if text else "[Imagem enviada]"})
    else:
        user_content = user_input.strip()

    # Clear pending
    st.session_state._pending_img = None

    st.session_state.messages.append({"role": "user", "content": user_content})

    with st.chat_message("user"):
        if isinstance(user_content, list):
            for part in user_content:
                if part.get("type") == "text" and part["text"] != "[Imagem enviada]":
                    st.markdown(part["text"])
                elif part.get("type") == "image_url":
                    st.image(part["image_url"]["url"])
        else:
            st.markdown(user_content)

    with st.chat_message("assistant"):
        with st.spinner(""):
            api_msgs    = build_messages_for_api(
                st.session_state.messages, st.session_state.system_prompt
            )
            reply, used = call_with_fallback(api_msgs)
            st.session_state.last_model_used = used
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Save chat
    if st.session_state.current_chat_file is None:
        st.session_state.current_chat_file = new_chat_filename()

    fname  = st.session_state.current_chat_file
    fpath  = os.path.join(HISTORIES_DIR, fname)
    prev   = load_chat(fname) if os.path.exists(fpath) else {}
    save_chat(fname, {
        "id":            prev.get("id",          fname.replace(".json", "")),
        "title":         make_title(st.session_state.messages),
        "pinned":        prev.get("pinned",      False),
        "cover_image":   prev.get("cover_image", None),
        "system_prompt": st.session_state.system_prompt,
        "messages":      st.session_state.messages,
    })
    st.rerun()
