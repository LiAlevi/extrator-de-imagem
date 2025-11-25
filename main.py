import streamlit as st
from PIL import Image
import tempfile
import os
import re
import json
from html import escape
import base64
from openai import OpenAI

# =========================
# Configurar página
# =========================
st.set_page_config(page_title="OCR com GPT-4o (multi-página)", layout="wide")
st.title("📸 Extrator de Texto de Imagens com GPT-4o")
st.write("Envie 1 ou 2 imagens de páginas e gere HTML com a formatação visual detectada.")

# =========================
# Utilidades
# =========================
def markdown_inline_to_html(text: str) -> str:
    """
    Converte ***bold+italic***, **bold** e *italic* em HTML.
    NÃO inventa formatação, apenas converte o que já vier com markdown.
    """
    if text is None:
        return ""

    # escapa caracteres HTML perigosos
    s = escape(text)

    # ***bold+italic***
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", s)

    # **bold**
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

    # *italic*
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)

    return s


def extract_json_from_model_output(text: str):
    """Extrai JSON mesmo se vier com ruído ou ```json ...```."""
    if not text:
        raise json.JSONDecodeError("empty", "", 0)

    s = text.strip()

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if fence:
        s = fence.group(1).strip()
    else:
        if s.startswith("["):
            a, b = s.find("["), s.rfind("]")
            s = s[a:b+1]
        else:
            a, b = s.find("{"), s.rfind("}")
            s = s[a:b+1]

    return json.loads(s)


def coerce_to_sections_schema(data):
    """Normaliza retorno para o formato { sections: [...] }."""
    if isinstance(data, dict) and "sections" in data:
        # compat: alguns modelos usam "content" em vez de "text"
        for sec in data["sections"]:
            for it in sec.get("items", []):
                if it.get("text") is None and it.get("content") is not None:
                    it["text"] = it["content"]
        return data

    if isinstance(data, list):
        items = []
        for it in data:
            if not isinstance(it, dict):
                continue
            txt = it.get("text") or it.get("content") or ""
            items.append({"type": (it.get("type") or "p"), "text": txt})

        heading = ""
        if items and items[0]["type"] in {"h1", "h2", "h3"}:
            heading = items[0]["text"]
            items = items[1:]

        return {"sections": [{"heading": heading, "type": "h2", "items": items}]}

    return {"sections": []}

# =========================
# Carregar OpenAI
# =========================
@st.cache_resource
def load_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("❌ OPENAI_API_KEY não definida. Configure antes de continuar.")
        st.stop()
    return OpenAI(api_key=key)

client = load_openai()

# =========================
# GPT-4o (multi-page)
# =========================
def analyze_text_formatting(image_paths: list[str]):
    # prompt mais rígido para respeitar a formatação visual
    prompt = """
TASK:
Analyze the IMAGE(S) and reproduce ONLY the REAL inline formatting exactly as it appears.
Multiple images represent consecutive pages of ONE document. Read them in order (page 1 → page 2).

INLINE FORMATTING RULES:
1) Bold → use **bold** ONLY if the text is visually thicker/darker.
2) Italic → use *italic* ONLY if the text is clearly slanted.
3) Bold+Italic → use ***bold+italic*** ONLY if BOTH bold and italic are clearly visible.
4) If the font looks normal (no slant, no extra thickness), output plain text (no *, ** or ***).
5) NEVER infer formatting from semantics or from your understanding of English. Only from what you SEE.

SPECIAL FORMATTING RULES:
1) Section Titles/Headers (Before Class, Learning Objectives, Cross-curricular skills, Materials needed, Classroom Arrangement, etc.) → ALWAYS use **bold**
2) Individual Letters when appearing alone → ALWAYS use *italic* (Example: *t*, *f*, *s*)
3) Song/Music Names → ALWAYS use *italic* (Example: *Hello and Goodbye songs*)
4) Labels with formatting (Arts:, Music:, P.E.:) → Apply formatting exactly as shown in image

HEADINGS:
- If text is visually a section title (larger font or clearly a heading), return it as a separate item or heading.
- Prefer `"type": "h2"` for main section headings like "**Before Class.**", "**Learning Objectives:**", etc.
- Do NOT mark headings as italic, even if the font is slightly stylized.

BULLETS:
- Mark an item as `"type": "li"` ONLY if there is a real bullet dot "•" or equivalent list marker in the image.
- If there is NO bullet marker, use `"type": "p"` instead, even if it looks like a list.
- The bullet symbol "•" should NOT appear in the text itself, only as list structure.

LETTERS AND SHORT WORDS:
- Preserve formatting of individual letters or short words exactly as in the image.
  Example: in "Say it right! *t*, *f*, and *s*", output exactly like this with italic letters.

OUTPUT:
Return ONLY valid JSON in this exact structure:

{
  "sections": [
    {
      "heading": "**Title if present**",
      "type": "h2",
      "items": [
        { "text": "inline formatted content", "type": "p" },
        { "text": "**Can sing along to the *Hello and Goodbye songs*.**", "type": "li" }
      ]
    }
  ]
}
"""

    content = [{"type": "input_text", "text": prompt}]

    # append pages in order
    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})

    resp = client.responses.create(
        model="gpt-4o",
        temperature=0,
        input=[{"role": "user", "content": content}],
    )

    # tenta pegar texto direto
    response_text = getattr(resp, "output_text", None)

    if not response_text:
        try:
            blocks = resp.output[0].content
            texts = [b.text for b in blocks if getattr(b, "type", "") == "output_text"]
            response_text = "\n".join(texts).strip() if texts else None
        except Exception:
            response_text = None

    if not response_text:
        st.error("❌ Resposta vazia da API.")
        return None

    try:
        raw = extract_json_from_model_output(response_text)
        return coerce_to_sections_schema(raw)
    except Exception:
        st.error("❌ Não foi possível extrair JSON.")
        st.code(response_text)
        return None

# =========================
# HTML
# =========================
def build_html_from_sections(data):
    if not data or "sections" not in data:
        return "<p>Erro ao processar dados</p>"

    html = []

    for section in data["sections"]:
        # ==== TÍTULO DA SEÇÃO ====
        heading = markdown_inline_to_html(section.get("heading", ""))
        if heading:
            # usamos <p><strong>...</strong></p> em vez de <h2>
            html.append(f"<p><strong>{heading}</strong></p>")

        # ==== LISTA DE ITENS ====
        buffer = []  # lista temporária de <li>

        def flush():
            """Descarrega os <li> acumulados em um <ul> e limpa o buffer."""
            nonlocal buffer
            if buffer:
                html.append("<ul>")
                html.extend(buffer)
                html.append("</ul>")
                buffer = []

        for item in section.get("items", []):
            raw_text = item.get("text", "") or ""
            text = markdown_inline_to_html(raw_text)
            t = item.get("type", "p")

            if t == "li":
                # remove um possível símbolo de bullet visual no início (•)
                cleaned = re.sub(r"^\s*•\s*", "", text).strip()
                buffer.append(f"<li>{cleaned}</li>")
            else:
                # antes de adicionar um parágrafo, fecha qualquer <ul> aberto
                flush()
                html.append(f"<p>{text}</p>")

        # se terminar a seção ainda com itens no buffer, fecha o <ul>
        flush()

    return "\n".join(html)

# =========================
# UI
# =========================
st.subheader("1) Envie a primeira página")
st.warning("⚠️ **A ordem importa** — Página 1 deve vir primeiro. Se enviar fora de ordem, use a opção 'Inverter ordem'.", icon="⚠️")

col1, col2 = st.columns(2)
with col1:
    img1 = st.file_uploader("Página 1 (obrigatória)", type=["jpg", "jpeg", "png", "bmp"])
with col2:
    img2 = None
    if img1:
        img2 = st.file_uploader("Página 2 (opcional)", type=["jpg", "jpeg", "png", "bmp"])

if img1:
    st.write("### Pré-visualização das páginas (na ordem atual)")
    c1, c2 = st.columns(2)
    with c1:
        st.image(Image.open(img1), caption="Página 1", use_column_width=True)
    with c2:
        if img2:
            st.image(Image.open(img2), caption="Página 2", use_column_width=True)
        else:
            st.info("Nenhuma Página 2 enviada.")

invert = img1 and st.checkbox("🗂️ Inverter ordem (Página 2 → Página 1)", value=False)

st.write("---")
run = st.button("🚀 Gerar HTML")

if run:
    if not img1:
        st.error("Envie ao menos a Página 1.")
    else:
        # Decide order
        files = [img1, img2] if not invert else [img2, img1]
        files = [f for f in files if f]  # remove None

        # Save temp files
        paths = []
        for file in files:
            img = Image.open(file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp.name)
                paths.append(tmp.name)

        try:
            st.info("🤖 Analisando com GPT-4o...")
            result = analyze_text_formatting(paths)

            if result:
                st.success("✅ Processado!")
                with st.expander("📄 JSON retornado"):
                    st.json(result)

                html = build_html_from_sections(result)
                st.subheader("📝 Visualização")
                st.markdown(html, unsafe_allow_html=True)

                st.write("---")
                st.subheader("📋 HTML para copiar")
                st.code(html, language="html")
            else:
                st.error("❌ Erro no processamento.")
        finally:
            for p in paths:
                if os.path.exists(p):
                    os.remove(p)
