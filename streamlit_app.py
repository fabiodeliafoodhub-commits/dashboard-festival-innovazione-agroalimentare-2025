import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from fpdf import FPDF
from PIL import Image  # necessario per il logo

# ----------------------------
# Configurazione pagina
# ----------------------------

PAGE_TITLE = (
    "Dashboard personale - Partecipanti alla mia sessione "
    "del Festival dell'Innovazione Agroalimentare"
)

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide"
)

# ----------------------------
# BANNER CON LOGO E TITOLO FOOD HUB
# ----------------------------

col_logo, col_title = st.columns([1, 5])

with col_logo:
    st.image(
        "assets/logo_foodhub.png",   # percorso corretto
        width=120
    )

with col_title:
    st.markdown(
        """
        <h1 style="margin-bottom:0px; color:#73b27d;">
            Dashboard personale – Sessione del Festival dell’Innovazione Agroalimentare
        </h1>
        <h4 style="margin-top:0px; color:#555;">
            Analisi professionale dei partecipanti · Profilazione · Insights
        </h4>
        """,
        unsafe_allow_html=True
    )

st.write("")   # piccolo spazio tra banner e contenuto

# ----------------------------
# Impostazioni analisi
# ----------------------------

ALL_PROFILING_CATEGORIES = [
    "Occupazione",
    "Tipologia di organizzazione presso cui lavori",
    "Organizzazione presso cui lavori o studi",
    "Seniority",
    "Area aziendale",
    "Settore produttivo",
]

ORGANIZATION_COLUMN = "Organizzazione presso cui lavori o studi"

CATEGORIES_WITH_CHARTS = [
    c for c in ALL_PROFILING_CATEGORIES if c != ORGANIZATION_COLUMN
]

FILTER_COLUMNS = [
    "Occupazione",
    "Tipologia di organizzazione presso cui lavori",
    "Seniority",
    "Area aziendale",
    "Settore produttivo",
]

BRAND_COLORS = ["#73b27d", "#f1ad72", "#d31048"]


def get_colors_for_bars(num_bars: int, chart_index: int, values=None):
    """
    - se num_bars <= 3: usa i 3 colori brand alternati
    - se num_bars > 3: usa un solo colore brand con gradiente di alpha
      (più trasparente per i valori più piccoli)
    """
    if num_bars <= 3:
        return [BRAND_COLORS[i % len(BRAND_COLORS)] for i in range(num_bars)]

    base_hex = BRAND_COLORS[chart_index % len(BRAND_COLORS)]
    base_rgba = mcolors.to_rgba(base_hex)

    if values is None or len(values) != num_bars:
        alphas = [1.0 for _ in range(num_bars)]
    else:
        vals = pd.Series(values, dtype=float)
        v_min = vals.min()
        v_max = vals.max()
        if v_max == v_min or np.isnan(v_max):
            alphas = [1.0 for _ in range(num_bars)]
        else:
            norm = (vals - v_min) / (v_max - v_min)
            alphas = 0.3 + 0.7 * norm

    colors = [
        (base_rgba[0], base_rgba[1], base_rgba[2], float(a))
        for a in alphas
    ]
    return colors


def find_column_case_insensitive(df: pd.DataFrame, target_name: str):
    for col in df.columns:
        if col.strip().lower() == target_name.strip().lower():
            return col
    return None


def normalize_role(role: str) -> str:
    """
    Normalizza job title simili in etichette più aggregate.
    Esempi:
    - R&D, ricerca, research -> 'Ricerca e Sviluppo (R&D)'
    - quality, qualità -> 'Qualità / Quality Assurance'
    - marketing -> 'Marketing'
    - innovation / innovazione -> 'Innovazione'
    ecc.
    """
    r = role.strip().lower()

    # pattern di normalizzazione (molto semplice ma efficace)
    if any(k in r for k in ["r&d", "ricerca", "research", "sviluppo prodotto", "product development"]):
        return "Ricerca e Sviluppo (R&D)"
    if any(k in r for k in ["quality", "qualità", "controllo qualità", "qa"]):
        return "Qualità / Quality Assurance"
    if "marketing" in r:
        return "Marketing"
    if any(k in r for k in ["innovation", "innovazione"]):
        return "Innovazione"
    if any(k in r for k in ["sales", "commerciale", "vendite", "account manager"]):
        return "Sales / Commerciale"
    if any(k in r for k in ["supply chain", "logistica"]):
        return "Supply Chain / Logistica"
    if any(k in r for k in ["direttore", "director", "ceo", "cfo", "cto", "chief"]):
        return "Top Management"
    if any(k in r for k in ["regolatorio", "regulatory", "affari regolatori"]):
        return "Affari Regolatori"
    if any(k in r for k in ["ricercatore", "professore", "phd", "ricerca accademica"]):
        return "Ricerca Accademica"

    # fallback: capitalizza in modo decente
    return r.title()


def generate_pdf_report(df: pd.DataFrame,
                        kpis: dict,
                        top_roles: list | None = None,
                        top_sectors: list | None = None) -> bytes:
    """
    Crea un PDF semplice con:
    - titolo
    - KPI principali
    - top ruoli normalizzati (escludendo 'Studio')
    - top settori produttivi
    Restituisce i bytes del PDF.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Titolo
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Report partecipanti - Festival dell'Innovazione Agroalimentare", ln=True)
    pdf.ln(5)

    # KPI
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "KPI principali", ln=True)
    pdf.set_font("Arial", "", 11)
    for label, value in kpis.items():
        pdf.cell(0, 7, f"- {label}: {value}", ln=True)

    # Top ruoli
    if top_roles:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Top ruoli (aggregati, esclusi studenti)", ln=True)
        pdf.set_font("Arial", "", 11)
        for role, count in top_roles:
            line = f"- {role}: {count} partecipanti"
            pdf.cell(0, 7, line[:120], ln=True)

    # Top settori
    if top_sectors:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Top settori produttivi", ln=True)
        pdf.set_font("Arial", "", 11)
        for sector, count in top_sectors:
            line = f"- {sector}: {count} partecipanti"
            pdf.cell(0, 7, line[:120], ln=True)

    pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
    return pdf_bytes


# ----------------------------
# Upload file
# ----------------------------
uploaded_file = st.file_uploader(
    "Scegli un file Excel (.xlsx o .xls)",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_excel(uploaded_file)

        st.success("File caricato correttamente!")

        # ----------------------------
        # KPI principali
        # ----------------------------
        total_participants = len(df_uploaded)

        org_col_present = ORGANIZATION_COLUMN in df_uploaded.columns
        unique_orgs = (
            int(df_uploaded[ORGANIZATION_COLUMN].nunique())
            if org_col_present else None
        )

        sectors_col = "Settore produttivo"
        sectors_present = sectors_col in df_uploaded.columns
        unique_sectors = (
            int(df_uploaded[sectors_col].nunique())
            if sectors_present else None
        )

        seniority_col = "Seniority"
        top_seniority_label = None
        top_seniority_pct = None
        if seniority_col in df_uploaded.columns:
            counts_seniority = df_uploaded[seniority_col].value_counts(dropna=True)
            if not counts_seniority.empty:
                top_seniority_label = counts_seniority.index[0]
                top_seniority_pct = (counts_seniority.iloc[0] / total_participants) * 100

        ruolo_col = find_column_case_insensitive(df_uploaded, "ruolo")
        num_unique_roles = None
        if ruolo_col is not None:
            # ruoli totali (anche studenti) solo per KPI numerico
            roles_series_all = df_uploaded[ruolo_col].dropna().astype(str)
            num_unique_roles = int(roles_series_all.nunique()) if not roles_series_all.empty else 0

        st.subheader("Panoramica partecipanti")

        kpi_row1 = st.columns(3)
        kpi_row1[0].metric("Totale partecipanti", total_participants)
        if unique_orgs is not None:
            kpi_row1[1].metric("Organizzazioni uniche", unique_orgs)
        if unique_sectors is not None:
            kpi_row1[2].metric("Settori produttivi unici", unique_sectors)

        kpi_row2 = st.columns(2)
        if top_seniority_label is not None:
            val = f"{top_seniority_label} ({top_seniority_pct:.1f}%)"
            kpi_row2[0].metric("Seniority più diffusa", val)
        if num_unique_roles is not None:
            kpi_row2[1].metric("Ruoli diversi dichiarati", num_unique_roles)

        kpis_for_pdf = {
            "Totale partecipanti": total_participants,
            "Organizzazioni uniche": unique_orgs if unique_orgs is not None else "n.d.",
            "Settori produttivi unici": unique_sectors if unique_sectors is not None else "n.d.",
            "Seniority più diffusa": (
                f"{top_seniority_label} ({top_seniority_pct:.1f}%)"
                if top_seniority_label is not None else "n.d."
            ),
            "Ruoli diversi dichiarati": (
                num_unique_roles if num_unique_roles is not None else "n.d."
            ),
        }

        # ----------------------------
        # Grafici per le categorie di profiling
        # ----------------------------
        st.subheader("Analisi grafica dei partecipanti")

        for idx, category in enumerate(CATEGORIES_WITH_CHARTS):
            if category not in df_uploaded.columns:
                st.warning(
                    f"La colonna '{category}' non è presente nel file caricato."
                )
                continue

            st.markdown(f"### {category}")

            value_counts = df_uploaded[category].value_counts(dropna=True)

            if value_counts.empty:
                st.info(
                    f"Nessun dato disponibile per '{category}' "
                    "dopo aver rimosso i valori mancanti."
                )
                continue

            value_counts = value_counts.sort_values(ascending=False)
            total = value_counts.sum()
            percentages = (value_counts / total * 100).round(1)

            dist_df = pd.DataFrame({
                category: value_counts.index.astype(str),
                "Numero": value_counts.values,
                "Percentuale": percentages.values,
            })

            num_bars = len(dist_df)
            colors = get_colors_for_bars(
                num_bars=num_bars,
                chart_index=idx,
                values=dist_df["Numero"].values,
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(dist_df[category], dist_df["Numero"], color=colors)

            ax.set_title(f"Distribuzione di {category}")
            ax.set_xlabel(category)
            ax.set_ylabel("Numero di partecipanti")
            plt.xticks(rotation=45, ha="right")

            for bar, pct in zip(bars, dist_df["Percentuale"]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander(f"Dettaglio valori per '{category}'"):
                st.dataframe(
                    dist_df.set_index(category),
                    use_container_width=True,
                )

        if ORGANIZATION_COLUMN in df_uploaded.columns:
            st.info(
                "La colonna "
                f"'{ORGANIZATION_COLUMN}' "
                "non viene visualizzata come grafico perché contiene "
                "troppi valori diversi. "
                "Puoi comunque analizzarla nella tabella completa qui sotto."
            )

        # ----------------------------
        # Analisi dei ruoli (escludendo occupazione = Studio)
        # ----------------------------
        st.subheader("Analisi dei ruoli professioanli dichiarati - aggregati")

        top_roles_for_pdf = None
        if ruolo_col is None:
            st.info(
                "Nessuna colonna 'ruolo' trovata nel file (ricerca case-insensitive)."
            )
        else:
            if "Occupazione" not in df_uploaded.columns:
                st.info(
                    "Non è possibile escludere gli studenti perché manca la colonna 'Occupazione'."
                )
                roles_for_analysis = df_uploaded[ruolo_col].dropna().astype(str)
            else:
                mask_not_studio = (
                    df_uploaded["Occupazione"]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    != "studio"
                )
                roles_for_analysis = (
                    df_uploaded.loc[mask_not_studio, ruolo_col]
                    .dropna()
                    .astype(str)
                )

            if roles_for_analysis.empty:
                st.info(
                    "Non ci sono ruoli disponibili (dopo aver escluso eventuali 'Studio')."
                )
            else:
                # Normalizza ruoli simili
                normalized_roles = [normalize_role(r) for r in roles_for_analysis]
                role_counts = Counter(normalized_roles)
                top_roles = role_counts.most_common(15)
                top_roles_for_pdf = top_roles.copy()

                st.markdown("**Top 15 ruoli professionali aggregati**")
                df_top_roles = pd.DataFrame(
                    top_roles, columns=["Ruolo aggregato", "Numero partecipanti"]
                )
                st.dataframe(df_top_roles, use_container_width=True)

        # ----------------------------
        # Tabella completa con filtri opzionali (una sola tabella)
        # ----------------------------
        st.subheader("Tabella completa dei partecipanti")

        df_filtered = df_uploaded.copy()

        with st.expander("Aggiungi filtri opzionali"):
            for col in FILTER_COLUMNS:
                if col not in df_uploaded.columns:
                    continue

                col_data = df_uploaded[col]
                unique_vals = sorted(
                    col_data.dropna().astype(str).unique().tolist()
                )
                if not unique_vals:
                    continue

                use_filter = st.checkbox(
                    f"Attiva filtro per '{col}'",
                    value=False,
                )

                if use_filter:
                    selected_vals = st.multiselect(
                        f"Seleziona i valori da mantenere per '{col}'",
                        options=unique_vals,
                        default=unique_vals,
                    )

                    if selected_vals and len(selected_vals) < len(unique_vals):
                        df_filtered = df_filtered[
                            df_filtered[col].astype(str).isin(selected_vals)
                        ]

        # Nascondi automaticamente le colonne completamente vuote
        cols_to_hide = []
        for col in df_filtered.columns:
            col_series = df_filtered[col]

            if col_series.dtype == "object":
                col_norm = col_series.replace(
                    ["", " ", "  ", "None", "none", "NaN", "nan"],
                    pd.NA,
                )
            else:
                col_norm = col_series

            if col_norm.isna().all():
                cols_to_hide.append(col)

        if cols_to_hide:
            df_to_show = df_filtered.drop(columns=cols_to_hide)
        else:
            df_to_show = df_filtered

        st.dataframe(
            df_to_show,
            use_container_width=True,
        )

        # ----------------------------
        # Sezione download report PDF
        # ----------------------------
        st.subheader("Scarica il report in PDF")

        top_sectors_for_pdf = None
        if sectors_present:
            counts_sectors = df_uploaded[sectors_col].value_counts(dropna=True)
            if not counts_sectors.empty:
                top5_sectors = counts_sectors.head(10)
                top_sectors_for_pdf = list(
                    zip(top5_sectors.index.tolist(), top5_sectors.values.tolist())
                )

        pdf_bytes = generate_pdf_report(
            df_uploaded,
            kpis=kpis_for_pdf,
            top_roles=top_roles_for_pdf,
            top_sectors=top_sectors_for_pdf,
        )

        st.download_button(
            label="Scarica report PDF della sessione",
            data=pdf_bytes,
            file_name="report_partecipanti_sessione.pdf",
            mime="application/pdf",
        )

    except Exception as e:
        st.error(
            f"Errore nella lettura del file: {e}. "
            "Assicurati che sia un file Excel valido."
        )
else:
    st.info("Carica un file Excel per procedere con l'analisi.")
