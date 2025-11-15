import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from fpdf import FPDF
from PIL import Image  # per il logo

# --- STILE CUSTOM PER RENDERE LA DASHBOARD PIÙ MODERNA ---
st.markdown(
    """
    <style>
        /* Sfondo più chiaro e pulito */
        .main {
            background-color: #f5f7fa;
        }

        /* Riduci un po' il padding laterale */
        div.block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Cards per le metriche */
        .stMetric {
            background: #ffffff;
            padding: 1rem 1.2rem;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        }

        /* Tabs in stile "pill" moderno */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px !important;
            background-color: #e5e7eb;
            padding: 0.35rem 0.9rem;
            font-weight: 500;
            color: #374151;
        }

        .stTabs [aria-selected="true"] {
            background-color: #73b27d !important;
            color: #ffffff !important;
        }

        /* Tabelle con angoli arrotondati */
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        }

        /* Separatori più eleganti */
        hr {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 1.5rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Configurazione pagina
# ----------------------------

PAGE_TITLE = (
    "Dashboard partecipanti alla mia sessione "
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
    st.write("5")  # un filo di spazio sopra
    st.image(
        "assets/logo_foodhub.png",
        width=90
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


def get_colors_for_bars(num_bars, chart_index, values=None):
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
    """
    r = role.strip().lower()

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

    return r.title()


def generate_pdf_report(df: pd.DataFrame,
                        kpis: dict,
                        top_roles=None,
                        top_sectors=None) -> bytes:
    """
    Crea un PDF semplice con:
    - titolo
    - KPI principali
    - top ruoli
    - top settori
    Restituisce i bytes del PDF.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Report partecipanti - Festival dell'Innovazione Agroalimentare", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "KPI principali", ln=True)
    pdf.set_font("Arial", "", 11)
    for label, value in kpis.items():
        pdf.cell(0, 7, f"- {label}: {value}", ln=True)

    if top_roles:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Top ruoli (aggregati, esclusi studenti)", ln=True)
        pdf.set_font("Arial", "", 11)
        for role, count in top_roles:
            line = f"- {role}: {count} partecipanti"
            pdf.cell(0, 7, line[:120], ln=True)

    if top_sectors:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Top settori produttivi", ln=True)
        pdf.set_font("Arial", "", 11)
        for sector, count in top_sectors:
            line = f"- {sector}: {count} partecipanti"
            pdf.cell(0, 7, line[:120], ln=True)

    pdf_raw = pdf.output(dest="S")

    if isinstance(pdf_raw, str):
        pdf_bytes = pdf_raw.encode("latin-1", "ignore")
    else:
        pdf_bytes = bytes(pdf_raw)

    return pdf_bytes


def plot_category_distribution(df, category, idx):
    """Funzione riutilizzabile per i grafici delle categorie."""
    if category not in df.columns:
        st.warning(f"La colonna '{category}' non è presente nel file caricato.")
        return

    value_counts = df[category].value_counts(dropna=True)
    if value_counts.empty:
        st.info(
            f"Nessun dato disponibile per '{category}' "
            "dopo aver rimosso i valori mancanti."
        )
        return

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
    # Rimuovi bordi inutili
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Griglia leggera solo sull'asse Y
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

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


def build_executive_summary(
    df,
    kpis_for_pdf,
    roles_for_analysis_norm,
    sectors_col,
    occup_col,
    org_type_col,
    seniority_col,
    perc_over_10=None,
):
    """Genera un riassunto testuale 'executive summary' basato sui dati,
    focalizzato su profili professionali di medio-alto livello.
    """
    total = len(df)
    if total == 0:
        return "Non ci sono partecipanti registrati per questa sessione."

    lines = []

    # Partecipanti totali
    lines.append(
        f"La sessione ha coinvolto **{total} partecipanti**."
    )

    # Tipologia organizzazione
    if org_type_col in df.columns:
        vc_org = df[org_type_col].value_counts(dropna=True)
        if not vc_org.empty:
            main_org = vc_org.index[0]
            pct_org = vc_org.iloc[0] / total * 100
            lines.append(
                f"La tipologia di organizzazione più rappresentata è **{main_org}** "
                f"({pct_org:.1f}% dei partecipanti)."
            )

    # Settore produttivo
    if sectors_col in df.columns:
        vc_sec = df[sectors_col].value_counts(dropna=True)
        if not vc_sec.empty:
            main_sec = vc_sec.index[0]
            pct_sec = vc_sec.iloc[0] / total * 100
            lines.append(
                f"Il settore produttivo prevalente è **{main_sec}**, "
                f"che pesa per circa il **{pct_sec:.1f}%** del totale."
            )

    # Seniority: mettiamo in luce i profili con >10 anni
    # Se non ci hanno passato perc_over_10, lo ricalcoliamo qui
    if perc_over_10 is None and seniority_col in df.columns:
        seniority_series = (
            df[seniority_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        if seniority_series.shape[0] > 0:
            senior_mask = seniority_series.isin(["11-20", ">20"])
            perc_over_10 = (senior_mask.sum() / seniority_series.shape[0]) * 100

    if perc_over_10 is not None:
        if perc_over_10 >= 40:
            lines.append(
                f"Il pubblico è fortemente **senior**, con circa il **{perc_over_10:.1f}%** "
                "dei partecipanti con oltre 10 anni di esperienza."
            )
        elif perc_over_10 >= 20:
            lines.append(
                f"È presente una componente rilevante di profili **mid-senior**, "
                f"con circa il **{perc_over_10:.1f}%** dei partecipanti oltre i 10 anni di esperienza."
            )
        elif perc_over_10 > 0:
            lines.append(
                f"La quota di profili con oltre 10 anni di esperienza è pari a circa **{perc_over_10:.1f}%**."
            )

    # Ruoli aggregati (R&D, Qualità, ecc.)
    if roles_for_analysis_norm:
        role_counts = Counter(roles_for_analysis_norm)
        top_role, top_role_count = role_counts.most_common(1)[0]
        pct_top_role = top_role_count / total * 100
        lines.append(
            f"A livello di profili professionali, il ruolo aggregato più rappresentato è "
            f"**{top_role}** (circa {pct_top_role:.1f}% dei partecipanti)."
        )

    # Nota: volutamente NON parliamo di studenti o profili poco strategici
    return "\n\n".join(lines)


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
        # KPI principali (calcolati una volta)
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
        if seniority_col in df_uploaded.columns and total_participants > 0:
            counts_seniority = df_uploaded[seniority_col].value_counts(dropna=True)
            if not counts_seniority.empty:
                top_seniority_label = counts_seniority.index[0]
                top_seniority_pct = (counts_seniority.iloc[0] / total_participants) * 100

        ruolo_col = find_column_case_insensitive(df_uploaded, "ruolo")
        num_unique_roles = None
        if ruolo_col is not None:
            roles_series_all = df_uploaded[ruolo_col].dropna().astype(str)
            num_unique_roles = int(roles_series_all.nunique()) if not roles_series_all.empty else 0

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

        # Pre-elaborazione ruoli per executive summary e tab Ruoli
        occup_col = "Occupazione"
        org_type_col = "Tipologia di organizzazione presso cui lavori"

        roles_for_analysis_norm = []
        top_roles_for_pdf = None

        if ruolo_col is not None:
            if occup_col in df_uploaded.columns:
                mask_not_studio = (
                    df_uploaded[occup_col]
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
            else:
                roles_for_analysis = df_uploaded[ruolo_col].dropna().astype(str)

            if not roles_for_analysis.empty:
                roles_for_analysis_norm = [normalize_role(r) for r in roles_for_analysis]
                role_counts = Counter(roles_for_analysis_norm)
                top_roles_for_pdf = role_counts.most_common(15)

            # ----------------------------
        # TABS
        # ----------------------------
        tab_overview, tab_grafici, tab_ruoli, tab_tabella = st.tabs(
            ["Overview", "Grafici", "Ruoli", "Tabella dettagliata"]
        )

        # ----------------------------
        # TAB 1: OVERVIEW
        # ----------------------------
        with tab_overview:
            # Pill introduttiva
            st.markdown(
                """
                <div style="
                    display:inline-block;
                    padding:4px 10px;
                    border-radius:999px;
                    background-color:#e3f2eb;
                    color:#14532d;
                    font-size:12px;
                    font-weight:600;
                    margin-bottom:8px;
                ">
                    Festival dell’Innovazione Agroalimentare · Audience Insights
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Panoramica partecipanti")
            st.markdown(
    """
    <div style="
        background-color:#ffffff;
        padding:16px 20px;
        border-radius:12px;
        box-shadow:0 4px 12px rgba(15,23,42,0.05);
        border-left:6px solid #73b27d;
        margin-bottom:20px;
    ">
        <p style="font-size:15px; color:#333; line-height:1.5; margin:0;">
            Questa dashboard offre una sintesi professionale dei partecipanti alla tua sessione,
            con l’obiettivo di evidenziare la composizione del pubblico e il suo potenziale strategico.
            È possibile esplorare sia le distribuzioni generali (occupazione, seniority, settore,
            tipo di organizzazione) sia analisi più mirate come la presenza di profili con 
            oltre 10 anni di esperienza o i ruoli maggiormente rappresentati. <br><br>
            La tabella finale consente un’analisi completa e filtrabile di ogni singolo partecipante,
            mentre i grafici e l’executive summary supportano una lettura immediata, utile per 
            stakeholder, sponsor e partner interessati a comprendere il valore dell’audience raggiunta.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

            # KPI riga 1
            kpi_row1 = st.columns(3)
            kpi_row1[0].metric("Totale partecipanti", total_participants)
            if unique_orgs is not None:
                kpi_row1[1].metric("Organizzazioni uniche", unique_orgs)
            if unique_sectors is not None:
                kpi_row1[2].metric("Settori produttivi unici", unique_sectors)

            # --- KPI Seniority: percentuale con più di 10 anni (11-20 e >20) ---
            seniority_col = "Seniority"
            perc_over_10 = None
            if seniority_col in df_uploaded.columns:
                seniority_series = (
                    df_uploaded[seniority_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                if seniority_series.shape[0] > 0:
                    senior_mask = seniority_series.isin(["11-20", ">20"])
                    perc_over_10 = (senior_mask.sum() / seniority_series.shape[0]) * 100

            # KPI riga 2
            kpi_row2 = st.columns(2)
            if perc_over_10 is not None:
                kpi_row2[0].metric(
                    "Senior (>10 anni di esperienza)",
                    f"{perc_over_10:.1f}%",
                    help="Percentuale dei partecipanti con più di 10 anni di esperienza (categorie 11-20 e >20)"
                )
            if num_unique_roles is not None:
                kpi_row2[1].metric("Ruoli diversi dichiarati", num_unique_roles)

            # Executive summary
            summary_text = build_executive_summary(
                df_uploaded,
                kpis_for_pdf,
                roles_for_analysis_norm,
                sectors_col,
                occup_col,
                org_type_col,
                seniority_col,
                perc_over_10=perc_over_10,
            )
            st.markdown(summary_text)

            st.markdown("---")
            st.subheader("Distribuzioni chiave")

            # Alcuni grafici chiave: Occupazione, Settore produttivo, Seniority (se esistono)
            key_categories = []
            if "Occupazione" in df_uploaded.columns:
                key_categories.append("Occupazione")
            if sectors_col in df_uploaded.columns:
                key_categories.append(sectors_col)
            if seniority_col in df_uploaded.columns:
                key_categories.append(seniority_col)

            for idx, cat in enumerate(key_categories):
                st.markdown(f"### {cat}")
                plot_category_distribution(df_uploaded, cat, idx)

        # ----------------------------
        # TAB 2: GRAFICI (tutti)
        # ----------------------------
        with tab_grafici:
            st.subheader("Analisi grafica completa")

            for idx, category in enumerate(CATEGORIES_WITH_CHARTS):
                st.markdown(f"### {category}")
                plot_category_distribution(df_uploaded, category, idx)

        # ----------------------------
        # TAB 3: RUOLI
        # ----------------------------
        with tab_ruoli:
            st.subheader("Analisi dei ruoli professionali dichiarati")

            if ruolo_col is None:
                st.info(
                    "Nessuna colonna 'ruolo' trovata nel file (ricerca case-insensitive)."
                )
            else:
                if occup_col not in df_uploaded.columns:
                    st.info(
                        "Non è possibile escludere gli studenti perché manca la colonna 'Occupazione'. "
                        "I ruoli mostrati includono tutti i partecipanti."
                    )

                if not roles_for_analysis_norm:
                    st.info(
                        "Non ci sono ruoli disponibili (dopo aver escluso eventuali 'Studio')."
                    )
                else:
                    role_counts = Counter(roles_for_analysis_norm)
                    top_roles = role_counts.most_common(15)

                    st.markdown("**Top 15 ruoli aggregati**")
                    df_top_roles = pd.DataFrame(
                        top_roles, columns=["Ruolo aggregato", "Numero partecipanti"]
                    )
                    st.dataframe(df_top_roles, use_container_width=True)

                    st.markdown("**Mappa visiva dei ruoli (dimensione ∝ frequenza)**")
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
                    ax_wc.set_title(
                        "Ruoli dichiarati aggregati",
                        fontsize=14
                    )

                    max_count = max(role_counts.values())
                    for role, count in role_counts.most_common(30):
                        x, y = np.random.rand(), np.random.rand()
                        fontsize = 8 + (count / max_count) * 20
                        ax_wc.text(
                            x,
                            y,
                            role,
                            fontsize=fontsize,
                            alpha=0.7,
                            color="#73b27d",
                            transform=ax_wc.transAxes,
                        )

                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)

        # ----------------------------
        # TAB 4: TABELLA DETTAGLIATA + EXPORT CSV
        # ----------------------------
        with tab_tabella:
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

            # Nascondi colonne completamente vuote
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

            # Export CSV dei dati filtrati
            st.markdown("### Esporta dati filtrati")
            csv_data = df_to_show.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Scarica dati filtrati (CSV)",
                data=csv_data,
                file_name="partecipanti_filtrati.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(
            f"Errore durante l'elaborazione del file: {e}. "
            "Assicurati che sia un file Excel valido e strutturato correttamente."
        )
else:
    st.info("Carica un file Excel (.xlsx o .xls) per procedere con l'analisi.")
