import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

st.title(PAGE_TITLE)
st.write(
    "Carica un file Excel con i dati dei partecipanti per "
    "visualizzare grafici e una tabella filtrabile."
)

# ----------------------------
# Impostazioni analisi
# ----------------------------

# Tutte le categorie di profiling presenti nel questionario
ALL_PROFILING_CATEGORIES = [
    "Occupazione",
    "Tipologia di organizzazione presso cui lavori",
    "Organizzazione presso cui lavori o studi",
    "Seniority",
    "Area aziendale",
    "Settore produttivo",
]

# Colonna che NON deve avere grafico
ORGANIZATION_COLUMN = "Organizzazione presso cui lavori o studi"

# Categorie per cui vogliamo disegnare un grafico
CATEGORIES_WITH_CHARTS = [
    c for c in ALL_PROFILING_CATEGORIES if c != ORGANIZATION_COLUMN
]

# Colonne per cui attivare filtri nella tabella finale
FILTER_COLUMNS = [
    "Occupazione",
    "Tipologia di organizzazione presso cui lavori",
    "Seniority",
    "Area aziendale",
    "Settore produttivo",
]

# Palette colori del brand
BRAND_COLORS = ["#73b27d", "#f1ad72", "#d31048"]


def get_colors_for_bars(num_bars: int, chart_index: int, values=None):
    """
    Restituisce una lista di colori per le barre:
    - se num_bars <= 3: usa i 3 colori brand alternati
    - se num_bars > 3: usa un solo colore brand con gradiente di alpha
      (più trasparente per i valori più piccoli).
    chart_index serve per scegliere il colore base a rotazione tra i 3.
    """
    if num_bars <= 3:
        return [BRAND_COLORS[i % len(BRAND_COLORS)] for i in range(num_bars)]

    # Scegli un colore base a rotazione
    base_hex = BRAND_COLORS[chart_index % len(BRAND_COLORS)]
    base_rgba = mcolors.to_rgba(base_hex)

    # Calcolo alpha in base ai valori (più grande = meno trasparente)
    if values is None or len(values) != num_bars:
        alphas = [1.0 for _ in range(num_bars)]
    else:
        vals = pd.Series(values, dtype=float)
        v_min = vals.min()
        v_max = vals.max()
        if v_max == v_min:
            alphas = [1.0 for _ in range(num_bars)]
        else:
            norm = (vals - v_min) / (v_max - v_min)
            alphas = 0.3 + 0.7 * norm  # tra 0.3 e 1.0

    colors = [
        (base_rgba[0], base_rgba[1], base_rgba[2], float(a))
        for a in alphas
    ]
    return colors


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

            # Conteggi e percentuali (ignoriamo i NaN)
            value_counts = df_uploaded[category].value_counts(dropna=True)

            if value_counts.empty:
                st.info(
                    f"Nessun dato disponibile per '{category}' "
                    "dopo aver rimosso i valori mancanti."
                )
                continue

            # Ordiniamo per numero decrescente per dare senso al gradiente
            value_counts = value_counts.sort_values(ascending=False)

            total = value_counts.sum()
            percentages = (value_counts / total * 100).round(1)

            # DataFrame di distribuzione
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

            # Grafico a barre
            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.bar(dist_df[category], dist_df["Numero"], color=colors)

            ax.set_title(f"Distribuzione di {category}")
            ax.set_xlabel(category)
            ax.set_ylabel("Numero di partecipanti")
            plt.xticks(rotation=45, ha="right")

            # Etichette con % sopra ogni barra
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

            # Tabellina riassuntiva per categoria
            with st.expander(f"Dettaglio valori per '{category}'"):
                st.dataframe(
                    dist_df.set_index(category),
                    use_container_width=True,
                )

        # Nota sulla colonna Organizzazione
        if ORGANIZATION_COLUMN in df_uploaded.columns:
            st.info(
                "La colonna "
                f"'{ORGANIZATION_COLUMN}' "
                "non viene visualizzata come grafico perché contiene "
                "troppi valori diversi. "
                "Puoi comunque analizzarla nella tabella completa qui sotto."
            )

        # ----------------------------
        # Tabella finale con filtri opzionali (una sola tabella)
        # ----------------------------
        st.subheader("Tabella completa dei partecipanti")

        # Di base: nessun filtro → tutte le righe
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

                # Checkbox per attivare il filtro
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

                    # Applico il filtro solo se l'utente ha effettivamente scelto qualcosa
                    if selected_vals and len(selected_vals) < len(unique_vals):
                        df_filtered = df_filtered[
                            df_filtered[col].astype(str).isin(selected_vals)
                        ]

        # Nascondi automaticamente le colonne di FILTER_COLUMNS che,
        # dopo l'applicazione dei filtri, sono completamente vuote (tutti NaN/None)
        cols_to_hide = []
        for col in FILTER_COLUMNS:
            if col in df_filtered.columns:
                # Se tutti i valori sono NaN/None → da nascondere
                if df_filtered[col].notna().sum() == 0:
                    cols_to_hide.append(col)

        if cols_to_hide:
            df_to_show = df_filtered.drop(columns=cols_to_hide)
        else:
            df_to_show = df_filtered

        st.dataframe(
            df_to_show,
            use_container_width=True,
        )

    except Exception as e:
        st.error(
            f"Errore nella lettura del file: {e}. "
            "Assicurati che sia un file Excel valido."
        )
else:
    st.info("Carica un file Excel per procedere con l'analisi.")
