# -*- coding: utf-8 -*-
"""
PFE - Biofilm Prediction System
Analyse microbiologique + Machine Learning + Interface decisionnelle
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Biofilm PFE", layout="wide")

st.title("Analyse et Prediction des Biofilms Pathogenes")
st.markdown("### Systeme d'aide a la decision pour environnements alimentaires")

# =========================
# UPLOAD DATA
# =========================
uploaded_file = st.file_uploader("Charger fichier CSV", type=["csv"])

# =========================
# CLEAN FUNCTION
# =========================
def clean_data(df):

    df = df.copy()
    df = df.replace("NA", np.nan)

    # numeric conversion
    for col in ["OD", "growth", "blank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fix year extraction (BUG FIX IMPORTANT)
    if "plateID" in df.columns:
        df["year"] = df["year"].fillna(
            df["plateID"].astype(str).str.extract(r"(\d{4})")[0].astype(float)
        )

    # OD correction
    df["OD_corrected"] = df["OD"] - df["blank"]
    df["OD_corrected"] = df["OD_corrected"].fillna(0)

    # type mapping
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower()

    return df


# =========================
# LOAD
# =========================
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df = clean_data(df)

    st.success("Donnees chargees")

    # =========================
    # DATA VIEW
    # =========================
    st.subheader("Apercu donnees")
    st.dataframe(df.head())

    # =========================
    # STATISTICS
    # =========================
    st.subheader("Analyse statistique")

    col1, col2, col3 = st.columns(3)

    col1.metric("Nb echantillons", len(df))
    col2.metric("OD moyenne", round(df["OD_corrected"].mean(), 3))
    col3.metric("Growth moyenne", round(df["growth"].mean(), 3))

    # =========================
    # VISUALIZATION 1
    # =========================
    st.subheader("Distribution Biofilm")

    fig1 = px.histogram(df, x="OD_corrected", color="type")
    st.plotly_chart(fig1, use_container_width=True, key="hist_biofilm")

    # =========================
    # RELATION GROWTH vs BIOFILM
    # =========================
    st.subheader("Relation Growth vs Biofilm")

    fig2 = px.scatter(
        df,
        x="growth",
        y="OD_corrected",
        color="type",
        trendline="ols"
    )
    st.plotly_chart(fig2, use_container_width=True, key="scatter_growth")

    # =========================
    # HEATMAP CORRELATION (Entre growth, OD, type, OD_corrected)
    # =========================
    st.subheader("Matrice de correlation - Variables microbiologiques")
    
    # Preparer les donnees pour la correlation
    df_corr = df.copy()
    
    # Convertir 'type' en variable numerique pour la correlation
    if "type" in df_corr.columns:
        le_type = LabelEncoder()
        df_corr["type_encoded"] = le_type.fit_transform(df_corr["type"].astype(str))
    
    # Selectionner les colonnes pour la matrice de correlation
    columns_for_corr = []
    
    if "growth" in df_corr.columns:
        columns_for_corr.append("growth")
    if "OD" in df_corr.columns:
        columns_for_corr.append("OD")
    if "OD_corrected" in df_corr.columns:
        columns_for_corr.append("OD_corrected")
    if "type_encoded" in df_corr.columns:
        columns_for_corr.append("type_encoded")
    
    if len(columns_for_corr) >= 2:
        # Calculer la matrice de correlation
        corr_matrix = df_corr[columns_for_corr].corr()
        
        # Renommer les colonnes pour l'affichage
        corr_matrix_renamed = corr_matrix.copy()
        corr_matrix_renamed.index = [idx.replace("type_encoded", "type") for idx in corr_matrix_renamed.index]
        corr_matrix_renamed.columns = [col.replace("type_encoded", "type") for col in corr_matrix_renamed.columns]
        
        # Afficher la matrice de correlation
        fig3 = px.imshow(
            corr_matrix_renamed,
            text_auto=True,
            color_continuous_scale="RdBu",
            title="Matrice de correlation: Growth, OD, OD_corrected, Type",
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig3, use_container_width=True, key="corr_heatmap")
        
        # Afficher les correlations fortes
        st.markdown("**Correlations significatives:**")
        
        # Afficher toutes les correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                var1 = corr_matrix.columns[i].replace("type_encoded", "type")
                var2 = corr_matrix.columns[j].replace("type_encoded", "type")
                st.write(f"- {var1} <-> {var2}: {corr_value:.3f}")
                
                # Ajouter une interpretation
                if abs(corr_value) > 0.7:
                    st.write(f"  *Correlation tres forte*")
                elif abs(corr_value) > 0.5:
                    st.write(f"  *Correlation moderee*")
                elif abs(corr_value) > 0.3:
                    st.write(f"  *Correlation faible*")
        
        # Graphique supplementaire: Relation entre les variables
        st.subheader("Visualisation des relations")
        
        # Creer un graphique de dispersion interactif
        fig_scatter_matrix = px.scatter_matrix(
            df_corr[columns_for_corr],
            dimensions=columns_for_corr,
            title="Matrice de dispersion: Relations entre les variables",
            labels={"growth": "Growth", "OD": "OD", "OD_corrected": "OD corrigee", "type_encoded": "Type"}
        )
        st.plotly_chart(fig_scatter_matrix, use_container_width=True, key="scatter_matrix")
        
    else:
        st.warning("Pas assez de variables pour l'analyse de correlation (besoin d'au moins 2 variables)")
    # =========================
    # ANALYSE DE CORRELATION: GROWTH, STRAIN ET TYPE
    # =========================
    st.subheader("🔬 Analyse de corrélation: Growth, Strain et Type de biofilm")
    
    # Preparer les donnees pour l'analyse Growth-Strain-Type
    df_corr_gst = df.copy()
    
    # Variables a analyser
    gst_variables = []
    
    # 1. Growth (déjà numerique)
    if "growth" in df_corr_gst.columns:
        gst_variables.append("growth")
        st.write("✅ Growth: variable numerique")
    
    # 2. Type (encoder en numerique)
    if "type" in df_corr_gst.columns:
        le_type_gst = LabelEncoder()
        df_corr_gst["type_encoded_gst"] = le_type_gst.fit_transform(df_corr_gst["type"].astype(str))
        gst_variables.append("type_encoded_gst")
        st.write(f"✅ Type de biofilm: {dict(zip(le_type_gst.classes_, range(len(le_type_gst.classes_))))}")
    
    # 3. Strain (encoder en numerique)
    if "strain" in df_corr_gst.columns:
        le_strain_gst = LabelEncoder()
        df_corr_gst["strain_encoded_gst"] = le_strain_gst.fit_transform(df_corr_gst["strain"].astype(str))
        gst_variables.append("strain_encoded_gst")
        st.write(f"✅ Strain: {len(le_strain_gst.classes_)} souches differentes")
        
        # Afficher le mapping des souches
        strain_mapping = dict(zip(le_strain_gst.classes_, range(len(le_strain_gst.classes_))))
        st.write("**Mapping des souches:**")
        strain_df = pd.DataFrame(list(strain_mapping.items()), columns=["Souche", "Code"])
        st.dataframe(strain_df, use_container_width=True)
    
    if len(gst_variables) >= 2:
        # Calculer la matrice de correlation
        corr_gst_matrix = df_corr_gst[gst_variables].corr()
        
        # Renommer les colonnes pour l'affichage
        rename_map = {
            "growth": "Growth",
            "type_encoded_gst": "Type Biofilm",
            "strain_encoded_gst": "Strain"
        }
        corr_gst_renamed = corr_gst_matrix.rename(index=rename_map, columns=rename_map)
        
        # Afficher la matrice de correlation
        fig_gst = px.imshow(
            corr_gst_renamed,
            text_auto=True,
            color_continuous_scale="RdBu",
            title="Matrice de correlation: Growth - Strain - Type",
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_gst, use_container_width=True, key="corr_gst_heatmap")
        
        # Afficher les correlations specifiques
        st.markdown("**📊 Correlations specifiques:**")
        
        if "growth" in gst_variables and "type_encoded_gst" in gst_variables:
            corr_growth_type = corr_gst_matrix.loc["growth", "type_encoded_gst"]
            st.write(f"- **Growth ↔ Type**: {corr_growth_type:.3f}")
            if corr_growth_type > 0.5:
                st.write("  *Plus la croissance est elevee, plus le biofilm est fort*")
            elif corr_growth_type < -0.5:
                st.write("  *Plus la croissance est elevee, plus le biofilm est faible*")
            else:
                st.write("  *Correlation faible entre croissance et type de biofilm*")
        
        if "growth" in gst_variables and "strain_encoded_gst" in gst_variables:
            corr_growth_strain = corr_gst_matrix.loc["growth", "strain_encoded_gst"]
            st.write(f"- **Growth ↔ Strain**: {corr_growth_strain:.3f}")
            if abs(corr_growth_strain) > 0.5:
                st.write("  *Certaines souches sont associees a une croissance plus elevee*")
            else:
                st.write("  *La croissance ne depend pas fortement de la souche*")
        
        if "type_encoded_gst" in gst_variables and "strain_encoded_gst" in gst_variables:
            corr_type_strain = corr_gst_matrix.loc["type_encoded_gst", "strain_encoded_gst"]
            st.write(f"- **Type ↔ Strain**: {corr_type_strain:.3f}")
            if abs(corr_type_strain) > 0.5:
                st.write("  *Certaines souches sont fortement associees a des types specifiques de biofilm*")
            elif abs(corr_type_strain) > 0.3:
                st.write("  *Relation moderee entre souche et type de biofilm*")
            else:
                st.write("  *Le type de biofilm varie independamment de la souche*")
        
        # Graphique de dispersion Growth vs Type (couleur par Strain)
        st.subheader("📈 Visualisation: Growth vs Type (couleur par Strain)")
        
        if "growth" in df.columns and "type" in df.columns and "strain" in df.columns:
            fig_scatter_strain = px.scatter(
                df,
                x="growth",
                y="type",
                color="strain",
                title="Relation Growth - Type de biofilm par Souche",
                labels={"growth": "Croissance (Growth)", "type": "Type de biofilm"},
                hover_data=["OD_corrected"]
            )
            st.plotly_chart(fig_scatter_strain, use_container_width=True, key="scatter_strain_type")
        
        # Boxplot Growth par Type et Strain
        st.subheader("📊 Distribution de Growth par Type et Strain")
        
        if "growth" in df.columns and "type" in df.columns and "strain" in df.columns:
            fig_box = px.box(
                df,
                x="type",
                y="growth",
                color="strain",
                title="Distribution de la croissance selon le type de biofilm et la souche",
                labels={"type": "Type de biofilm", "growth": "Croissance (Growth)"}
            )
            st.plotly_chart(fig_box, use_container_width=True, key="box_growth_type_strain")
        
        # Heatmap de la frequence Type-Strain
        st.subheader("📊 Matrice de frequence: Type de biofilm vs Strain")
        
        if "type" in df.columns and "strain" in df.columns:
            # Creer une table de contingence
            contingency_table = pd.crosstab(df["type"], df["strain"], normalize="columns")
            
            fig_heatmap_freq = px.imshow(
                contingency_table,
                text_auto=True,
                color_continuous_scale="Viridis",
                title="Frequence des types de biofilm par souche",
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap_freq, use_container_width=True, key="freq_heatmap")
            
            # Afficher les souches les plus productrices de biofilm
            st.markdown("**🔍 Analyse par souche:**")
            
            # Compter les occurrences par type et strain
            type_strain_counts = df.groupby(["strain", "type"]).size().unstack(fill_value=0)
            st.dataframe(type_strain_counts, use_container_width=True)
            
            # Souche la plus frequente pour chaque type
            st.markdown("**Souche predominante par type de biofilm:**")
            for biofilm_type in df["type"].unique():
                dominant_strain = df[df["type"] == biofilm_type]["strain"].mode()
                if len(dominant_strain) > 0:
                    st.write(f"- Type '{biofilm_type}': Souche {dominant_strain.values[0]}")
        
        # Analyse statistique ANOVA
        st.subheader("📊 Analyse statistique (ANOVA)")
        
        if "growth" in df.columns and "strain" in df.columns:
            from scipy import stats
            
            # Grouper par strain
            strains = df["strain"].unique()
            growth_by_strain = [df[df["strain"] == s]["growth"].dropna().values for s in strains if len(df[df["strain"] == s]["growth"].dropna()) > 0]
            
            if len(growth_by_strain) >= 2:
                # ANOVA test
                f_stat, p_value = stats.f_oneway(*growth_by_strain)
                
                st.write(f"**ANOVA - Comparaison de Growth entre les souches:**")
                st.write(f"- F-statistic: {f_stat:.3f}")
                st.write(f"- P-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ Difference significative entre les souches (p < 0.05)")
                    st.write("  *Les souches ont des niveaux de croissance significativement differents*")
                else:
                    st.warning("❌ Pas de difference significative entre les souches (p > 0.05)")
                    st.write("  *Les souches ont des niveaux de croissance similaires*")
        
        # Analyse de correlation pour chaque type de biofilm
        st.subheader("📈 Correlation Growth-Strain par type de biofilm")
        
        if "growth" in df.columns and "strain" in df.columns and "type" in df.columns:
            for biofilm_type in df["type"].unique():
                df_type = df[df["type"] == biofilm_type]
                if len(df_type) > 1 and len(df_type["strain"].unique()) > 1:
                    # Encoder strain pour ce sous-groupe
                    le_type_specific = LabelEncoder()
                    strain_encoded = le_type_specific.fit_transform(df_type["strain"].astype(str))
                    
                    # Calculer correlation
                    if len(strain_encoded) > 1:
                        corr_value = np.corrcoef(df_type["growth"], strain_encoded)[0, 1]
                        st.write(f"- **Type '{biofilm_type}'**: Correlation Growth-Strain = {corr_value:.3f}")
                        
                        if abs(corr_value) > 0.5:
                            st.write(f"  *Pour le biofilm de type {biofilm_type}, la souche influence fortement la croissance*")
                        elif abs(corr_value) > 0.3:
                            st.write(f"  *Pour le biofilm de type {biofilm_type}, la souche influence moderement la croissance*")
        
    else:
        st.warning("Variables necessaires (growth, type, strain) non disponibles pour l'analyse de correlation")
    # =========================
    # ML MODEL
    # =========================
    st.subheader("Modelisation predictive")

    # Preparer les features pour le modele
    # Inclure medium et strain si disponibles
    features = ["OD_corrected", "growth"]
    
    # Ajouter medium et strain comme features categorielles encodees
    df_ml = df.dropna(subset=["type"]).copy()
    
    # Encoder les variables categorielles si elles existent
    le_medium = None
    le_strain = None
    
    if "medium" in df_ml.columns:
        le_medium = LabelEncoder()
        df_ml["medium_encoded"] = le_medium.fit_transform(df_ml["medium"].astype(str))
        features.append("medium_encoded")
        st.info("Variable 'medium' integree au modele")
    
    if "strain" in df_ml.columns:
        le_strain = LabelEncoder()
        df_ml["strain_encoded"] = le_strain.fit_transform(df_ml["strain"].astype(str))
        features.append("strain_encoded")
        st.info("Variable 'strain' integree au modele")
    
    # S'assurer qu'il n'y a pas de valeurs manquantes
    X = df_ml[features].fillna(0)

    le = LabelEncoder()
    y = le.fit_transform(df_ml["type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    st.success(f"Accuracy du modele: {acc:.2f}")

    st.text("Classification Report")
    st.text(classification_report(y_test, preds))

    st.write("Matrice de Confusion")
    st.write(confusion_matrix(y_test, preds))
    
    # Importance des features
    st.subheader("Importance des variables")
    importance_df = pd.DataFrame({
        'Variable': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(importance_df, x='Variable', y='Importance', 
                            title="Importance des variables dans la prediction")
    st.plotly_chart(fig_importance, use_container_width=True, key="importance_bar")

    # =========================
    # INTERACTIVE PREDICTION (AVEC medium, strain, growth, OD, blank)
    # =========================
    st.subheader("Prediction laboratoire virtuel")
    
    st.markdown("**Saisissez les parametres microbiologiques:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Saisie des parametres
        growth = st.number_input("Growth (Croissance)", 0.0, 5.0, 0.5, step=0.1, key="growth_input")
        od = st.number_input("OD (Densite Optique)", 0.0, 3.0, 0.2, step=0.05, key="od_input")
        blank = st.number_input("Blank (Blanc)", 0.0, 3.0, 0.0, step=0.05, key="blank_input")
        
    with col2:
        # Saisie des variables categorielles si disponibles dans les donnees
        if le_medium is not None:
            medium_options = le_medium.classes_.tolist()
            medium = st.selectbox("Medium", medium_options, key="medium_select")
        else:
            medium = None
            st.info("La variable 'medium' n'est pas disponible dans vos donnees")
            
        if le_strain is not None:
            strain_options = le_strain.classes_.tolist()
            strain = st.selectbox("Strain (Souche)", strain_options, key="strain_select")
        else:
            strain = None
            st.info("La variable 'strain' n'est pas disponible dans vos donnees")
        
        year = st.number_input("Annee", 2000, 2030, 2024, step=1, key="year_input")

    if st.button("Predire le biofilm", type="primary", key="predict_button"):
        
        # Construire l'input pour la prediction
        input_dict = {
            "OD_corrected": od - blank,
            "growth": growth
        }
        
        # Ajouter les variables encodees si disponibles
        if le_medium is not None and medium is not None:
            input_dict["medium_encoded"] = le_medium.transform([medium])[0]
            
        if le_strain is not None and strain is not None:
            input_dict["strain_encoded"] = le_strain.transform([strain])[0]
        
        input_data = pd.DataFrame([input_dict])
        
        # Verifier que toutes les colonnes sont presentes
        for col in features:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reordonner les colonnes comme dans l'entraînement
        input_data = input_data[features]
        
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        result_label = le.inverse_transform([prediction])[0]
        st.success(f"Resultat de la prediction: {result_label}")

        st.subheader("Probabilites par classe:")
        
        # Afficher les probabilites avec une barre de progression
        for i, p in enumerate(proba):
            classe = le.inverse_transform([i])[0]
            st.write(f"- {classe}: {p:.2%}")
            st.progress(p)

        # Interpretation automatique detaillee
        st.subheader("Interpretation automatique")
        
        st.markdown("**Analyse des parametres saisis:**")
        st.write(f"- Croissance (Growth): {growth}")
        st.write(f"- Densite Optique (OD): {od}")
        st.write(f"- Blanc (Blank): {blank}")
        st.write(f"- OD corrigee: {od - blank:.3f}")
        
        if medium:
            st.write(f"- Medium: {medium}")
        if strain:
            st.write(f"- Souche: {strain}")
            
        st.markdown("**Prediction finale:**")
        st.write(f"Le modele estime que la formation de biofilm est de type **{result_label}**")
        
        st.markdown("**Niveau de confiance:**")
        confiance = max(proba)*100
        st.write(f"{confiance:.1f}%")
        
        st.markdown("**Recommandation:**")
        if max(proba) > 0.8:
            st.success("Prediction tres fiable - Confiance elevee")
        elif max(proba) > 0.6:
            st.warning("Prediction moderement fiable - Validation supplementaire recommandee")
        else:
            st.error("Prediction incertaine - Analyse complementaire necessaire")

    # =========================
    # DECISION SUPPORT
    # =========================
    st.subheader("Aide a la decision biologique")

    if st.checkbox("Activer analyse de risque", key="risk_checkbox"):
        
        # Definir les seuils de risque bases sur l'OD
        df["risk"] = df["OD_corrected"].apply(
            lambda x: "Risque Eleve" if x > 1 else ("Risque Modere" if x > 0.5 else "Risque Faible")
        )

        fig4 = px.histogram(df, x="risk", color="risk", 
                           title="Distribution des niveaux de risque",
                           color_discrete_map={
                               "Risque Eleve": "red",
                               "Risque Modere": "orange", 
                               "Risque Faible": "green"
                           })
        st.plotly_chart(fig4, use_container_width=True, key="risk_hist")
        
        st.write("**Comptage des risques:**")
        st.write(df["risk"].value_counts())
        
        # Recommandations basees sur le risque
        st.markdown("**Recommandations:**")
        if df["OD_corrected"].mean() > 1:
            st.error("Action immediate requise - Nettoyage approfondi et verification des protocoles")
        elif df["OD_corrected"].mean() > 0.5:
            st.warning("Surveillance renforcee - Controles reguliers recommandés")
        else:
            st.success("Situation maitrisee - Maintenir les bonnes pratiques")

       # =========================
    # SOLUTION BAYESIENNE
    # =========================
    st.markdown("---")
    st.subheader("Solution Bayesienne - Analyse Probabiliste des Biofilms")

    from sklearn.naive_bayes import GaussianNB

    # =========================
    # INPUT UTILISATEUR BAYESIEN
    # =========================
    st.markdown("**Saisissez les parametres pour l'analyse bayesienne:**")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        growth_b = st.number_input("Growth (Bayes)", 0.0, 5.0, 0.5, step=0.1, key="bayes_growth")
        od_b = st.number_input("OD (Bayes)", 0.0, 3.0, 0.2, step=0.05, key="bayes_od")
        
    with col_b2:
        blank_b = st.number_input("Blank (Bayes)", 0.0, 3.0, 0.0, step=0.05, key="bayes_blank")
        
        # Ajouter medium et strain pour Bayes si disponibles
        if le_medium is not None:
            medium_b = st.selectbox("Medium (Bayes)", le_medium.classes_.tolist(), key="bayes_medium")
        else:
            medium_b = None
            
        if le_strain is not None:
            strain_b = st.selectbox("Strain (Bayes)", le_strain.classes_.tolist(), key="bayes_strain")
        else:
            strain_b = None

    # =========================
    # BOUTON EXECUTION
    # =========================
    if st.button("Lancer analyse bayesienne", key="bayes_button"):

        # =========================
        # PREPARATION DATA
        # =========================
        df_ml_bayes = df.dropna(subset=["type"]).copy()
        
        # Features pour le modele bayesien
        features_bayes = ["OD_corrected", "growth"]
        
        # Ajouter medium et strain encodes si disponibles
        if "medium" in df_ml_bayes.columns and medium_b is not None:
            le_medium_bayes = LabelEncoder()
            df_ml_bayes["medium_encoded"] = le_medium_bayes.fit_transform(df_ml_bayes["medium"].astype(str))
            features_bayes.append("medium_encoded")
            
        if "strain" in df_ml_bayes.columns and strain_b is not None:
            le_strain_bayes = LabelEncoder()
            df_ml_bayes["strain_encoded"] = le_strain_bayes.fit_transform(df_ml_bayes["strain"].astype(str))
            features_bayes.append("strain_encoded")
        
        df_ml_bayes["OD_corrected"] = df_ml_bayes["OD_corrected"].fillna(0)
        df_ml_bayes["growth"] = df_ml_bayes["growth"].fillna(0)

        X_bayes = df_ml_bayes[features_bayes]
        y_bayes = df_ml_bayes["type"]

        # encode labels
        le_bayes = LabelEncoder()
        y_encoded_bayes = le_bayes.fit_transform(y_bayes)

        # =========================
        # MODEL BAYESIEN
        # =========================
        model_bayes = GaussianNB()
        model_bayes.fit(X_bayes, y_encoded_bayes)

        # =========================
        # INPUT PREDICTION
        # =========================
        input_dict_bayes = {
            "OD_corrected": od_b - blank_b,
            "growth": growth_b
        }
        
        # Ajouter les variables encodees
        if "medium_encoded" in features_bayes and medium_b is not None:
            input_dict_bayes["medium_encoded"] = le_medium_bayes.transform([medium_b])[0]
            
        if "strain_encoded" in features_bayes and strain_b is not None:
            input_dict_bayes["strain_encoded"] = le_strain_bayes.transform([strain_b])[0]
        
        input_data_bayes = pd.DataFrame([input_dict_bayes])
        
        # Reordonner les colonnes
        input_data_bayes = input_data_bayes[features_bayes]

        prediction_bayes = model_bayes.predict(input_data_bayes)[0]
        proba_bayes = model_bayes.predict_proba(input_data_bayes)[0]

        label_bayes = le_bayes.inverse_transform([prediction_bayes])[0]

        # =========================
        # RESULTAT
        # =========================
        st.success(f"Biofilm predit (Bayesien): {label_bayes}")

        st.subheader("Probabilites completes par classe")

        # Afficher les probabilites avec une barre de progression
        for i, p in enumerate(proba_bayes):
            classe = le_bayes.inverse_transform([i])[0]
            st.write(f"- {classe}: {p:.3f} ({p:.1%})")
            st.progress(p)

        # =========================
        # GRAPH PROBABILITES
        # =========================
        fig_bayes = px.bar(
            x=le_bayes.classes_,
            y=proba_bayes,
            labels={"x": "Classe de biofilm", "y": "Probabilite"},
            title="Distribution bayesienne des classes de biofilm",
            color=proba_bayes,
            color_continuous_scale="Viridis"
        )
        fig_bayes.update_traces(text=proba_bayes, textposition="outside")
        st.plotly_chart(fig_bayes, use_container_width=True, key="bayes_bar")

        # =========================
        # GRAPHE BAYESIEN SIMPLE (sans networkx pour éviter les erreurs)
        # =========================
        st.subheader("🕸️ Graphe Bayésien simplifié")
        
        # Utiliser directement des st.markdown pour afficher un graphe textuel
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h4>Réseau de dépendances bayésien</h4>
            <pre style="font-size: 16px; font-family: monospace;">
                      ┌─────────────┐
                      │   Growth    │
                      └──────┬──────┘
                             │
                      ┌──────▼──────┐
                      │             │
                      │   Biofilm   │
                      │             │
                      └──────▲──────┘
                             │
                      ┌──────┴──────┐
                      │     OD      │
                      └─────────────┘
            </pre>
            <p><b>Relations causales:</b> Growth et OD influencent la formation du biofilm</p>
        </div>
        """, unsafe_allow_html=True)
        
        # =========================
        # TABLEAU DES PROBABILITES SIMPLE
        # =========================
        st.subheader("📊 Analyse des probabilités")
        
        # Créer un tableau simple des probabilités
        prob_data = []
        for i, p in enumerate(proba_bayes):
            classe = le_bayes.inverse_transform([i])[0]
            prob_data.append({
                "Classe": classe,
                "Probabilité": f"{p:.2%}",
                "Niveau": "🟢 Élevée" if p > 0.5 else "🟡 Moyenne" if p > 0.2 else "🔴 Faible"
            })
        
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df, use_container_width=True)
        
        # =========================
        # ANALYSE DES SCENARIOS
        # =========================
        st.subheader("📈 Analyse de sensibilité")
        
        # Créer différents scénarios
        scenarios_data = []
        for growth_val in [0.2, 0.5, 0.8]:
            for od_val in [0.2, 0.5, 0.8]:
                test_input = pd.DataFrame([{
                    "OD_corrected": od_val,
                    "growth": growth_val
                }])
                
                if "medium_encoded" in features_bayes and medium_b is not None:
                    test_input["medium_encoded"] = le_medium_bayes.transform([medium_b])[0]
                if "strain_encoded" in features_bayes and strain_b is not None:
                    test_input["strain_encoded"] = le_strain_bayes.transform([strain_b])[0]
                
                for col in features_bayes:
                    if col not in test_input.columns:
                        test_input[col] = 0
                
                test_input = test_input[features_bayes]
                proba_scenario = model_bayes.predict_proba(test_input)[0]
                
                scenarios_data.append({
                    "Growth": growth_val,
                    "OD": od_val,
                    "Prédiction": le_bayes.inverse_transform([model_bayes.predict(test_input)[0]])[0],
                    "Confiance": f"{max(proba_scenario):.2%}"
                })
        
        scenarios_df = pd.DataFrame(scenarios_data)
        st.dataframe(scenarios_df, use_container_width=True)
        
        # =========================
        # INTERPRETATION
        # =========================
        st.subheader("Interpretation bayesienne")

        st.markdown(f"**Le modele bayesien estime que les conditions observees produisent un biofilm de type {label_bayes}.**")

        st.markdown("**Parametres analyses:**")
        st.write(f"- Croissance (Growth) = {growth_b}")
        st.write(f"- OD brute = {od_b}")
        st.write(f"- Blank = {blank_b}")
        st.write(f"- OD corrigee = {od_b - blank_b:.3f}")

        if medium_b:
            st.write(f"- Medium = {medium_b}")
        if strain_b:
            st.write(f"- Souche = {strain_b}")
            
        st.markdown("**Probabilite maximale:**")
        st.write(f"{max(proba_bayes):.3f} ({max(proba_bayes)*100:.1f}%)")

        st.info("La decision est basee sur une distribution probabiliste des classes microbiologiques")

        # =========================
        # NIVEAU DE CONFIANCE
        # =========================
        st.subheader("Niveau de confiance de la prediction")

        confiance = max(proba_bayes)
        if confiance > 0.8:
            st.success(f"Prediction tres fiable - Confiance: {confiance*100:.1f}%")
            st.markdown("Recommandation: Decision peut etre prise avec assurance")
        elif confiance > 0.6:
            st.warning(f"Prediction moderee - Confiance: {confiance*100:.1f}%")
            st.markdown("Recommandation: Validation supplementaire recommandee")
        else:
            st.error(f"Prediction incertaine - Confiance: {confiance*100:.1f}%")
            st.markdown("Recommandation: Refaire les mesures ou consulter un expert")
        
          # =========================
        # GRAPHE DE DEPENDANCE AVEC FLECHES ET PROBABILITES
        # =========================
        st.subheader("🎯 Visualisation du réseau de dépendances")
        
        # Créer un graphique de réseau avec flèches et probabilités
        import plotly.graph_objects as go
        
        # Définir les nœuds avec leurs positions
        nodes = {
            "Growth": {"x": 0, "y": 1.2, "color": "#87CEEB", "prob": f"{growth_b:.1f}"},
            "OD": {"x": 0, "y": 0.4, "color": "#87CEEB", "prob": f"{od_b - blank_b:.2f}"},
            "Medium": {"x": 1.2, "y": 1.2, "color": "#87CEEB", "prob": medium_b if medium_b else "N/A"},
            "Strain": {"x": 1.2, "y": 0.4, "color": "#87CEEB", "prob": strain_b if strain_b else "N/A"},
            "Biofilm": {"x": 0.6, "y": 0.8, "color": "#FFA07A", "prob": f"{max(proba_bayes):.1%}"}
        }
        
        # Créer les arêtes avec flèches
        edges = [
            {"from": "Growth", "to": "Biofilm", "color": "gray", "width": 2},
            {"from": "OD", "to": "Biofilm", "color": "gray", "width": 2},
            {"from": "Medium", "to": "Biofilm", "color": "gray", "width": 2},
            {"from": "Strain", "to": "Biofilm", "color": "gray", "width": 2}
        ]
        
        # Créer les traces pour les arêtes (avec flèches)
        edge_traces = []
        for edge in edges:
            # Coordonnées de départ et d'arrivée
            x0, y0 = nodes[edge["from"]]["x"], nodes[edge["from"]]["y"]
            x1, y1 = nodes[edge["to"]]["x"], nodes[edge["to"]]["y"]
            
            # Calculer la direction pour la flèche
            dx = x1 - x0
            dy = y1 - y0
            length = (dx**2 + dy**2)**0.5
            
            # Réduire la longueur pour ne pas toucher le nœud
            if length > 0:
                dx = dx / length * (length - 0.15)
                dy = dy / length * (length - 0.15)
                x1_adj = x0 + dx
                y1_adj = y0 + dy
            else:
                x1_adj, y1_adj = x1, y1
            
            # Ajouter la ligne
            edge_traces.append(go.Scatter(
                x=[x0, x1_adj, None],
                y=[y0, y1_adj, None],
                line=dict(width=edge["width"], color=edge["color"]),
                mode='lines',
                hoverinfo='none',
                showlegend=False
            ))
            
            # Ajouter la flèche (triangle)
            # Calculer l'angle de la flèche
            angle = np.arctan2(dy, dx)
            arrow_length = 0.08
            
            # Point de la flèche
            arrow_x = x1_adj
            arrow_y = y1_adj
            
            # Ailes de la flèche
            arrow_left_x = arrow_x - arrow_length * np.cos(angle - np.pi/6)
            arrow_left_y = arrow_y - arrow_length * np.sin(angle - np.pi/6)
            arrow_right_x = arrow_x - arrow_length * np.cos(angle + np.pi/6)
            arrow_right_y = arrow_y - arrow_length * np.sin(angle + np.pi/6)
            
            edge_traces.append(go.Scatter(
                x=[arrow_x, arrow_left_x, arrow_right_x, arrow_x],
                y=[arrow_y, arrow_left_y, arrow_right_y, arrow_y],
                fill='toself',
                fillcolor=edge["color"],
                line=dict(width=0),
                mode='lines',
                hoverinfo='none',
                showlegend=False
            ))
        
        # Créer les traces pour les nœuds
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node_name, node_info in nodes.items():
            node_x.append(node_info["x"])
            node_y.append(node_info["y"])
            node_colors.append(node_info["color"])
            
            # Taille du nœud (plus grand pour Biofilm)
            if node_name == "Biofilm":
                node_sizes.append(60)
                # Texte avec probabilité
                node_text.append(f"<b>{node_name}</b><br>Prob: {node_info['prob']}")
            else:
                node_sizes.append(40)
                # Texte avec valeur
                node_text.append(f"<b>{node_name}</b><br>{node_info['prob']}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n for n in nodes.keys()],
            textposition="bottom center",
            textfont=dict(size=12, color="black"),
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='DarkSlateGrey'),
                symbol='circle'
            ),
            showlegend=False
        )
        
        # Créer la figure
        fig_network = go.Figure(data=edge_traces + [node_trace],
                                layout=go.Layout(
                                    title=dict(
                                        text="<b>Réseau de Dépendances Bayésien</b>",
                                        font=dict(size=16)
                                    ),
                                    showlegend=False,
                                    hovermode='closest',
                                    xaxis=dict(
                                        showgrid=False, 
                                        zeroline=False, 
                                        showticklabels=False,
                                        range=[-0.3, 1.6]
                                    ),
                                    yaxis=dict(
                                        showgrid=False, 
                                        zeroline=False, 
                                        showticklabels=False,
                                        range=[0, 1.6]
                                    ),
                                    plot_bgcolor='white',
                                    height=500,
                                    shapes=[
                                        # Ajouter un cadre autour du graphe
                                        dict(
                                            type="rect",
                                            xref="paper", yref="paper",
                                            x0=0, y0=0, x1=1, y1=1,
                                            line=dict(color="lightgray", width=1)
                                        )
                                    ]
                                ))
        
        # Ajouter un titre pour chaque nœud
        for node_name, node_info in nodes.items():
            fig_network.add_annotation(
                x=node_info["x"],
                y=node_info["y"] - 0.12,
                text=f"<i>{node_info['prob']}</i>",
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="white",
                borderpad=2
            )
        
        st.plotly_chart(fig_network, use_container_width=True, key="bayesian_network")
        
        # =========================
        # GRAPHE ALTERNATIF AVEC MATPLOTLIB (plus fiable pour les flèches)
        # =========================
        st.subheader("📊 Réseau Bayésien - Vue alternative")
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig_mpl, ax_mpl = plt.subplots(figsize=(10, 6))
        
        # Définir les positions des nœuds
        pos_mpl = {
            "Growth": (0.1, 0.8),
            "Medium": (0.4, 0.8),
            "OD": (0.1, 0.4),
            "Strain": (0.4, 0.4),
            "Biofilm": (0.7, 0.6)
        }
        
        # Dessiner les nœuds
        for node_name, (x, y) in pos_mpl.items():
            if node_name == "Biofilm":
                # Nœud orange pour Biofilm
                circle = plt.Circle((x, y), 0.08, color="#FFA07A", ec="black", linewidth=2)
                ax_mpl.add_patch(circle)
                # Ajouter la probabilité
                prob_text = f"{max(proba_bayes):.1%}"
                ax_mpl.text(x, y - 0.12, prob_text, ha='center', va='top', fontsize=9, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                # Nœuds bleus pour les variables d'entrée
                circle = plt.Circle((x, y), 0.07, color="#87CEEB", ec="black", linewidth=2)
                ax_mpl.add_patch(circle)
                # Ajouter la valeur
                if node_name == "Growth":
                    value_text = f"{growth_b}"
                elif node_name == "OD":
                    value_text = f"{od_b - blank_b:.2f}"
                elif node_name == "Medium":
                    value_text = medium_b if medium_b else "N/A"
                elif node_name == "Strain":
                    value_text = strain_b if strain_b else "N/A"
                ax_mpl.text(x, y - 0.1, value_text, ha='center', va='top', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            
            # Texte du nœud
            ax_mpl.text(x, y + 0.08, node_name, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Dessiner les flèches
        for from_node, to_node in [("Growth", "Biofilm"), ("OD", "Biofilm"), 
                                    ("Medium", "Biofilm"), ("Strain", "Biofilm")]:
            x1, y1 = pos_mpl[from_node]
            x2, y2 = pos_mpl[to_node]
            
            # Calculer la direction
            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            
            # Ajuster pour ne pas toucher les bords des cercles
            if from_node == "Biofilm":
                radius = 0.08
            else:
                radius = 0.07
            
            # Point de départ ajusté
            if length > 0:
                start_x = x1 + (dx / length) * radius
                start_y = y1 + (dy / length) * radius
                end_x = x2 - (dx / length) * 0.08
                end_y = y2 - (dy / length) * 0.08
            else:
                start_x, start_y = x1, y1
                end_x, end_y = x2, y2
            
            # Dessiner la flèche
            ax_mpl.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.7))
        
        # Paramètres du graphique
        ax_mpl.set_xlim(0, 1)
        ax_mpl.set_ylim(0, 1)
        ax_mpl.set_aspect('equal')
        ax_mpl.axis('off')
        ax_mpl.set_title("Réseau de Dépendances Bayésien\n(avec probabilités et valeurs)", 
                         fontsize=14, fontweight='bold', pad=20)
        
        # Ajouter une légende
        legend_elements = [
            patches.Patch(facecolor='#87CEEB', edgecolor='black', label='Variables d\'entrée'),
            patches.Patch(facecolor='#FFA07A', edgecolor='black', label='Variable cible (Biofilm)')
        ]
        ax_mpl.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)
        
        # Explication du graphe
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 10px;">
        <b>📖 Interprétation du graphe:</b><br>
        🔵 <b>Nœuds bleus</b>: Variables d'entrée (Growth, OD, Medium, Strain)<br>
        🟠 <b>Nœud orange</b>: Variable cible (Biofilm)<br>
        ➡️ <b>Flèches</b>: Dépendances causales - ces variables influencent la formation du biofilm<br>
        📊 <b>Probabilités</b>: La probabilité affichée sur le nœud Biofilm correspond à la confiance de la prédiction (valeur maximale)<br>
        📝 <b>Valeurs</b>: Les valeurs sous chaque nœud montrent les paramètres saisis
        </div>
        """, unsafe_allow_html=True)