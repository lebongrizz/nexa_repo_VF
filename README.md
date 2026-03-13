# 🏥 Health-InsurTech — Prédiction des Frais de Santé

Outil web transparent d'estimation des frais médicaux annuels, développé avec **Streamlit**, conforme **RGPD** et **WCAG 2.1 AA**.

---

## 🚀 Lancement rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Placer le dataset dans le même dossier
cp insurance_data.csv ./

# 3. Lancer l'application
streamlit run app.py
```

L'application sera disponible sur `http://localhost:8501`

**Comptes de démonstration :**
| Utilisateur | Mot de passe |
|-------------|-------------|
| admin | admin |
| demo | demo |

---

## 📋 Fonctionnalités

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPIs, corrélations IMC/âge/charges, boxplots par région |
| 🔮 Simulateur | Formulaire temps réel, prédiction double modèle + facteurs d'influence |
| 🤖 Modèle & Biais | Performances R²/MAE, coefficients, analyse des biais par groupe |
| 🔒 Conformité RGPD | Note d'impact, mesures PII, droits des personnes, accessibilité |

---

## 🤖 Modèles ML

- **Régression Linéaire** — R² = 0.78, MAE = 4 181 € — coefficients interprétables
- **Arbre de Décision** (max_depth=4) — R² = 0.86, MAE = 2 697 € — règles explicables
- Features utilisées : `age`, `bmi`, `children`, `smoker`, `sex`, `region`

---

## 🔒 RGPD & Sécurité

- ✅ Consentement explicite à l'entrée (Art. 6.1.a & Art. 9)
- ✅ Données PII supprimées du modèle (nom, email, NSS, IP, etc.)
- ✅ Authentification par hachage SHA-256
- ✅ Aucune persistance des données de simulation
- ✅ Logs d'accès sans données personnelles

---

## 🌐 Déploiement sur Streamlit Cloud

1. Pusher le code sur GitHub (avec `insurance_data.csv`)
2. Connecter le dépôt sur [share.streamlit.io](https://share.streamlit.io)
3. Définir `app.py` comme point d'entrée
4. L'URL publique HTTPS est générée automatiquement

---

## ♿ Accessibilité WCAG 2.1 AA

- Contraste couleurs ≥ 4.5:1
- Navigation clavier complète
- Labels explicites sur tous les champs
- Police corps ≥ 16px, design responsive
