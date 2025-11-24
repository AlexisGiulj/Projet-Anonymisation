# 🔒 Anonymisation de Graphes Sociaux

Application interactive d'anonymisation de graphes sociaux basée sur la thèse de NGUYEN Huu-Hiep (2016).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📖 Description

Cette application démontre **7 méthodes d'anonymisation** de graphes sociaux avec :
- 🎨 Visualisations interactives
- 📐 Explications mathématiques détaillées
- 💬 Explications en langage naturel
- 📊 Métriques d'anonymisation
- 🛡️ Analyse des attaques et garanties de privacy
- 🥋 Exemples concrets sur le graphe Karate Club

## 🚀 Démo en Ligne

[Lancer l'application](https://votre-app.streamlit.app) (À déployer sur Streamlit Cloud)

## 🔬 Méthodes Implémentées

### 1. Randomisation
- **Random Add/Del** : Ajoute/supprime k arêtes aléatoirement
- **Random Switch** : Échange k paires d'arêtes (préserve les degrés)

### 2. K-Anonymisation
- **k-degree anonymity** : Garantit ≥k nœuds par degré

### 3. Généralisation
- **Super-nodes** : Regroupe les nœuds en clusters

### 4. Approches Probabilistes
- **(k,ε)-obfuscation** : Graphe incertain avec probabilités

### 5. Privacy Différentielle
- **EdgeFlip** : Randomized Response Technique
- **Laplace** : Mécanisme de Laplace

## 💻 Installation Locale

### Prérequis
- Python 3.8+
- pip

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/graph-anonymization-demo.git
cd graph-anonymization-demo

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run graph_anonymization_app.py
```

### Installation sur Windows

Double-cliquez sur `LANCER.bat` pour un menu interactif avec :
- Option 1 : Lancer l'application Streamlit
- Option 2 : Version batch (génère des PNG)
- Option 3 : Installer les dépendances
- Option 4 : Ouvrir le dossier
- Option 5 : Lire la documentation

## 📚 Utilisation

1. **Sélectionner un graphe** : Karate Club, graphe aléatoire petit/moyen
2. **Choisir une méthode** : 7 méthodes disponibles
3. **Anonymiser** : Cliquer sur "🚀 Anonymiser le Graphe"
4. **Explorer** :
   - Onglet **Résultats** : Visualisations comparatives
   - Onglet **Explications** : Théorie mathématique et intuitions
   - Onglet **Métriques** : Statistiques détaillées + Garanties de privacy
   - Onglet **Attaques & Garanties** : Analyse de sécurité complète
   - Onglet **Anonymisation** : Comprendre les taux d'anonymisation

## 🎯 Fonctionnalités

### Visualisations
- **Graphe original vs anonymisé** côte-à-côte
- **Arêtes colorées** : bleues (préservées), rouges pointillées (ajoutées)
- **Super-nodes** : Cercles autour des clusters, tailles proportionnelles
- **Distributions de degrés** : Histogrammes comparatifs

### Métriques
- Modification des arêtes (ajoutées, supprimées, préservées)
- Changements de degrés
- Propriétés structurelles (clustering, densité)
- **Garanties de privacy spécifiques** à chaque méthode

### Attaques & Garanties
- ✅ Attaques contre lesquelles la méthode protège
- ⚠️ Vulnérabilités connues
- ✅ Avantages
- ❌ Inconvénients
- 🥋 Exemples sur Karate Club

## 📦 Structure du Projet

```
GraphAnonymizationDemo/
├── graph_anonymization_app.py      # Application Streamlit principale
├── graph_anonymization_demo.py     # Version batch (génère PNG)
├── method_details.py               # Détails des attaques et garanties
├── requirements.txt                # Dépendances Python
├── LANCER.bat                      # Lanceur Windows
├── README_GITHUB.md                # Ce fichier
├── README.md                       # Documentation technique complète
├── README_APP.md                   # Guide d'utilisation
└── GUIDE_EXPOSE.md                 # Guide pour présentation
```

## 🎓 Fondements Théoriques

Basé sur la thèse :
**NGUYEN Huu-Hiep** (2016). *Anonymisation de Graphes Sociaux*.
Université de Lorraine, LORIA.

### Papiers Clés
- Hay et al. (2008) - Randomization
- Liu & Terzi (2008) - k-degree anonymity
- Backstrom et al. (2007) - De-anonymization attacks
- Boldi et al. (2012) - (k,ε)-obfuscation
- Sala et al. (2011) - Differential Privacy for Graphs

## 🛡️ Garanties de Privacy

### k-degree anonymity
- Garantie : P(ré-identification | degré) ≤ 1/k
- Protège contre : Attaques par degré
- Vulnérable à : Attaques par sous-graphe de voisinage

### Differential Privacy
- Garantie : ε-DP (indépendante de la connaissance de l'adversaire)
- Protège contre : TOUTES les attaques
- Trade-off : Privacy maximale vs utilité

### Généralisation
- Garantie : k-anonymity structurelle
- Protège contre : Ré-identification totale
- Trade-off : Privacy maximale vs perte d'information

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Ouvrir une issue pour signaler un bug
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation
- Ajouter de nouvelles méthodes d'anonymisation

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- NGUYEN Huu-Hiep pour sa thèse fondatrice
- NetworkX pour la manipulation de graphes
- Streamlit pour l'interface interactive
- La communauté de recherche en privacy-preserving data publishing

## 📧 Contact

Pour toute question ou suggestion, ouvrez une issue sur GitHub.

---

**⭐ Si ce projet vous est utile, n'oubliez pas de lui donner une étoile !**
