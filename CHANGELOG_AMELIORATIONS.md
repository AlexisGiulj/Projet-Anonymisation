# Changelog - Améliorations Majeures de l'Application

**Date** : 24 novembre 2025
**Commit** : 7b9cbd9

## 🎯 Vue d'Ensemble

L'application a été considérablement améliorée avec de nouvelles fonctionnalités interactives, des simulations d'attaques réelles, et un contenu éducatif enrichi. Le nombre d'onglets est passé de 5 à 8 pour une meilleure organisation.

---

## 🆕 Nouvelles Fonctionnalités

### 1. **Paramètres de Privacy Modulables** ⚙️

**Avant** : Les paramètres (k, epsilon) étaient codés en dur dans le code.

**Maintenant** : Sliders interactifs dans la sidebar permettant d'ajuster :
- **k** (pour Random Add/Del, Random Switch, k-degree anonymity, Generalization, Probabilistic) : 2-50
- **epsilon** (pour EdgeFlip, Laplace, Probabilistic) : 0.1-3.0

**Feedback en temps réel** :
- ✅ Privacy Forte (ε < 1.0) : Perte ≤ e^ε
- ⚠️ Privacy Moyenne (1.0 ≤ ε < 2.0)
- ❌ Privacy Faible (ε ≥ 2.0)

### 2. **Nouvelle Structure avec 8 Onglets**

| Onglet | Nouveau/Modifié | Contenu |
|--------|-----------------|---------|
| 📊 Résultats | Inchangé | Visualisations graphiques comparatives |
| 📖 Définitions | **NOUVEAU** | Définitions formelles + intuitions pour chaque concept |
| 📈 Métriques Utilité | **NOUVEAU** | Densité, clustering, diamètre, corrélation degrés |
| 🔒 Métriques Privacy | **NOUVEAU** | k-anonymité, epsilon, garanties spécifiques |
| 🎯 Simulations d'Attaques | **NOUVEAU** | Degree Attack + Subgraph Attack interactives |
| 🛡️ Attaques & Garanties | Déplacé | Protections et vulnérabilités par méthode |
| 📚 Dict. Attaques | **NOUVEAU** | 7 attaques documentées avec exemples Karate Club |
| 🔍 Dict. Propriétés | **NOUVEAU** | 12 propriétés de graphes expliquées |

### 3. **Onglet Définitions** 📖

Exploration interactive des 5 concepts d'anonymisation :

**Pour chaque concept** :
- 📝 **Définition Formelle** : Notation mathématique rigoureuse
- 🔢 **Formule** : Expression mathématique complète
- 💡 **Intuition** : Explication en langage naturel avec analogies
- 🔒 **Garantie de Privacy** : Promesse formelle offerte
- ⚙️ **Signification des Paramètres** : Interprétation de k, epsilon, etc.

**Exemple pour k-anonymité** :
```
Définition : ∀v ∈ V, |{u ∈ V : deg(u) = deg(v)}| ≥ k
Intuition : Comme se cacher dans une foule - si k personnes ont la même taille,
            vous ne pouvez pas être distingué parmi elles.
Garantie : P(ré-identification | degré) ≤ 1/k
```

### 4. **Métriques d'Utilité** 📈

Mesure la préservation de l'utilité du graphe :

**Métriques de Base** :
- Nombre de nœuds/arêtes
- Densité (avec delta par rapport à l'original)
- Coefficient de clustering moyen (avec delta)

**Métriques Globales** :
- Diamètre du graphe
- Longueur moyenne des chemins
- Corrélation de Spearman des degrés (0 = aucune, 1 = parfaite)

**Visualisations** :
- Graphique en barres : Arêtes préservées/ajoutées/supprimées
- Barre de progression : Taux de modification
- Feedback coloré :
  - ✅ < 10% : Utilité très bien préservée
  - ℹ️ 10-30% : Utilité correctement préservée
  - ⚠️ > 30% : Modifications importantes

### 5. **Métriques Privacy** 🔒

Métriques spécifiques à chaque type de méthode :

**k-anonymity** :
- k requis vs ensemble d'anonymat minimum
- Probabilité de ré-identification : P = 1/k
- Indicateur ✅/❌ si k-anonymité satisfaite
- Barre de progression du risque

**Differential Privacy (EdgeFlip, Laplace)** :
- Budget epsilon actuel
- Borne de perte : e^ε ≈ X.XX
- Niveau de privacy (Forte/Moyenne/Faible)
- Probabilité de flip (EdgeFlip)
- Nombre attendu d'arêtes bruitées

**Probabilistic (k,ε)-obfuscation** :
- Nombre de graphes candidats (k)
- Tolérance epsilon
- Entropie minimale : log(k) - ε
- Facteur de confusion

**Generalization** :
- Taille min/moy des clusters
- Probabilité maximale de ré-identification
- Ratio intra-cluster/inter-cluster

### 6. **Simulations d'Attaques** 🎯

**Interface Interactive** :
- Sélection du nœud cible (0 à n-1)
- Choix du type d'attaque :
  - **Degree Attack** : Recherche par degré uniquement
  - **Subgraph Attack** : Recherche par degré + triangles

**Résultats de Simulation** :
- ✅/❌ Succès ou échec de l'attaque
- Explication détaillée
- Liste des nœuds candidats trouvés
- Probabilité de succès si ambiguë : 1/|candidats|

**Exemple de Résultat** :
```
⚠️ Ré-identification ambiguë : 3 nœuds ont le degré 16.
Probabilité de succès : 33.3%

Candidats trouvés : [0, 5, 23]
```

**Section Éducative** :
- Explications détaillées de chaque type d'attaque
- Méthodes de protection efficaces

### 7. **Dictionnaire des Attaques** 📚

7 attaques documentées en détail :

| Attaque | Sévérité | Protection Efficace |
|---------|----------|---------------------|
| Degree Attack | Moyenne | k-degree anonymity, Randomization |
| Active Attack (Sybil) | Élevée | Differential Privacy |
| Passive Attack (Interne) | Moyenne | Randomization, Generalization |
| Subgraph Attack | Élevée | Generalization, DP |
| Neighborhood Attack | Très élevée | Generalization, (k,ε)-obfuscation |
| Walk-based Attack | Moyenne | DP sur marches aléatoires |
| Auxiliary Info Attack | Très élevée | Differential Privacy seule |

**Pour chaque attaque** :
- 📝 Description détaillée
- 💡 Exemple concret (souvent sur Karate Club)
- ⚠️ Niveau de sévérité
- 🛡️ Méthode de protection recommandée

**Exemples Concrets sur Karate Club** :
- Scénario d'attaque pas à pas
- Étapes détaillées
- Taux de succès :
  - Sans protection
  - Avec k-anonymity
  - Avec Randomization
  - Avec Differential Privacy
- Code de simulation Python

**Exemple : Degree Attack sur Mr. Hi** :
```
Scénario : L'adversaire sait que Mr. Hi (nœud 0) a 16 connexions

Étapes :
1. Observer le graphe anonymisé
2. Chercher le nœud ayant degré = 16
3. Si unique → Ré-identification réussie !

Taux de succès :
- Sans protection : 100% (degré unique)
- Avec k=2 anonymity : ≤ 50% (au moins 2 nœuds de degré 16)
- Avec randomization : ~40% (degré bruité)
```

### 8. **Dictionnaire des Propriétés** 🔍

12 propriétés de graphes expliquées :

| Propriété | Utilité | Privacy Risk |
|-----------|---------|--------------|
| Degré | Critique | Élevé |
| Clustering Coefficient | Élevée | Moyen |
| Betweenness Centrality | Critique | Élevé |
| Closeness Centrality | Moyenne | Moyen |
| Eigenvector Centrality | Élevée | Élevé |
| Densité | Moyenne | Faible |
| Diamètre | Moyenne | Faible |
| Average Path Length | Élevée | Faible |
| Degree Distribution | Critique | Moyen |
| Modularité | Élevée | Moyen |
| Triangles | Élevée | Moyen |
| Assortativité | Moyenne | Faible |

**Pour chaque propriété** :
- 📝 Définition claire
- 🔢 Formule mathématique
- 💡 Exemple concret
- 📊 Importance pour l'utilité (Critique/Élevée/Moyenne/Faible)
- ⚠️ Risque pour la privacy (Élevé/Moyen/Faible)

**Calcul en Temps Réel** :
- Valeurs calculées pour le graphe actuellement anonymisé
- Exemples : degré moyen, clustering, densité, diamètre, etc.

---

## 🔧 Améliorations Techniques

### Nouvelles Fonctions Python

**1. `simulate_degree_attack(G_orig, G_anon, target_node=0)`**
- Simule une attaque par degré
- Retourne : succès/échec, candidats, explication, probabilité

**2. `simulate_subgraph_attack(G_orig, G_anon, target_node=0)`**
- Simule une attaque par sous-graphe (triangles)
- Plus sophistiquée que l'attaque par degré seul
- Combine degré + nombre de triangles

**3. `calculate_utility_metrics(G_orig, G_anon)`**
- Calcule toutes les métriques d'utilité
- Retourne : densité, clustering, diamètre, corrélation, etc.
- Gère les graphes non connexes (composante principale)

**4. `calculate_privacy_metrics_separated(G_orig, G_anon, method_key, method_params)`**
- Calcule les métriques de privacy spécifiques à chaque méthode
- Séparé de calculate_privacy_guarantees pour meilleure organisation
- Retourne : k-value, epsilon, probabilités, etc.

### Gestion des Paramètres Dynamiques

**Avant** :
```python
G_anon = anonymizer.random_add_del(**method['params'])
```

**Maintenant** :
```python
dynamic_params = {}  # Récupérés des sliders
st.session_state.method_params = dynamic_params
G_anon = anonymizer.random_add_del(**dynamic_params)
```

Les paramètres sont maintenant :
- Sauvegardés dans `st.session_state.method_params`
- Passés aux fonctions de calcul de métriques
- Affichés dans les onglets pour référence

---

## 📚 Contenu Éducatif Enrichi

### Définitions Formelles ET Intuitives

**Exemple : Privacy Différentielle**

**Définition Formelle** :
```
Un algorithme A satisfait la ε-differential privacy si pour deux graphes
G et G' différant d'une seule arête, et pour tout résultat O :

P[A(G) = O] ≤ e^ε × P[A(G') = O]
```

**Intuition** :
```
La privacy différentielle garantit que la présence ou l'absence d'un individu
change très peu les résultats. C'est comme ajouter du bruit calibré :
un adversaire ne peut pas déterminer si vous êtes dans la base de données,
même avec une connaissance parfaite de tous les autres.

Le paramètre ε contrôle le "budget de privacy" :
- ε petit = protection forte, données bruitées
- ε grand = protection faible, données préservées
```

### Exemples Concrets Systématiques

Chaque attaque et propriété est illustrée avec :
- Des cas sur le **graphe Karate Club** (familier à l'utilisateur)
- Des **valeurs numériques réelles** (ex: "Mr. Hi a degré 16")
- Des **scénarios réalistes** (ex: "adversaire connaît les connexions")
- Des **taux de succès quantifiés** (ex: "95% sans protection, 20% avec")

### Trade-offs Privacy vs Utilité

**Visualisations Claires** :
- Graphiques côte à côte : modifications vs préservation
- Barres de progression : % de modification, % de risque
- Indicateurs colorés :
  - Vert : Bon équilibre
  - Orange : Compromis acceptable
  - Rouge : Trade-off défavorable

---

## 📊 Statistiques

### Lignes de Code

- **Fichier principal** : graph_anonymization_app.py
  - Avant : ~1,470 lignes
  - Après : ~2,063 lignes (+593 lignes, +40%)

- **Nouveau fichier** : definitions_and_attacks.py
  - Contenu : ~550 lignes
  - Dictionnaires : 5 concepts + 7 attaques + 12 propriétés + 3 exemples

### Fonctionnalités

- **Onglets** : 5 → 8 (+3 onglets, +60%)
- **Fonctions Python** : 9 → 13 (+4 fonctions)
- **Paramètres modulables** : 0 → 7 paramètres
- **Attaques documentées** : 0 → 7 attaques
- **Propriétés documentées** : 0 → 12 propriétés
- **Simulations interactives** : 0 → 2 types d'attaques

---

## 🚀 Utilisation

### Lancer l'Application

```bash
cd GraphAnonymizationDemo
streamlit run graph_anonymization_app.py
```

Ou via le lanceur Windows :
```bash
LANCER.bat
# Choisir Option 1
```

### Workflow Recommandé

1. **Choisir un graphe** (Karate Club recommandé pour les exemples)
2. **Sélectionner une méthode** d'anonymisation
3. **Ajuster les paramètres** avec les sliders (epsilon, k)
4. **Anonymiser** et explorer les 8 onglets :
   - 📊 Voir les résultats visuels
   - 📖 Comprendre les définitions
   - 📈 Évaluer l'utilité préservée
   - 🔒 Mesurer la privacy obtenue
   - 🎯 Tester des attaques réelles
   - 🛡️ Vérifier les garanties
   - 📚 Apprendre sur les attaques
   - 🔍 Explorer les propriétés

### Cas d'Usage Pédagogiques

**Pour un cours** :
1. Montrer l'onglet **Définitions** pour introduire les concepts
2. Utiliser l'onglet **Dict. Propriétés** pour expliquer les métriques
3. Lancer une anonymisation et comparer **Utilité** vs **Privacy**
4. Simuler des **Attaques** pour illustrer les risques
5. Consulter le **Dict. Attaques** pour voir les protections

**Pour une présentation** :
1. Commencer par **Karate Club** (graphe familier)
2. Essayer **k=2 degree anonymity** avec le slider
3. Montrer l'onglet **Simulations d'Attaques** : attaquer Mr. Hi (nœud 0)
4. Comparer avec **Differential Privacy** (epsilon=0.5)
5. Re-simuler l'attaque → taux de succès diminué

---

## 🔄 Comparaison Avant/Après

| Aspect | Avant | Après |
|--------|-------|-------|
| **Paramètres** | Codés en dur | Sliders interactifs |
| **Métriques** | Mélangées | Séparées Utilité/Privacy |
| **Attaques** | Liste statique | Simulations interactives |
| **Définitions** | Texte dans code | Onglet dédié avec formules |
| **Propriétés** | Non documentées | 12 propriétés expliquées |
| **Trade-off** | Pas visualisé | Graphiques et indicateurs |
| **Pédagogie** | Formules seules | Formules + Intuitions + Exemples |
| **Feedback** | Aucun | En temps réel sur epsilon |

---

## 📝 Notes de Version

**Version** : 2.0
**Compatibilité** : Python 3.8+
**Dépendances** : Inchangées (streamlit, networkx, matplotlib, numpy, scipy)

**Fichiers Ajoutés** :
- `definitions_and_attacks.py` : Dictionnaires de contenu éducatif

**Fichiers Modifiés** :
- `graph_anonymization_app.py` : Application principale

**Fichiers Temporaires** (non versionnés) :
- `graph_anonymization_app_backup.py` : Backup de l'ancienne version
- `new_tabs_content.py` : Fichier de travail (peut être supprimé)

---

## 🙏 Remerciements

Cette mise à jour majeure a été réalisée pour rendre l'application plus **pédagogique**, **interactive** et **complète**. Elle est maintenant un véritable **outil d'apprentissage** pour comprendre l'anonymisation de graphes sociaux.

Merci à NGUYEN Huu-Hiep pour sa thèse qui a inspiré cette application.

---

**Pour toute question ou suggestion** : Ouvrir une issue sur GitHub
**Repository** : https://github.com/AlexisGiulj/Projet-Anonymisation
