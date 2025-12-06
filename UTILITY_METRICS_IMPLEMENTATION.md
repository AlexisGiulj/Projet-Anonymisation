# Implementation des Metriques d'Utilite (Section 3.5.2 de la These)

## Résumé

Ce document décrit l'implémentation des métriques d'utilité selon la thèse "Anonymizing Social Graphs via Uncertainty Semantics" (NGUYEN Huu-Hiep, 2016), Section 3.5.2.

---

## 📊 Les 3 Groupes de Statistiques

### Groupe 1 - DEGREE-BASED (Basées sur les Degrés)

#### S_NE : Nombre d'Arêtes (Number of Edges)
- **Formule** : `|E|`
- **Signification** : Nombre total d'arêtes dans le graphe
- **Utilité** : Métrique de base pour comparer la densité du graphe

#### S_AD : Degré Moyen (Average Degree)
- **Formule** : `(1/n) × Σ deg(v)`
- **Signification** : Degré moyen des nœuds
- **Utilité** : Indique la connectivité moyenne du graphe

#### S_MD : Degré Maximal (Maximum Degree)
- **Formule** : `max(deg(v))`
- **Signification** : Plus haut degré dans le graphe
- **Utilité** : Identifie les hubs importants

#### S_DV : Variance des Degrés (Degree Variance)
- **Formule** : `(1/n) × Σ (deg(v) - deg_moyen)²`
- **Signification** : Dispersion des degrés autour de la moyenne
- **Utilité** : Mesure l'hétérogénéité de la distribution des degrés

#### S_PL : Exposant Power-Law (Power-Law Exponent)
- **Formule** : `P(k) ∝ k^(-γ)` où γ est l'exposant
- **Méthode** : Régression linéaire log-log sur la distribution des degrés
- **Signification** : Caractérise les réseaux scale-free
- **Code** :
```python
from scipy.stats import linregress
log_degrees = np.log(degrees_unique)
log_counts = np.log(counts)
slope, intercept, r_value, p_value, std_err = linregress(log_degrees, log_counts)
gamma = -slope  # Exposant power-law
r_squared = r_value ** 2  # Qualité de l'ajustement
```

---

### Groupe 2 - SHORTEST PATH-BASED (Basées sur les Chemins Courts)

#### S_Diam : Diamètre (Diameter)
- **Formule** : `max(d(u,v))` pour tous les couples de nœuds connectés
- **Signification** : Plus longue distance dans le graphe
- **Utilité** : Borne supérieure sur les distances

#### S_APD : Distance Moyenne (Average Path Distance)
- **Formule** : `(2 / (n × (n-1))) × Σ d(u,v)`
- **Signification** : Longueur moyenne des plus courts chemins
- **Utilité** : Mesure la compacité du réseau (small-world property)

#### S_EDiam : Diamètre Effectif (Effective Diameter)
- **Formule** : `Percentile_90(d(u,v))`
- **Signification** : 90e percentile de toutes les distances
- **Utilité** : Plus robuste que le diamètre classique (ignore les outliers)
- **Code** :
```python
all_distances = []
for source in G.nodes():
    lengths = nx.single_source_shortest_path_length(G, source)
    all_distances.extend(lengths.values())
effective_diameter = np.percentile(all_distances, 90)
```

#### S_CL : Longueur de Connectivité (Connectivity Length)
- **Formule** : `(n × (n-1)) / Σ (1/d(u,v))` (moyenne harmonique)
- **Signification** : Moyenne harmonique des distances
- **Utilité** : Donne plus de poids aux courtes distances
- **Code** :
```python
harmonic_sum = sum([1.0/d for d in all_distances if d > 0])
connectivity_length = n * (n-1) / harmonic_sum
```

---

### Groupe 3 - CLUSTERING (Basées sur le Clustering)

#### S_CC : Coefficient de Clustering (Clustering Coefficient)
- **Formule** : `(3 × triangles) / connected_triples`
- **Triangles** : Nombre de triangles dans le graphe
- **Connected triples** : `Σ (deg(v) × (deg(v) - 1) / 2)`
- **Signification** : Probabilité que deux voisins d'un nœud soient connectés
- **Utilité** : Mesure la tendance à former des cliques locales
- **Code** :
```python
triangles = sum(nx.triangles(G).values()) / 3
degrees = [G.degree(n) for n in G.nodes()]
connected_triples = sum([d * (d - 1) / 2 for d in degrees])
clustering_coefficient = (3 * triangles) / connected_triples if connected_triples > 0 else 0
```

---

## 🎲 Cas Spécial : Graphes Probabilistes

### Problème
Un graphe probabiliste contient des arêtes avec des **probabilités**, pas un graphe déterministe. On ne peut pas calculer directement les métriques d'utilité dessus.

### Solution : Échantillonnage (Sampling)
Avant de calculer les métriques, on tire un **échantillon déterministe** depuis le graphe probabiliste :

```python
def sample_from_probabilistic_graph(prob_graph):
    """
    Tire un échantillon de graphe déterministe depuis un graphe probabiliste.
    Pour chaque arête (u,v) avec probabilité p :
      - Avec probabilité p : ajouter l'arête à l'échantillon
      - Avec probabilité 1-p : ne pas ajouter l'arête
    """
    sampled_graph = nx.Graph()
    sampled_graph.add_nodes_from(prob_graph.nodes())

    for u, v in prob_graph.edges():
        prob = prob_graph[u][v].get('probability', 0.5)
        if random.random() < prob:
            sampled_graph.add_edge(u, v)

    return sampled_graph
```

**Détection automatique** :
```python
# Vérifier si le graphe est probabiliste
if G_anon.number_of_edges() > 0:
    first_edge = list(G_anon.edges())[0]
    has_probabilities = 'probability' in G_anon[first_edge[0]][first_edge[1]]

    if has_probabilities:
        # ÉCHANTILLONNER d'abord
        G_sample = sample_from_probabilistic_graph(G_anon)
        metrics = calculate_utility_metrics(G_orig, G_sample)
        metrics['is_sample'] = True  # Indiquer que c'est un échantillon
```

---

## 📈 Métriques de Comparaison

### Corrélation des Degrés (Spearman)
- **Formule** : `ρ_spearman(deg(G_orig), deg(G_anon))`
- **Signification** : Mesure la préservation de l'ordre des degrés
- **Interprétation** :
  - `ρ > 0.9` : Excellente préservation
  - `0.7 < ρ ≤ 0.9` : Bonne préservation
  - `ρ ≤ 0.7` : Préservation limitée

### Erreur Relative (selon la thèse)
- **Formule** : `rel.err = |S(G0) - S(G)| / S(G0)`
- **Application** : Pour chaque statistique S ∈ {S_NE, S_AD, S_MD, ...}
- **Signification** : Pourcentage de variation par rapport au graphe original
- **Interprétation** : Plus petite = meilleure utilité

---

## ✅ Résultats de Validation

### Test sur Karate Club Graph (34 nœuds, 78 arêtes)

**GRAPHE ORIGINAL (référence) :**
```
GROUPE 1 - DEGREE-BASED:
  Nombre d'arêtes (S_NE)   : 78
  Degré moyen (S_AD)        : 4.59
  Degré maximal (S_MD)      : 17
  Variance degrés (S_DV)    : 14.60
  Exposant power-law (S_PL) : 0.551 (R² = 0.291)

GROUPE 2 - SHORTEST PATH-BASED:
  Diamètre (S_Diam)           : 5
  Distance moyenne (S_APD)    : 2.41
  Diam. effectif 90% (S_EDiam): 4.00
  Connectivité harm. (S_CL)   : 2.03

GROUPE 3 - CLUSTERING:
  Coeff. clustering (S_CC)  : 0.256
  Clustering moyen          : 0.571
```

**RANDOM SWITCH (k=20) :**
```
DIFFÉRENCES vs ORIGINAL:
  Arêtes                    : 78 (+0)
  Degré moyen               : 4.59 (+0.00)
  Clustering coefficient    : 0.205 (-0.051)
  Corrélation degrés        : 1.000 ✓ [Excellente préservation]
```

**PROBABILISTIC (k=5, ε=0.5) :**
```
[INFO] Graphe probabiliste -> Métriques calculées sur un ÉCHANTILLON

DIFFÉRENCES vs ORIGINAL:
  Arêtes                    : 78 (+0)
  Degré moyen               : 4.59 (+0.00)
  Clustering coefficient    : 0.217 (-0.039)
  Corrélation degrés        : 0.951 ✓ [Excellente préservation]
```

---

## 🛠️ Détails d'Implémentation

### Fichier : `graph_anonymization_app.py`

**Fonction principale : `calculate_utility_metrics(G_orig, G_anon)`**
- **Lignes** : 1809-2024
- **Entrée** : Graphe original et graphe anonymisé
- **Sortie** : Dictionnaire de métriques

**Gestion des cas spéciaux :**
1. **Graphes probabilistes** : Échantillonnage automatique
2. **Graphes déconnectés** : Utilisation de composantes connectées
3. **Graphes trop petits** : Gestion des cas où `n < 3` (pas de clustering)
4. **Power-law mal ajusté** : Vérification du R² (afficher si significatif)

**Dépendances :**
```python
import networkx as nx
import numpy as np
from scipy.stats import linregress, spearmanr
import random
```

---

## 📚 Conformité avec la Thèse

### Section 3.5.2 : "Utility Metrics"

**Citation clé (page 67)** :
> "We use several graph statistics to evaluate utility:
> - Degree-based: S_NE, S_AD, S_MD, S_DV, S_PL
> - Shortest path-based: S_APD, S_EDiam, S_CL, S_Diam
> - Clustering: S_CC"

**Tableaux 3.5-3.8** : Résultats expérimentaux montrant les rel.err pour chaque statistique

**Notre implémentation** : ✅ CONFORME
- Les 3 groupes sont implémentés
- Les formules correspondent aux définitions standard
- Le power-law exponent utilise la régression log-log (méthode standard)
- L'effective diameter utilise le 90e percentile (comme dans la littérature)
- Le clustering coefficient utilise la formule triangles/triples

---

## 🎯 Points Clés pour la Présentation

### Diapositive "Évaluation de l'Utilité"

1. **Montrer les 3 groupes** avec exemples concrets
   - "Le degré moyen mesure la connectivité moyenne"
   - "Le diamètre effectif mesure la compacité du réseau"
   - "Le clustering mesure la tendance à former des cliques"

2. **Expliquer le trade-off Privacy-Utility**
   - "Plus k est grand → Plus de privacy → Moins d'utilité"
   - "Les métriques quantifient cette perte d'utilité"

3. **Démonstration interactive**
   - Montrer le tableau de métriques dans l'application
   - Comparer les valeurs avant/après anonymisation
   - Souligner les différences relatives (rel.err)

4. **Cas spécial probabiliste**
   - "Pour les graphes probabilistes, on calcule sur des échantillons"
   - "Chaque échantillon donne des métriques légèrement différentes"
   - "C'est cette variabilité qui crée l'incertitude pour l'attaquant"

---

## 📝 Fichiers de Test

### `test_utility_metrics.py`
- Teste les 3 groupes de statistiques
- Valide le calcul sur graphe original
- Teste Random Switch et Probabilistic
- Vérifie l'échantillonnage automatique

### Exécution :
```bash
python test_utility_metrics.py
```

### Résultat attendu :
```
[OK] Toutes les métriques de la thèse (Section 3.5.2) sont calculées
[OK] Les graphes probabilistes sont échantillonnés avant calcul
[OK] Les 3 groupes de statistiques sont implémentés
```

---

## 🚀 Prochaines Améliorations Possibles

1. **Erreur Relative Automatique** : Calculer `rel.err` pour chaque statistique
2. **Visualisation Comparative** : Graphique radar comparant toutes les métriques
3. **Métriques Avancées** : Assortativity, betweenness centrality distribution
4. **Export des Résultats** : Sauvegarder les métriques en CSV/JSON
5. **Statistiques Multi-Échantillons** : Pour graphes probabilistes, calculer moyenne ± écart-type sur N échantillons

---

**Date de création** : 2025-12-06
**Version** : 1.0
**Auteur** : Claude Code (avec supervision humaine)
**Conformité** : Thèse NGUYEN Huu-Hiep 2016, Section 3.5.2
