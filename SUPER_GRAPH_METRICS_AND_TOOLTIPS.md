# Métriques pour Super-Graphes et Tooltips Interactifs

## Résumé des Améliorations

Ce document décrit les améliorations majeures apportées pour :
1. **Calculer correctement les métriques sur les super-graphes** (méthode de généralisation)
2. **Ajouter des tooltips interactifs** avec définitions et formules pour toutes les métriques
3. **Distinguer les 3 types de graphes** et adapter l'affichage en conséquence

---

## 🏗️ Problème Initial

### Méthode de Généralisation

Lorsqu'on utilise la **méthode de généralisation**, le graphe anonymisé a une **structure complètement différente** :
- **Graphe original** : n nœuds individuels, m arêtes
- **Super-graphe** : k super-nœuds (clusters), arêtes intra-cluster + inter-cluster

**Problème** : Les métriques classiques (densité, clustering, diamètre) n'ont **aucun sens** sur un super-graphe !

### Manque de Documentation

Les métriques étaient affichées sans explication :
- Utilisateur voit "Densité : 0.139" → **Que signifie ce chiffre ?**
- Pas de définition, pas de formule, pas d'interprétation

---

## ✅ Solution Implémentée

### 1. Métriques Spécifiques pour Super-Graphes

#### Nouvelle fonction : `calculate_supergraph_metrics(G_orig, G_super)`

Cette fonction calcule des **métriques adaptées** à la structure en clusters :

**📦 Métriques de Clustering**
```python
num_clusters            # Nombre de super-nœuds (clusters)
min_cluster_size        # Plus petit cluster (doit être ≥ k pour k-anonymity)
avg_cluster_size        # Taille moyenne (≈ n/k)
max_cluster_size        # Plus grand cluster
cluster_size_variance   # Hétérogénéité des tailles
```

**🔗 Métriques d'Arêtes**
```python
intra_cluster_edges     # Arêtes à l'intérieur des clusters (structure locale)
inter_cluster_edges     # Arêtes entre clusters (connexions globales)
num_edges              # Total = intra + inter
intra_ratio            # Proportion intra/total (préservation locale)
inter_ratio            # Proportion inter/total
```

**📊 Perte d'Information**
```python
node_compression_ratio  # k_clusters / n_nodes (combien de compression)
information_loss       # 1 - compression_ratio (combien perdu)
edge_preservation_ratio # edges_anon / edges_orig (arêtes conservées)
```

**🌐 Structure du Super-Graphe**
```python
super_graph_density     # Densité du graphe des clusters (sans self-loops)
avg_cluster_degree      # Nombre moyen de clusters voisins
max_cluster_degree      # Plus connecté des clusters
super_graph_connected   # Est-ce que les clusters forment un graphe connexe ?
super_graph_diameter    # Diamètre du graphe des clusters
```

#### Extraction des Informations

Les métriques sont calculées **directement depuis les attributs du super-graphe** :

```python
# Chaque nœud du super-graphe a ces attributs :
super_graph.nodes[cluster_id] = {
    'cluster_size': 10,        # Nombre de nœuds dans ce cluster
    'internal_edges': 25,      # Nombre d'arêtes internes
    'nodes': [0, 1, 2, ...]   # Liste des nœuds originaux
}

# Les arêtes inter-cluster ont un poids :
super_graph[cluster_A][cluster_B] = {
    'weight': 5  # 5 arêtes reliant cluster_A et cluster_B
}

# Les arêtes intra-cluster sont des self-loops :
super_graph[cluster_A][cluster_A] = {
    'weight': 25  # 25 arêtes internes au cluster_A
}
```

---

### 2. Dictionnaire de Définitions (METRIC_DEFINITIONS)

Un **dictionnaire complet** avec 23+ métriques :

```python
METRIC_DEFINITIONS = {
    'density': {
        'name': 'Densité',
        'definition': "Proportion d'arêtes existantes par rapport au maximum possible",
        'formula': "D = 2m / (n(n-1))",
        'interpretation': "0 = vide, 1 = complet, ~0.1 = épars, ~0.5 = dense"
    },
    'avg_degree': {
        'name': 'Degré Moyen (S_AD)',
        'definition': "Nombre moyen de voisins par nœud",
        'formula': "d_avg = (1/n) × Σ deg(v)",
        'interpretation': "Mesure la connectivité moyenne du graphe"
    },
    # ... 21 autres métriques
}
```

#### Catégories de Métriques

**Métriques de Base**
- `num_nodes`, `num_edges`, `density`

**Groupe 1 : Degree-based (S_AD, S_MD, S_DV, S_PL)**
- `avg_degree`, `max_degree`, `degree_variance`, `power_law_exponent`

**Groupe 2 : Shortest path-based (S_APD, S_EDiam, S_CL, S_Diam)**
- `diameter`, `avg_shortest_path`, `effective_diameter`, `connectivity_length`

**Groupe 3 : Clustering (S_CC)**
- `clustering_coefficient`, `avg_clustering`

**Métriques de Préservation**
- `degree_correlation`

**Métriques Super-Graphe**
- `num_clusters`, `min_cluster_size`, `avg_cluster_size`
- `intra_cluster_edges`, `inter_cluster_edges`, `intra_ratio`
- `information_loss`, `edge_preservation_ratio`, `super_graph_density`

---

### 3. Tooltips Interactifs

#### Fonction : `get_metric_tooltip(metric_key)`

Génère un tooltip formaté pour chaque métrique :

```python
def get_metric_tooltip(metric_key):
    if metric_key not in METRIC_DEFINITIONS:
        return None

    info = METRIC_DEFINITIONS[metric_key]

    tooltip = (
        f"📖 **Définition**: {info['definition']}\n\n"
        f"📐 **Formule**: {info['formula']}\n\n"
        f"💡 **Interprétation**: {info['interpretation']}"
    )

    return tooltip
```

#### Utilisation dans l'UI

Toutes les métriques ont maintenant un **ℹ️ cliquable** :

```python
st.metric("Densité",
         f"{anon_density:.3f}",
         delta=f"{delta_density:+.3f}",
         help=get_metric_tooltip('density'))  # ← Tooltip interactif
```

**Résultat** : L'utilisateur passe sa souris sur le ℹ️ et voit :

```
📖 Définition: Proportion d'arêtes existantes par rapport au maximum possible

📐 Formule: D = 2m / (n(n-1))

💡 Interprétation: 0 = vide, 1 = complet, ~0.1 = épars, ~0.5 = dense
```

---

### 4. Détection Automatique du Type de Graphe

La fonction `calculate_utility_metrics()` détecte automatiquement le type :

```python
def calculate_utility_metrics(G_orig, G_anon):
    # CAS 1 : Vérifier si c'est un super-graphe
    is_super_graph = False
    if G_anon.number_of_nodes() > 0:
        first_node = list(G_anon.nodes())[0]
        node_data = G_anon.nodes[first_node]
        is_super_graph = 'cluster_size' in node_data  # Attribut spécifique

    if is_super_graph:
        return calculate_supergraph_metrics(G_orig, G_anon)

    # CAS 2 : Vérifier si c'est un graphe probabiliste
    is_probabilistic = False
    if G_anon.number_of_edges() > 0:
        first_edge = list(G_anon.edges())[0]
        is_probabilistic = 'probability' in G_anon[first_edge[0]][first_edge[1]]

    if is_probabilistic:
        G_sample = sample_from_probabilistic_graph(G_anon)
        metrics['is_sample'] = True
        # ... calculer sur l'échantillon

    # CAS 3 : Graphe classique
    # ... calculer normalement
```

---

### 5. Interface Utilisateur Adaptative

#### Tab 3 : Métriques d'Utilité

L'affichage **s'adapte automatiquement** au type de graphe :

**CAS 1 : SUPER-GRAPHE (Généralisation)**

```python
if utility_metrics.get('is_super_graph', False):
    st.info("🔍 Type de graphe : Super-Graphe (Généralisation)")

    # Section 1 : Métriques de Clustering
    st.markdown("### 🏘️ Métriques de Clustering")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de Clusters",
                 utility_metrics.get('num_clusters'),
                 help=get_metric_tooltip('num_clusters'))
    # ...

    # Section 2 : Métriques d'Arêtes
    st.markdown("### 🔗 Métriques d'Arêtes")
    # Intra-cluster, Inter-cluster, Ratio

    # Section 3 : Perte d'Information
    st.markdown("### 📊 Perte d'Information")
    # Information loss, Edge preservation, Density

    # Section 4 : Comparaison Original ↔ Anonymisé
    st.markdown("### 📉 Comparaison Original ↔ Anonymisé")
    # Tableau côte à côte
```

**CAS 2 : GRAPHE PROBABILISTE**

```python
elif utility_metrics.get('comparable', True):
    if utility_metrics.get('is_sample', False):
        st.info("🎲 Type de graphe : Échantillon tiré depuis un graphe probabiliste")

    # Affichage standard avec tooltips
    st.markdown("### 📊 Métriques de Base")
    # Nœuds, Arêtes, Densité, Clustering

    st.markdown("### 🌐 Métriques Globales")
    # Diamètre, Chemin Moyen, Corrélation
```

**CAS 3 : GRAPHE CLASSIQUE**

```python
else:
    # Même affichage que probabiliste mais sans indicateur d'échantillon
```

---

## 🧪 Validation

### Fichier de Test : `test_all_metric_types.py`

Ce test valide les **3 types de graphes** :

```python
# TEST 1 : Graphe classique (Random Switch k=10)
G_random = anonymizer.random_switch(k=10)
metrics_random = calculate_utility_metrics(G, G_random)
assert metrics_random.get('comparable') == True
assert metrics_random.get('is_super_graph') == False

# TEST 2 : Graphe probabiliste ((k,ε)-obfuscation)
G_prob = anonymizer.probabilistic_obfuscation(k=5, epsilon=0.5)
metrics_prob = calculate_utility_metrics(G, G_prob)
assert metrics_prob.get('is_sample') == True
assert metrics_prob.get('comparable') == True

# TEST 3 : Super-graphe (Généralisation k=5)
G_super, node_to_cluster = anonymizer.generalization(k=5)
metrics_super = calculate_utility_metrics(G, G_super)
assert metrics_super.get('is_super_graph') == True
assert metrics_super.get('num_clusters') is not None
```

### Résultats des Tests

```
TEST 1 : GRAPHE CLASSIQUE (Random Switch)
  Type detecte : graphe classique
  Comparable : True
  Est un super-graphe : False
  Metriques calculees : 34 noeuds, 78 aretes, densite 0.139
  [OK] Graphe classique detecte et metriques calculees

TEST 2 : GRAPHE PROBABILISTE ((k,epsilon)-obfuscation)
  Type detecte : graphe classique
  Est un echantillon probabiliste : True
  Metriques calculees : 34 noeuds, 81 aretes, densite 0.144
  [OK] Graphe probabiliste detecte et echantillonnage effectue

TEST 3 : SUPER-GRAPHE (Generalisation)
  Type detecte : super-graph
  Est un super-graphe : True
  Metriques calculees :
    - Nombre de clusters : 4
    - Taille min/moy/max : 8 / 8.5 / 10
    - Aretes intra : 36 (46.2%)
    - Aretes inter : 42 (53.8%)
    - Perte d'information : 88.2%
    - Preservation aretes : 100.0%
  [OK] Super-graphe detecte et metriques adaptees calculees

TOOLTIPS : 23 disponibles
  [OK] 23 tooltips disponibles

SUCCES : Tous les tests passes !
```

---

## 📊 Exemple d'Affichage UI

### Super-Graphe (Généralisation k=5)

```
🔍 Type de graphe : Super-Graphe (Généralisation) - Métriques adaptées

🏘️ Métriques de Clustering
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Nb Clusters ℹ️  │ Taille Min ℹ️   │ Taille Moy ℹ️   │ Taille Max      │
│       4         │       8         │      8.5        │      10         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

🔗 Métriques d'Arêtes
┌─────────────────┬─────────────────┬─────────────────┐
│ Intra-Cluster ℹ️│ Inter-Cluster ℹ️│ Ratio Intra ℹ️  │
│      36         │      42         │    46.2%        │
└─────────────────┴─────────────────┴─────────────────┘

📊 Perte d'Information
┌─────────────────┬─────────────────┬─────────────────┐
│ Perte Info ℹ️   │ Préserv. Arêtes │ Densité Super ℹ️│
│    88.2%        │    100.0%       │    1.000        │
└─────────────────┴─────────────────┴─────────────────┘

📉 Comparaison Original ↔ Anonymisé
┌─────────────────────────┬─────────────────────────┐
│    Graphe Original      │      Super-Graphe       │
├─────────────────────────┼─────────────────────────┤
│ Nœuds : 34              │ Clusters : 4            │
│ Arêtes : 78             │ Arêtes Totales : 78     │
└─────────────────────────┴─────────────────────────┘
```

**Tooltip sur "Nb Clusters ℹ️"** :
```
📖 Définition: Nombre de super-nœuds dans le graphe de généralisation

📐 Formule: k = nombre de clusters

💡 Interprétation: Plus faible = plus de privacy, moins d'utilité
```

---

## 🎯 Avantages pour la Présentation

### 1. Clarté Pédagogique

**Avant** :
- "Densité : 0.139" → ❓ Qu'est-ce que ça veut dire ?

**Après** :
- Hover sur ℹ️ → "Proportion d'arêtes existantes / max possible"
- Formule : D = 2m / (n(n-1))
- Interprétation : 0.139 = graphe épars

### 2. Comparaison Méthodes

Vous pouvez maintenant **comparer quantitativement** les méthodes :

| Méthode             | Type           | Perte Info | Préserv. Arêtes | Clustering |
|---------------------|----------------|------------|-----------------|------------|
| Random Switch       | Classique      | 0%         | 100%            | 0.283      |
| (k,ε)-obfuscation   | Probabiliste   | 0%         | ~100%           | 0.250      |
| Généralisation k=5  | Super-graphe   | 88.2%      | 100%            | N/A        |

**Observation** :
- Random Switch : Excellente préservation (corr. = 1.000)
- Probabiliste : Bonne préservation (corr. = 0.969)
- Généralisation : **88% de perte d'information** mais garantie k-anonymity !

### 3. Trade-off Privacy-Utility

Les métriques montrent clairement le **trade-off** :

```
Généralisation k=2 :
  - Clusters : 16 (petits)
  - Perte info : 52.9%
  - Privacy : Faible (k=2)

Généralisation k=10 :
  - Clusters : 2 (gros)
  - Perte info : 94.1%
  - Privacy : Forte (k=10)
```

**Message** : "Plus k augmente, plus on perd d'information, mais plus la privacy est forte"

---

## 🔧 Modifications Techniques

### Fichiers Modifiés

**`graph_anonymization_app.py`**

1. **Ligne 145-300** : `generalization()`
   - Ajouté attributs `cluster_size` et `internal_edges` aux nœuds
   - Incrémente `internal_edges` pour chaque arête intra-cluster

2. **Ligne 1809-1921** : `calculate_supergraph_metrics()` (nouveau)
   - Calcule métriques spécifiques aux super-graphes

3. **Ligne 1924-2140** : `calculate_utility_metrics()`
   - Détection automatique du type de graphe
   - Appel à `calculate_supergraph_metrics()` si super-graphe

4. **Ligne 2196-2374** : `METRIC_DEFINITIONS` et `get_metric_tooltip()`
   - Dictionnaire avec 23 définitions
   - Fonction helper pour générer les tooltips

5. **Ligne 2729-2941** : Tab 3 (Métriques d'Utilité)
   - Affichage adaptatif selon le type
   - Tous les st.metric() ont `help=get_metric_tooltip(...)`

### Fichiers Créés

**`test_all_metric_types.py`** (nouveau)
- Teste les 3 types de graphes
- Valide la détection automatique
- Vérifie les tooltips

---

## 📚 Références

**Thèse** : "Anonymizing Social Graphs via Uncertainty Semantics" - NGUYEN Huu-Hiep, 2016
- **Section 3.2** : Généralisation et k-anonymity structurelle
- **Section 3.5.2** : Métriques d'utilité (S_NE, S_AD, S_MD, etc.)

**Littérature** :
- Liu & Terzi (2008) : "Towards Identity Anonymization on Graphs" (k-anonymity)
- Hay et al. (2008) : "Resisting Structural Re-identification in Anonymized Social Networks" (généralisation)

---

## ✅ Checklist de Validation

- [x] Super-graphe détecté automatiquement via attribut `cluster_size`
- [x] Métriques spécifiques calculées depuis les attributs du graphe
- [x] 23+ tooltips disponibles avec définition + formule + interprétation
- [x] Affichage UI adapté aux 3 types de graphes
- [x] Tests complets sur Karate Club (34 nœuds, 78 arêtes)
- [x] Documentation complète (ce fichier)
- [x] Commit et push sur GitHub

---

## 🚀 Utilisation dans la Présentation

### Diapositive "Évaluation de l'Utilité"

**Slide 1 : Métriques Classiques**
- Montrer Random Switch avec tooltips
- Expliquer : "Densité = proportion arêtes existantes / max"
- Montrer corrélation = 1.000 → "Parfaite préservation"

**Slide 2 : Métriques Probabilistes**
- Montrer (k,ε)-obfuscation avec indicateur "Échantillon"
- Expliquer : "Métriques calculées sur un tirage au sort"
- Montrer corrélation = 0.969 → "Excellente préservation"

**Slide 3 : Métriques Super-Graphe**
- Montrer Généralisation avec métriques adaptées
- Expliquer : "88% de perte d'information mais garantie k-anonymity"
- Montrer ratio intra/total → "46% arêtes locales préservées"

**Message final** :
> "Les tooltips permettent de comprendre chaque métrique.
> Les métriques adaptées montrent le vrai trade-off privacy-utility.
> Pour la généralisation, on perd de l'information (88%) mais on gagne de la privacy (k-anonymity)."

---

**Date de création** : 2025-12-06
**Version** : 1.0
**Auteur** : Claude Code (avec supervision humaine)
