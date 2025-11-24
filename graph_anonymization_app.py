"""
Application Interactive d'Anonymisation de Graphes Sociaux
Basée sur la thèse "Anonymisation de Graphes Sociaux" par NGUYEN Huu-Hiep

Application Streamlit avec sélection de méthodes et explications détaillées
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random
from copy import deepcopy
import io
from method_details import ATTACKS_AND_GUARANTEES

# Configuration de la page
st.set_page_config(
    page_title="Anonymisation de Graphes Sociaux",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .method-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
    .math-formula {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)


class GraphAnonymizer:
    """Classe pour anonymiser des graphes sociaux - VERSION ÉQUILIBRÉE"""

    def __init__(self, graph):
        self.original_graph = graph.copy()
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()

    def random_add_del(self, k=20):
        """Random Add/Del optimisé pour effet visible"""
        G = self.original_graph.copy()
        added = 0
        attempts = 0
        max_attempts = k * 100

        while added < k and attempts < max_attempts:
            u, v = random.sample(list(G.nodes()), 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1
            attempts += 1

        edges = list(self.original_graph.edges())
        if len(edges) >= k:
            edges_to_remove = random.sample(edges, min(k, len(edges)))
            for u, v in edges_to_remove:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)

        return G

    def random_switch(self, k=25):
        """Random Switch optimisé"""
        G = self.original_graph.copy()
        successful_switches = 0

        for _ in range(k * 3):  # Plus de tentatives
            if successful_switches >= k:
                break

            edges = list(G.edges())
            if len(edges) < 2:
                break

            (u, w), (v, x) = random.sample(edges, 2)

            if u != v and u != x and w != v and w != x:
                if not G.has_edge(u, v) and not G.has_edge(w, x):
                    G.remove_edge(u, w)
                    G.remove_edge(v, x)
                    G.add_edge(u, v)
                    G.add_edge(w, x)
                    successful_switches += 1

        return G

    def k_degree_anonymity(self, k=2):
        """k-degree anonymity avec k=2 pour effet visible"""
        G = self.original_graph.copy()
        degrees = dict(G.degree())
        degree_counts = Counter(degrees.values())

        modifications = 0
        for degree, count in sorted(degree_counts.items()):
            if count < k:
                nodes_with_degree = [n for n, d in degrees.items() if d == degree]

                for node in nodes_with_degree:
                    while degrees[node] < degree + 1 and modifications < 30:
                        candidates = [n for n in G.nodes()
                                    if n != node and not G.has_edge(node, n)]
                        if candidates:
                            target = random.choice(candidates)
                            G.add_edge(node, target)
                            degrees[node] += 1
                            degrees[target] += 1
                            modifications += 1

        return G

    def generalization(self, k=4):
        """Généralisation avec k=4 pour clusters moyens"""
        G = self.original_graph.copy()

        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except:
            # Fallback simple
            communities = [set(G.nodes())]

        super_graph = nx.Graph()
        node_to_cluster = {}
        cluster_to_nodes = {}
        cluster_id = 0

        for community in communities:
            community = set(community)

            for node in community:
                node_to_cluster[node] = cluster_id

            super_graph.add_node(cluster_id, size=len(community), nodes=list(community))
            cluster_to_nodes[cluster_id] = list(community)
            cluster_id += 1

        # Compter les arêtes intra et inter-clusters
        intra_edges = 0
        inter_edges = 0

        # Ajouter les super-arêtes
        for u, v in G.edges():
            cluster_u = node_to_cluster.get(u)
            cluster_v = node_to_cluster.get(v)

            if cluster_u is not None and cluster_v is not None:
                if cluster_u != cluster_v:
                    inter_edges += 1
                    if super_graph.has_edge(cluster_u, cluster_v):
                        super_graph[cluster_u][cluster_v]['weight'] += 1
                    else:
                        super_graph.add_edge(cluster_u, cluster_v, weight=1)
                else:
                    # Self-loops pour les arêtes internes
                    intra_edges += 1
                    if super_graph.has_edge(cluster_u, cluster_u):
                        super_graph[cluster_u][cluster_u]['weight'] += 1
                    else:
                        super_graph.add_edge(cluster_u, cluster_u, weight=1)

        # Stocker les statistiques
        super_graph.graph['intra_edges'] = intra_edges
        super_graph.graph['inter_edges'] = inter_edges
        super_graph.graph['cluster_to_nodes'] = cluster_to_nodes
        super_graph.graph['node_to_cluster'] = node_to_cluster

        return super_graph, node_to_cluster

    def probabilistic_obfuscation(self, k=5, epsilon=0.3):
        """(k,ε)-obfuscation optimisé"""
        G = self.original_graph.copy()
        prob_graph = nx.Graph()
        prob_graph.add_nodes_from(G.nodes())

        # Arêtes existantes avec haute probabilité
        for u, v in G.edges():
            prob_graph.add_edge(u, v, probability=1.0 - epsilon/k, is_original=True)

        # Ajouter des arêtes potentielles
        non_edges = [(u, v) for u in G.nodes() for v in G.nodes()
                     if u < v and not G.has_edge(u, v)]

        # Ajouter ~30% des non-arêtes
        num_to_add = int(len(non_edges) * 0.3)
        edges_to_add = random.sample(non_edges, min(num_to_add, len(non_edges)))

        for u, v in edges_to_add:
            prob = epsilon / (2 * k)
            prob_graph.add_edge(u, v, probability=prob, is_original=False)

        return prob_graph

    def differential_privacy_edgeflip(self, epsilon=0.8):
        """EdgeFlip avec epsilon=0.8 pour effet visible"""
        G = nx.Graph()
        G.add_nodes_from(self.original_graph.nodes())

        s = 1 - np.exp(-epsilon)

        for u in self.original_graph.nodes():
            for v in self.original_graph.nodes():
                if u < v:
                    exists = self.original_graph.has_edge(u, v)

                    if random.random() < s/2:
                        if not exists:
                            G.add_edge(u, v)
                    else:
                        if exists:
                            G.add_edge(u, v)

        return G

    def differential_privacy_laplace(self, epsilon=1.2):
        """Mécanisme de Laplace optimisé"""
        G = self.original_graph.copy()
        sensitivity = 1
        scale = sensitivity / epsilon

        new_graph = nx.Graph()
        new_graph.add_nodes_from(G.nodes())

        for u in G.nodes():
            for v in G.nodes():
                if u < v:
                    true_value = 1 if G.has_edge(u, v) else 0
                    noise = np.random.laplace(0, scale)
                    noisy_value = true_value + noise

                    if noisy_value > 0.5:
                        new_graph.add_edge(u, v)

        return new_graph


# Définitions des méthodes avec explications
METHODS = {
    "Random Add/Del": {
        "name": "Randomisation - Random Add/Del",
        "category": "1. Anonymisation par Randomisation",
        "params": {"k": 20},
        "description_short": "Ajoute k fausses arêtes puis supprime k vraies arêtes aléatoirement",
        "description": """
### Principe en Langage Naturel

La méthode **Random Add/Del** est l'une des plus simples. Elle fonctionne en deux étapes :
1. **Ajout** : On ajoute k arêtes aléatoires entre des nœuds non connectés
2. **Suppression** : On supprime k arêtes existantes choisies au hasard

Cette approche crée de l'incertitude en modifiant la structure du graphe de manière aléatoire.
Un attaquant qui connaîtrait le degré d'un nœud ne pourra plus le retrouver avec certitude
car les degrés ont changé.

### Formalisation Mathématique

Soit G = (V, E) le graphe original.

**Algorithme** :
```
1. Pour i = 1 à k :
   - Choisir (u, v) ∈ V × V tel que (u,v) ∉ E
   - E ← E ∪ {(u,v)}

2. Pour i = 1 à k :
   - Choisir (u, v) ∈ E uniformément
   - E ← E \\ {(u,v)}

3. Retourner G' = (V, E)
```

**Propriétés** :
- Nombre d'arêtes préservé : |E'| = |E|
- Distribution des degrés modifiée
- Pas de garantie formelle de privacy

**Complexité** : O(k)
        """,
        "formula": r"P(edge_{added}) = \frac{k}{|V|(|V|-1)/2 - |E|}, \quad P(edge_{removed}) = \frac{k}{|E|}",
        "privacy_level": "Faible (pas de garantie formelle)",
        "utility_preservation": "Moyenne à Bonne"
    },

    "Random Switch": {
        "name": "Randomisation - Random Switch",
        "category": "1. Anonymisation par Randomisation",
        "params": {"k": 25},
        "description_short": "Échange k paires d'arêtes en préservant les degrés",
        "description": """
### Principe en Langage Naturel

**Random Switch** améliore Random Add/Del en préservant une propriété importante : **les degrés des nœuds**.

Au lieu d'ajouter/supprimer des arêtes indépendamment, on **échange** des arêtes :
- On choisit deux arêtes (u,w) et (v,x)
- On les remplace par (u,v) et (w,x)
- Si ces nouvelles arêtes n'existent pas déjà

Ainsi, chaque nœud conserve exactement le même nombre de connexions, mais ces connexions
pointent vers d'autres nœuds. C'est comme si on "réarrangeait" les liens sociaux sans
changer le nombre d'amis de chacun.

### Formalisation Mathématique

**Algorithme** :
```
Pour i = 1 à k :
  1. Choisir (u,w), (v,x) ∈ E uniformément
  2. Si u ≠ v ≠ w ≠ x et (u,v) ∉ E et (w,x) ∉ E :
     - E ← E \\ {(u,w), (v,x)}
     - E ← E ∪ {(u,v), (w,x)}

Retourner G' = (V, E)
```

**Invariants préservés** :
- Séquence de degrés : deg_G'(v) = deg_G(v) ∀v ∈ V
- Nombre d'arêtes : |E'| = |E|

**Propriété clé** : Les chemins et la structure globale sont modifiés tout en
préservant les propriétés locales (degrés).

**Complexité** : O(k)
        """,
        "formula": r"deg_{G'}(v) = deg_G(v) \quad \forall v \in V",
        "privacy_level": "Faible à Moyenne",
        "utility_preservation": "Très Bonne (degrés préservés)"
    },

    "k-degree anonymity": {
        "name": "K-Anonymisation - k-degree anonymity",
        "category": "2. K-Anonymisation",
        "params": {"k": 2},
        "description_short": "Garantit que chaque degré apparaît au moins k fois",
        "description": """
### Principe en Langage Naturel

La **k-degree anonymity** fournit une garantie formelle : chaque nœud doit être
**indistinguable** d'au moins k-1 autres nœuds en termes de degré.

**Intuition** : Si un attaquant connaît le degré d'un nœud cible (ex: 5 amis),
il doit y avoir au moins k nœuds avec ce même degré. L'attaquant ne peut donc
identifier le nœud cible qu'avec une probabilité ≤ 1/k.

**Exemple** : Avec k=3, si Alice a 7 amis, on s'assure qu'au moins 2 autres
personnes ont aussi 7 amis. L'attaquant ne peut pas dire laquelle est Alice.

L'algorithme ajoute des arêtes de manière **déterministe** pour atteindre cette propriété.

### Formalisation Mathématique

**Définition formelle** :

Un graphe G = (V, E) satisfait la k-degree anonymity si :

∀d ∈ {deg(v) : v ∈ V}, |{v ∈ V : deg(v) = d}| ≥ k

C'est-à-dire : pour tout degré d qui apparaît dans le graphe,
il doit y avoir au moins k nœuds avec ce degré.

**Algorithme** :
```
Entrée : G = (V, E), k
Sortie : G' = (V, E') satisfaisant k-degree anonymity

1. Calculer la séquence de degrés D = [deg(v) : v ∈ V]
2. Pour chaque degré d apparaissant moins de k fois :
   - Identifier les nœuds V_d = {v : deg(v) = d}
   - Ajouter des arêtes pour augmenter/uniformiser les degrés
3. Retourner G'
```

**Garantie de privacy** :

P(identité de v | deg(v) = d) ≤ 1/k

**NP-complétude** : Trouver le nombre minimum d'arêtes à ajouter est NP-difficile.

**Complexité** : O(n²) (avec heuristiques)
        """,
        "formula": r"|\{v \in V : deg(v) = d\}| \geq k \quad \forall d",
        "privacy_level": "Moyenne à Forte (garantie k-anonymity)",
        "utility_preservation": "Bonne"
    },

    "Generalization": {
        "name": "Généralisation - Super-nodes",
        "category": "3. Anonymisation par Généralisation",
        "params": {"k": 4},
        "description_short": "Regroupe les nœuds en super-nœuds de taille ≥ k",
        "description": """
### Principe en Langage Naturel

La **généralisation** adopte une approche radicalement différente : au lieu de modifier
les arêtes, on **regroupe** les nœuds similaires en "super-nœuds".

**Analogie** : C'est comme publier des statistiques par département plutôt que par personne.
- Au lieu de "Alice (Paris) connectée à Bob (Lyon)"
- On dit "Région Île-de-France (10000 personnes) connectée à Région Auvergne-Rhône-Alpes (5000 personnes)"

**Avantages** :
- Protection maximale de l'identité individuelle
- Réduction de la taille du graphe publié
- Chaque individu est "caché" dans un groupe de k personnes minimum

**Inconvénient** : Perte importante d'information structurelle fine.

### Formalisation Mathématique

**Modèle de graphe généralisé** :

Soit G = (V, E) le graphe original. On crée une partition P = {C₁, C₂, ..., Cₘ}
de V telle que |Cᵢ| ≥ k ∀i.

Le **super-graphe** G* = (V*, E*) est défini par :
- V* = {C₁, C₂, ..., Cₘ} (les clusters)
- E* = {(Cᵢ, Cⱼ) : ∃(u,v) ∈ E avec u ∈ Cᵢ, v ∈ Cⱼ}

Chaque super-arête (Cᵢ, Cⱼ) a un **poids** :

w(Cᵢ, Cⱼ) = |{(u,v) ∈ E : u ∈ Cᵢ, v ∈ Cⱼ}|

**Probabilité d'arête dans le cluster** :

P(edge | Cᵢ, Cⱼ) = w(Cᵢ, Cⱼ) / (|Cᵢ| × |Cⱼ|)

**Garantie de privacy** : Un individu est caché parmi au moins k-1 autres
dans son cluster.

**Problème d'optimisation** : Trouver la partition P qui minimise la perte
d'information tout en respectant |Cᵢ| ≥ k est NP-difficile.

**Complexité** : O(n²) à O(n³) selon l'algorithme de clustering
        """,
        "formula": r"G^* = (V^*, E^*) \text{ où } V^* = \{C_i : |C_i| \geq k\}",
        "privacy_level": "Forte (k-anonymity structurelle)",
        "utility_preservation": "Faible à Moyenne"
    },

    "Probabilistic": {
        "name": "Probabiliste - (k,ε)-obfuscation",
        "category": "4. Approches Probabilistes",
        "params": {"k": 5, "epsilon": 0.3},
        "description_short": "Crée un graphe incertain avec probabilités sur les arêtes",
        "description": """
### Principe en Langage Naturel

Les approches **probabilistes** créent un "graphe incertain" où chaque arête existe
avec une certaine **probabilité**.

**Idée clé** : Au lieu de publier un graphe déterministe (arête = oui/non), on publie
des probabilités. Par exemple :
- Arête (Alice, Bob) : 95% de probabilité d'exister
- Arête (Alice, Charlie) : 20% de probabilité d'exister

Un attaquant ne peut plus être certain de rien : même les vraies arêtes ont une incertitude.

**Modèle (k,ε)-obfuscation** :
- **k** : niveau d'anonymisation souhaité (plus k est grand, plus de protection)
- **ε** : paramètre de tolérance (plus ε est petit, plus de protection)

### Formalisation Mathématique

**Graphe incertain** :

Un graphe incertain est un triplet G̃ = (V, E, p) où :
- V : ensemble de nœuds
- E : ensemble d'arêtes (réelles + potentielles)
- p : E → [0,1] fonction de probabilité

**Définition (k,ε)-obfuscation** :

Pour tout nœud v ∈ V, l'entropie de Shannon de la distribution
de probabilité sur les k voisins candidats doit être ≥ log(k) - ε :

H(N_k(v)) = -∑ᵢ p_i log(p_i) ≥ log(k) - ε

où N_k(v) sont les k nœuds les plus susceptibles d'être voisins de v.

**Assignation des probabilités** :

Pour les arêtes existantes :
p((u,v)) = 1 - ε/k

Pour les arêtes potentielles (ajoutées pour l'obfuscation) :
p((u,v)) = ε/(2k)

**Graphe d'exemple (sample graph)** :

À partir de G̃, on peut générer des graphes compatibles en échantillonnant :

G_sample = (V, E_sample) où e ∈ E_sample ssi X_e ≤ p(e), X_e ~ U[0,1]

**Propriété** : L'espérance des degrés est préservée.

**Complexité** : O(|E| + k·n)
        """,
        "formula": r"H(N_k(v)) = -\sum_i p_i \log(p_i) \geq \log(k) - \varepsilon",
        "privacy_level": "Moyenne à Forte (contrôle via k et ε)",
        "utility_preservation": "Bonne (espérance préservée)"
    },

    "EdgeFlip": {
        "name": "Privacy Différentielle - EdgeFlip",
        "category": "5. Privacy Différentielle",
        "params": {"epsilon": 0.8},
        "description_short": "Applique le Randomized Response Technique avec ε-DP",
        "description": """
### Principe en Langage Naturel

**EdgeFlip** applique le célèbre **Randomized Response Technique** (RRT) des statistiques
à la publication de graphes.

**Intuition du RRT** (exemple classique) :
Pour une question sensible ("Avez-vous triché à l'examen ?") :
- Lancez une pièce en secret
- Si Face : répondez la vérité
- Si Pile : répondez au hasard (oui/non à pile ou face)

Résultat : Votre réponse a du "déni plausible" mais les statistiques globales
restent calculables.

**Application à EdgeFlip** :
Pour chaque paire de nœuds (u,v) :
- Avec probabilité s/2 : **inverser** l'arête (0→1 ou 1→0)
- Avec probabilité 1-s/2 : garder l'état réel

où s est déterminé par le paramètre de privacy ε.

**Garantie ε-differential privacy** : La présence/absence d'une arête
est protégée avec garantie mathématique ε-DP.

### Formalisation Mathématique

**Définition ε-Differential Privacy** :

Un algorithme A satisfait ε-DP si pour tous graphes voisins G, G'
(différant par une arête) et pour tout output O :

P[A(G) = O] ≤ e^ε · P[A(G') = O]

Plus ε est petit, plus forte est la garantie de privacy.

**Algorithme EdgeFlip** :

```
Entrée : G = (V, E), ε
Paramètre : s = 1 - e^(-ε)

Pour chaque paire (u, v) avec u < v :
  exists = (u,v) ∈ E

  Avec probabilité s/2 :
    output = NOT exists   // Inverser
  Sinon :
    output = exists       // Garder

  Si output = TRUE :
    Ajouter (u,v) à E_output

Retourner G_output = (V, E_output)
```

**Preuve de ε-DP** :

Pour une arête (u,v) :

P[output=1 | exists=1] = 1 - s/2
P[output=1 | exists=0] = s/2

Ratio : (1 - s/2) / (s/2) = e^ε

Donc EdgeFlip satisfait ε-edge-DP.

**Espérance du nombre d'arêtes** :

E[|E_output|] = |E| · (1 - s/2) + (n(n-1)/2 - |E|) · s/2
              ≈ n(n-1)/4  (pour s ≈ 1, très bruité)

**Complexité** : O(n²)

**Inconvénient** : Complexité quadratique limite le passage à l'échelle.
        """,
        "formula": r"P[\mathcal{A}(G) = O] \leq e^\varepsilon \cdot P[\mathcal{A}(G') = O]",
        "privacy_level": "Très Forte (ε-differential privacy)",
        "utility_preservation": "Variable (dépend de ε)"
    },

    "Laplace": {
        "name": "Privacy Différentielle - Mécanisme de Laplace",
        "category": "5. Privacy Différentielle",
        "params": {"epsilon": 1.2},
        "description_short": "Ajoute du bruit Laplacien pour décider de l'inclusion des arêtes",
        "description": """
### Principe en Langage Naturel

Le **Mécanisme de Laplace** est la technique fondamentale de la privacy différentielle.

**Principe général** : Pour publier une statistique f(données) de manière privée,
on ajoute du **bruit aléatoire** calibré à la **sensibilité** de f.

**Pour les graphes** :
- On considère chaque arête potentielle (u,v)
- Valeur réelle : 1 si l'arête existe, 0 sinon
- On ajoute du bruit Laplacien ~ Lap(Δf/ε)
- On décide d'inclure l'arête si valeur_bruitée > seuil

**Intuition du bruit** : Le bruit "masque" la contribution d'une arête individuelle,
rendant impossible de déterminer si une arête spécifique était présente ou non.

### Formalisation Mathématique

**Mécanisme de Laplace général** :

Pour une fonction f : D → ℝ^d, le mécanisme de Laplace est :

M(D) = f(D) + (Y₁, ..., Y_d)

où Y_i ~ Lap(Δf/ε) sont indépendants et Δf est la sensibilité globale.

**Sensibilité globale** :

Δf = max_{G,G' voisins} ||f(G) - f(G')||₁

Pour les graphes (edge-DP), deux graphes sont voisins s'ils diffèrent par une arête.
Donc : Δf = 1 pour une requête de type "cette arête existe-t-elle ?"

**Distribution de Laplace** :

Lap(b) a la densité : p(x|b) = (1/2b) · exp(-|x|/b)
- Moyenne : 0
- Variance : 2b²
- Plus b est grand, plus le bruit est important

**Application aux graphes** :

```
Entrée : G = (V, E), ε
Scale : b = 1/ε

Pour chaque paire (u, v) avec u < v :
  true_value = 1 si (u,v) ∈ E, 0 sinon
  noise = Laplace(0, b)
  noisy_value = true_value + noise

  Si noisy_value > 0.5 :
    Ajouter (u,v) à E_output

Retourner G_output = (V, E_output)
```

**Théorème** : Ce mécanisme satisfait ε-differential privacy.

**Preuve (sketch)** :
Pour G et G' différant par une arête (u₀, v₀) :

P[M(G) = O] / P[M(G') = O] = exp(-ε·|f(G)-f(G')|) ≤ e^ε

car |f(G) - f(G')| ≤ Δf = 1.

**Trade-off ε** :
- ε petit (ex: 0.1) : forte privacy, beaucoup de bruit, faible utilité
- ε grand (ex: 10) : faible privacy, peu de bruit, forte utilité
- Valeurs typiques : ε ∈ [0.1, 10]

**Complexité** : O(n²)
        """,
        "formula": r"M(D) = f(D) + \text{Lap}(\Delta f / \varepsilon)",
        "privacy_level": "Très Forte (ε-differential privacy)",
        "utility_preservation": "Variable (dépend de ε)"
    }
}


def calculate_anonymization_metrics(G_orig, G_anon):
    """Calcule des métriques d'anonymisation détaillées"""
    metrics = {}

    # Changements dans les arêtes
    if isinstance(G_anon, nx.Graph):
        orig_edges = set(G_orig.edges())
        anon_edges = set(G_anon.edges())

        added = len(anon_edges - orig_edges)
        removed = len(orig_edges - anon_edges)
        preserved = len(orig_edges & anon_edges)

        metrics['edges_added'] = added
        metrics['edges_removed'] = removed
        metrics['edges_preserved'] = preserved
        metrics['modification_rate'] = (added + removed) / (2 * len(orig_edges)) if len(orig_edges) > 0 else 0

        # Changements dans les degrés
        deg_orig = dict(G_orig.degree())
        deg_anon = dict(G_anon.degree())

        if set(deg_orig.keys()) == set(deg_anon.keys()):
            deg_changes = sum(abs(deg_orig[v] - deg_anon[v]) for v in deg_orig.keys())
            metrics['total_degree_change'] = deg_changes
            metrics['avg_degree_change'] = deg_changes / len(deg_orig)

            # Incorrectness (combien de nœuds ont changé de degré)
            metrics['nodes_with_degree_change'] = sum(1 for v in deg_orig.keys() if deg_orig[v] != deg_anon[v])
            metrics['degree_preservation_rate'] = 1 - (metrics['nodes_with_degree_change'] / len(deg_orig))

        # Métriques structurelles
        try:
            metrics['clustering_change'] = abs(
                nx.average_clustering(G_orig) - nx.average_clustering(G_anon)
            )
        except:
            metrics['clustering_change'] = None

        metrics['density_change'] = abs(nx.density(G_orig) - nx.density(G_anon))

    return metrics


def calculate_privacy_guarantees(G_orig, G_anon, method_key, method_params):
    """Calcule les garanties de privacy spécifiques à chaque méthode"""
    guarantees = {}

    if method_key == "k-degree anonymity":
        # Vérifier la k-anonymité des degrés
        degrees = dict(G_anon.degree())
        degree_counts = Counter(degrees.values())

        k_value = method_params.get('k', 2)
        min_count = min(degree_counts.values()) if degree_counts else 0
        is_k_anonymous = min_count >= k_value

        guarantees['k_anonymity_satisfied'] = is_k_anonymous
        guarantees['min_degree_count'] = min_count
        guarantees['k_required'] = k_value
        guarantees['re_identification_risk'] = f"≤ 1/{min_count}" if min_count > 0 else "N/A"
        guarantees['unique_degrees'] = len(degree_counts)

    elif method_key == "Generalization":
        # Métriques pour super-nodes
        if hasattr(G_anon, 'graph'):
            cluster_sizes = [G_anon.nodes[n].get('size', 0) for n in G_anon.nodes()]
            min_cluster_size = min(cluster_sizes) if cluster_sizes else 0
            max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
            avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

            intra_edges = G_anon.graph.get('intra_edges', 0)
            inter_edges = G_anon.graph.get('inter_edges', 0)
            total_edges = intra_edges + inter_edges

            guarantees['num_clusters'] = G_anon.number_of_nodes()
            guarantees['min_cluster_size'] = min_cluster_size
            guarantees['max_cluster_size'] = max_cluster_size
            guarantees['avg_cluster_size'] = f"{avg_cluster_size:.1f}"
            guarantees['intra_cluster_edges'] = intra_edges
            guarantees['inter_cluster_edges'] = inter_edges
            guarantees['intra_ratio'] = f"{intra_edges/total_edges*100:.1f}%" if total_edges > 0 else "N/A"
            guarantees['inter_ratio'] = f"{inter_edges/total_edges*100:.1f}%" if total_edges > 0 else "N/A"
            guarantees['re_identification_risk'] = f"≤ 1/{min_cluster_size}" if min_cluster_size > 0 else "N/A"
            guarantees['information_loss'] = f"{(1 - G_anon.number_of_nodes()/G_orig.number_of_nodes())*100:.1f}%"

    elif method_key in ["EdgeFlip", "Laplace"]:
        # Privacy différentielle
        epsilon = method_params.get('epsilon', 1.0)
        guarantees['epsilon'] = epsilon
        guarantees['privacy_budget'] = epsilon
        guarantees['privacy_level'] = "Forte" if epsilon < 1.0 else ("Moyenne" if epsilon < 2.0 else "Faible")
        guarantees['max_privacy_loss'] = f"e^{epsilon:.2f} ≈ {np.exp(epsilon):.2f}"

        # Calculer le taux de faux positifs/négatifs attendu
        if method_key == "EdgeFlip":
            s = 1 - np.exp(-epsilon)
            false_positive_rate = s/2
            false_negative_rate = s/2
            guarantees['expected_false_positive_rate'] = f"{false_positive_rate*100:.1f}%"
            guarantees['expected_false_negative_rate'] = f"{false_negative_rate*100:.1f}%"

    elif method_key == "Probabilistic":
        # (k,ε)-obfuscation
        k = method_params.get('k', 5)
        eps = method_params.get('epsilon', 0.3)

        guarantees['k_neighborhood'] = k
        guarantees['epsilon_tolerance'] = eps
        guarantees['min_entropy'] = f"log({k}) - {eps:.2f} ≈ {np.log(k) - eps:.2f}"
        guarantees['uncertainty_level'] = "Élevée" if eps < 0.5 else "Moyenne"

    elif method_key == "Random Switch":
        # Préservation de la séquence de degrés
        deg_orig = sorted([d for n, d in G_orig.degree()])
        deg_anon = sorted([d for n, d in G_anon.degree()])

        degree_sequence_preserved = deg_orig == deg_anon
        guarantees['degree_sequence_preserved'] = degree_sequence_preserved
        guarantees['structural_property'] = "Séquence de degrés préservée" if degree_sequence_preserved else "Modifiée"

    elif method_key == "Random Add/Del":
        # Quantifier l'incertitude introduite
        k = method_params.get('k', 20)
        total_possible_edges = G_orig.number_of_nodes() * (G_orig.number_of_nodes() - 1) // 2

        guarantees['edges_modified'] = 2 * k  # k ajoutées + k supprimées (théorique)
        guarantees['modification_budget'] = k
        guarantees['structural_uncertainty'] = "Modérée"

    return guarantees


def plot_graph_comparison(G_orig, G_anon, method_name, node_to_cluster=None):
    """Crée une comparaison visuelle des graphes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Position commune pour comparaison
    pos = nx.spring_layout(G_orig, seed=42, k=0.5, iterations=50)

    # Graphe original
    if node_to_cluster is not None:
        # Si c'est une généralisation, colorier par cluster
        clusters = {}
        for node, cluster in node_to_cluster.items():
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(node)

        # Générer des couleurs pour chaque cluster
        import matplotlib.cm as cm
        colors = cm.tab20(np.linspace(0, 1, len(clusters)))

        # Dessiner les nœuds par cluster
        for idx, (cluster_id, nodes) in enumerate(clusters.items()):
            node_color = colors[idx % len(colors)]
            nx.draw_networkx_nodes(G_orig, pos, nodelist=nodes, ax=ax1,
                                  node_color=[node_color], node_size=500, alpha=0.7)

            # Encercler le cluster
            node_positions = np.array([pos[n] for n in nodes])
            if len(node_positions) > 0:
                center = node_positions.mean(axis=0)
                max_dist = np.max(np.linalg.norm(node_positions - center, axis=1))
                circle = plt.Circle(center, max_dist + 0.15, color=node_color,
                                  fill=False, linewidth=2.5, linestyle='--', alpha=0.6)
                ax1.add_patch(circle)

        # Dessiner les arêtes par type
        intra_edges = []
        inter_edges = []
        for u, v in G_orig.edges():
            if node_to_cluster[u] == node_to_cluster[v]:
                intra_edges.append((u, v))
            else:
                inter_edges.append((u, v))

        if intra_edges:
            nx.draw_networkx_edges(G_orig, pos, intra_edges, ax=ax1,
                                  edge_color='green', width=2, alpha=0.4,
                                  label='Intra-cluster')
        if inter_edges:
            nx.draw_networkx_edges(G_orig, pos, inter_edges, ax=ax1,
                                  edge_color='red', width=1.5, alpha=0.5,
                                  label='Inter-cluster', style='dashed')

        ax1.legend(loc='upper right')
        ax1.set_title(f'Graphe Original avec Clusters\n{G_orig.number_of_nodes()} nœuds, {len(clusters)} clusters',
                      fontsize=14, fontweight='bold')
    else:
        # Affichage normal
        nx.draw_networkx_nodes(G_orig, pos, ax=ax1, node_color='lightblue',
                              node_size=500, alpha=0.9)
        nx.draw_networkx_edges(G_orig, pos, ax=ax1, edge_color='gray',
                              width=1.5, alpha=0.6)
        ax1.set_title(f'Graphe Original\n{G_orig.number_of_nodes()} nœuds, {G_orig.number_of_edges()} arêtes',
                      fontsize=14, fontweight='bold')

    nx.draw_networkx_labels(G_orig, pos, ax=ax1, font_size=8, font_weight='bold')
    ax1.axis('off')

    # Graphe anonymisé
    if isinstance(G_anon, nx.Graph) and G_anon.number_of_nodes() > 0:
        # Adapter la position si différent nombre de nœuds
        if set(G_anon.nodes()) != set(G_orig.nodes()):
            pos_anon = nx.spring_layout(G_anon, seed=42, k=0.5, iterations=50)

            # Si c'est un super-graphe, ajuster la visualisation
            if node_to_cluster is not None and hasattr(G_anon, 'graph'):
                # Dessiner le super-graphe
                node_sizes = [G_anon.nodes[n].get('size', 1) * 300 for n in G_anon.nodes()]
                nx.draw_networkx_nodes(G_anon, pos_anon, ax=ax2, node_color='orange',
                                      node_size=node_sizes, alpha=0.8)

                # Dessiner les arêtes avec poids
                edges = G_anon.edges()
                weights = [G_anon[u][v].get('weight', 1) for u, v in edges]
                max_weight = max(weights) if weights else 1

                for (u, v), weight in zip(edges, weights):
                    if u == v:  # Self-loop (arêtes intra-cluster)
                        # Dessiner une boucle
                        continue
                    else:
                        width = 1 + 4 * (weight / max_weight)
                        nx.draw_networkx_edges(G_anon, pos_anon, [(u, v)], ax=ax2,
                                              width=width, alpha=0.6, edge_color='purple')

                # Labels avec taille de cluster
                labels = {n: f"C{n}\n({G_anon.nodes[n].get('size', '?')})" for n in G_anon.nodes()}
                nx.draw_networkx_labels(G_anon, pos_anon, labels, ax=ax2,
                                       font_size=10, font_weight='bold')

                ax2.set_title(f'Super-Graphe - {method_name}\n{G_anon.number_of_nodes()} super-nœuds',
                             fontsize=14, fontweight='bold')
            else:
                # Graphe normal avec nœuds différents
                nx.draw_networkx_nodes(G_anon, pos_anon, ax=ax2, node_color='lightgreen',
                                      node_size=500, alpha=0.9)
                nx.draw_networkx_edges(G_anon, pos_anon, ax=ax2, edge_color='gray',
                                      width=1.5, alpha=0.6)
                nx.draw_networkx_labels(G_anon, pos_anon, ax=ax2, font_size=8, font_weight='bold')
                ax2.set_title(f'Graphe Anonymisé - {method_name}\n{G_anon.number_of_nodes()} nœuds',
                             fontsize=14, fontweight='bold')
        else:
            pos_anon = pos

            # Colorer les arêtes différemment
            orig_edges = set(G_orig.edges())

            # Dessiner les nœuds
            nx.draw_networkx_nodes(G_anon, pos_anon, ax=ax2, node_color='lightgreen',
                                  node_size=500, alpha=0.9)

            # Dessiner les arêtes par type
            preserved_edges = [(u,v) for u,v in G_anon.edges()
                              if (u,v) in orig_edges or (v,u) in orig_edges]
            added_edges = [(u,v) for u,v in G_anon.edges()
                          if (u,v) not in orig_edges and (v,u) not in orig_edges]

            if preserved_edges:
                nx.draw_networkx_edges(G_anon, pos_anon, preserved_edges, ax=ax2,
                                      edge_color='blue', width=1.5, alpha=0.6,
                                      style='solid', label='Arêtes préservées')
            if added_edges:
                nx.draw_networkx_edges(G_anon, pos_anon, added_edges, ax=ax2,
                                      edge_color='red', width=1.5, alpha=0.6,
                                      style='dashed', label='Arêtes ajoutées')

            nx.draw_networkx_labels(G_anon, pos_anon, ax=ax2, font_size=8, font_weight='bold')
            ax2.legend(loc='upper right')

            ax2.set_title(f'Graphe Anonymisé - {method_name}\n{G_anon.number_of_nodes()} nœuds, {G_anon.number_of_edges()} arêtes',
                         fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Graphe non visualisable\n(format incompatible)',
                ha='center', va='center', fontsize=12)
        ax2.set_title(f'Graphe Anonymisé - {method_name}', fontsize=14, fontweight='bold')

    ax2.axis('off')

    plt.tight_layout()
    return fig


def plot_degree_distribution(G_orig, G_anon, method_name):
    """Compare les distributions de degrés"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Distribution originale
    degrees_orig = [d for n, d in G_orig.degree()]
    ax1.hist(degrees_orig, bins=range(max(degrees_orig)+2),
            alpha=0.7, color='blue', edgecolor='black', rwidth=0.8)
    ax1.set_xlabel('Degré', fontsize=12)
    ax1.set_ylabel('Nombre de nœuds', fontsize=12)
    ax1.set_title('Distribution des degrés - Original', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Distribution anonymisée
    if isinstance(G_anon, nx.Graph) and G_anon.number_of_nodes() > 0:
        if set(G_anon.nodes()).issubset(set(G_orig.nodes())):
            degrees_anon = [d for n, d in G_anon.degree()]
            ax2.hist(degrees_anon, bins=range(max(degrees_anon)+2),
                    alpha=0.7, color='green', edgecolor='black', rwidth=0.8)
            ax2.set_xlabel('Degré', fontsize=12)
            ax2.set_ylabel('Nombre de nœuds', fontsize=12)
            ax2.set_title(f'Distribution des degrés - {method_name}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_axisbelow(True)
        else:
            ax2.text(0.5, 0.5, 'Distribution non comparable\n(nœuds différents)',
                    ha='center', va='center', fontsize=12)
            ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, 'Pas de distribution\n(format non standard)',
                ha='center', va='center', fontsize=12)
        ax2.axis('off')

    plt.tight_layout()
    return fig


def main():
    """Application principale Streamlit"""

    # En-tête
    st.markdown('<p class="main-header">🔒 Anonymisation de Graphes Sociaux</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### Application Interactive basée sur la thèse de NGUYEN Huu-Hiep (2016)

    Cette application démontre les **5 types de méthodes d'anonymisation** de graphes sociaux
    avec explications mathématiques détaillées et métriques d'anonymisation.
    """)

    # Sidebar - Sélection de la méthode
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("### 📊 Graphe de Test")
    graph_choice = st.sidebar.selectbox(
        "Choisir un graphe",
        ["Karate Club (34 nœuds)", "Graphe Aléatoire Petit (20 nœuds)",
         "Graphe Aléatoire Moyen (50 nœuds)"]
    )

    # Charger le graphe
    if "Karate" in graph_choice:
        G = nx.karate_club_graph()
        st.sidebar.success(f"✓ Graphe Karate Club chargé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    elif "Petit" in graph_choice:
        G = nx.erdos_renyi_graph(20, 0.15, seed=42)
        st.sidebar.success(f"✓ Graphe aléatoire chargé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    else:
        G = nx.erdos_renyi_graph(50, 0.1, seed=42)
        st.sidebar.success(f"✓ Graphe aléatoire chargé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Méthode d'Anonymisation")

    method_key = st.sidebar.selectbox(
        "Choisir une méthode",
        list(METHODS.keys()),
        format_func=lambda x: METHODS[x]["name"]
    )

    method = METHODS[method_key]

    st.sidebar.markdown(f"**Catégorie** : {method['category']}")
    st.sidebar.markdown(f"**Description** : {method['description_short']}")

    # Bouton pour anonymiser
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Anonymiser le Graphe", type="primary"):
        st.session_state.anonymized = True
        st.session_state.method_key = method_key

        # Anonymiser
        anonymizer = GraphAnonymizer(G)

        with st.spinner('Anonymisation en cours...'):
            node_to_cluster = None
            if method_key == "Random Add/Del":
                G_anon = anonymizer.random_add_del(**method['params'])
            elif method_key == "Random Switch":
                G_anon = anonymizer.random_switch(**method['params'])
            elif method_key == "k-degree anonymity":
                G_anon = anonymizer.k_degree_anonymity(**method['params'])
            elif method_key == "Generalization":
                G_anon, node_to_cluster = anonymizer.generalization(**method['params'])
                st.session_state.node_to_cluster = node_to_cluster
            elif method_key == "Probabilistic":
                G_anon = anonymizer.probabilistic_obfuscation(**method['params'])
            elif method_key == "EdgeFlip":
                G_anon = anonymizer.differential_privacy_edgeflip(**method['params'])
            elif method_key == "Laplace":
                G_anon = anonymizer.differential_privacy_laplace(**method['params'])

            st.session_state.G_anon = G_anon
            st.session_state.G_orig = G
            if node_to_cluster is None:
                st.session_state.node_to_cluster = None

    # Affichage des résultats
    if 'anonymized' in st.session_state and st.session_state.anonymized:
        G_orig = st.session_state.G_orig
        G_anon = st.session_state.G_anon
        current_method = METHODS[st.session_state.method_key]

        # Onglets
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Résultats", "📖 Explications", "📈 Métriques", "🛡️ Attaques & Garanties", "ℹ️ Anonymisation"])

        with tab1:
            st.markdown("## 📊 Résultats de l'Anonymisation")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Nœuds Originaux", G_orig.number_of_nodes())
                st.metric("Arêtes Originales", G_orig.number_of_edges())

            with col2:
                if isinstance(G_anon, nx.Graph):
                    st.metric("Nœuds Anonymisés", G_anon.number_of_nodes())
                    st.metric("Arêtes Anonymisées", G_anon.number_of_edges(),
                             delta=f"{G_anon.number_of_edges() - G_orig.number_of_edges():+d}")
                else:
                    st.info("Format de graphe non standard (super-nodes)")

            st.markdown("---")
            st.markdown("### Comparaison Visuelle")

            node_to_cluster = st.session_state.get('node_to_cluster', None)
            fig = plot_graph_comparison(G_orig, G_anon, current_method['name'], node_to_cluster)
            st.pyplot(fig)

            # Afficher les statistiques spécifiques aux super-nodes
            if st.session_state.method_key == "Generalization" and hasattr(G_anon, 'graph'):
                st.markdown("---")
                st.markdown("### 📊 Statistiques des Super-Nodes")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Arêtes Intra-Cluster", G_anon.graph.get('intra_edges', 'N/A'),
                             help="Arêtes à l'intérieur des clusters (vert)")
                with col2:
                    st.metric("Arêtes Inter-Cluster", G_anon.graph.get('inter_edges', 'N/A'),
                             help="Arêtes entre différents clusters (rouge)")
                with col3:
                    total = G_anon.graph.get('intra_edges', 0) + G_anon.graph.get('inter_edges', 0)
                    ratio = G_anon.graph.get('intra_edges', 0) / total * 100 if total > 0 else 0
                    st.metric("Ratio Intra/Total", f"{ratio:.1f}%")

            st.markdown("---")
            st.markdown("### Distribution des Degrés")

            fig_dist = plot_degree_distribution(G_orig, G_anon, current_method['name'])
            st.pyplot(fig_dist)

        with tab2:
            st.markdown(f"## 📖 Explication Détaillée : {current_method['name']}")

            # Bouton pour afficher/masquer les explications
            if st.button("📚 Afficher l'Explication Complète"):
                st.session_state.show_explanation = not st.session_state.get('show_explanation', False)

            if st.session_state.get('show_explanation', False):
                st.markdown(current_method['description'])

                st.markdown("---")
                st.markdown("### 🔢 Formule Clé")
                st.latex(current_method['formula'])

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔒 Niveau de Privacy")
                    st.info(current_method['privacy_level'])
                with col2:
                    st.markdown("### 📊 Préservation de l'Utilité")
                    st.info(current_method['utility_preservation'])

        with tab3:
            st.markdown("## 📈 Métriques d'Anonymisation Détaillées")

            metrics = calculate_anonymization_metrics(G_orig, G_anon)

            if metrics:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Arêtes Ajoutées", metrics.get('edges_added', 'N/A'))
                with col2:
                    st.metric("Arêtes Supprimées", metrics.get('edges_removed', 'N/A'))
                with col3:
                    st.metric("Arêtes Préservées", metrics.get('edges_preserved', 'N/A'))
                with col4:
                    rate = metrics.get('modification_rate', 0)
                    st.metric("Taux de Modification", f"{rate*100:.1f}%")

                st.markdown("---")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if 'total_degree_change' in metrics:
                        st.metric("Changement Total de Degrés", f"{metrics['total_degree_change']}")

                with col2:
                    if 'nodes_with_degree_change' in metrics:
                        st.metric("Nœuds avec Degré Modifié", metrics['nodes_with_degree_change'])

                with col3:
                    if 'degree_preservation_rate' in metrics:
                        st.metric("Taux de Préservation des Degrés",
                                 f"{metrics['degree_preservation_rate']*100:.1f}%")

                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:
                    if metrics.get('clustering_change') is not None:
                        st.metric("Changement de Clustering",
                                 f"{metrics['clustering_change']:.4f}")

                with col2:
                    st.metric("Changement de Densité",
                             f"{metrics['density_change']:.4f}")

            # Nouvelles métriques de garanties
            st.markdown("---")
            st.markdown("## 🔐 Garanties de Privacy")

            guarantees = calculate_privacy_guarantees(G_orig, G_anon,
                                                     st.session_state.method_key,
                                                     current_method['params'])

            if guarantees:
                if st.session_state.method_key == "k-degree anonymity":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        is_satisfied = guarantees.get('k_anonymity_satisfied', False)
                        st.metric("k-Anonymité Satisfaite", "✅ Oui" if is_satisfied else "❌ Non")
                    with col2:
                        st.metric("Minimum de Nœuds par Degré", guarantees.get('min_degree_count', 'N/A'))
                    with col3:
                        st.metric("Risque de Ré-identification", guarantees.get('re_identification_risk', 'N/A'))

                elif st.session_state.method_key == "Generalization":
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Nombre de Clusters", guarantees.get('num_clusters', 'N/A'))
                    with col2:
                        st.metric("Taille Min/Moy/Max", f"{guarantees.get('min_cluster_size', '?')}/{guarantees.get('avg_cluster_size', '?')}/{guarantees.get('max_cluster_size', '?')}")
                    with col3:
                        st.metric("Risque de Ré-identification", guarantees.get('re_identification_risk', 'N/A'))
                    with col4:
                        st.metric("Perte d'Information", guarantees.get('information_loss', 'N/A'))

                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Arêtes Intra-Cluster", f"{guarantees.get('intra_cluster_edges', 0)} ({guarantees.get('intra_ratio', 'N/A')})")
                    with col2:
                        st.metric("Arêtes Inter-Cluster", f"{guarantees.get('inter_cluster_edges', 0)} ({guarantees.get('inter_ratio', 'N/A')})")

                elif st.session_state.method_key in ["EdgeFlip", "Laplace"]:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Budget ε (Epsilon)", guarantees.get('epsilon', 'N/A'))
                    with col2:
                        st.metric("Niveau de Privacy", guarantees.get('privacy_level', 'N/A'))
                    with col3:
                        st.metric("Perte Max de Privacy", guarantees.get('max_privacy_loss', 'N/A'))

                    if st.session_state.method_key == "EdgeFlip":
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Taux Faux Positifs Attendu", guarantees.get('expected_false_positive_rate', 'N/A'))
                        with col2:
                            st.metric("Taux Faux Négatifs Attendu", guarantees.get('expected_false_negative_rate', 'N/A'))

                elif st.session_state.method_key == "Probabilistic":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("k-Voisinage", guarantees.get('k_neighborhood', 'N/A'))
                    with col2:
                        st.metric("Tolérance ε", guarantees.get('epsilon_tolerance', 'N/A'))
                    with col3:
                        st.metric("Entropie Minimale", guarantees.get('min_entropy', 'N/A'))

                elif st.session_state.method_key == "Random Switch":
                    col1, col2 = st.columns(2)
                    with col1:
                        preserved = guarantees.get('degree_sequence_preserved', False)
                        st.metric("Séquence de Degrés Préservée", "✅ Oui" if preserved else "❌ Non")
                    with col2:
                        st.metric("Propriété Structurelle", guarantees.get('structural_property', 'N/A'))

                elif st.session_state.method_key == "Random Add/Del":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Budget de Modification (k)", guarantees.get('modification_budget', 'N/A'))
                    with col2:
                        st.metric("Incertitude Structurelle", guarantees.get('structural_uncertainty', 'N/A'))

        with tab4:
            st.markdown("## ℹ️ Comprendre le Taux d'Anonymisation")

            if st.button("❓ Qu'est-ce que l'Anonymisation ?"):
                st.session_state.show_anon_info = not st.session_state.get('show_anon_info', False)

            if st.session_state.get('show_anon_info', False):
                st.markdown("""
                ### 🎯 Définition du Taux d'Anonymisation

                Le **taux d'anonymisation** quantifie à quel point le graphe a été modifié
                pour protéger la vie privée. Il n'y a pas une seule mesure universelle,
                mais plusieurs indicateurs :

                #### 1. **Taux de Modification des Arêtes**

                ```
                Taux = (Arêtes Ajoutées + Arêtes Supprimées) / (2 × Arêtes Originales)
                ```

                - **0%** : Aucune modification (pas d'anonymisation)
                - **50%** : Modification substantielle
                - **100%** : Graphe complètement différent

                **Interprétation** : Plus ce taux est élevé, plus l'attaquant aura du mal
                à retrouver la structure originale, mais plus l'utilité du graphe sera dégradée.

                ---

                #### 2. **Incorrectness (Taux d'Erreur)**

                Mesure combien de nœuds ne peuvent plus être ré-identifiés correctement
                par leur degré :

                ```
                Incorrectness = Nœuds avec Degré Modifié / Nombre Total de Nœuds
                ```

                - **0%** : Tous les degrés préservés (ex: Random Switch)
                - **50%** : La moitié des nœuds protégés
                - **100%** : Tous les nœuds protégés

                **Exemple** : Si un attaquant connaît que Alice a 5 amis, mais qu'après
                anonymisation elle en a 7, il ne pourra pas la retrouver.

                ---

                #### 3. **Entropie de Shannon**

                Mesure l'incertitude moyenne dans l'identification :

                ```
                H = -∑ p_i log₂(p_i)
                ```

                où p_i est la probabilité qu'un attaquant assigne au nœud i.

                - **H = 0** : Certitude complète (pas de privacy)
                - **H = log₂(k)** : k nœuds également probables (k-anonymity)
                - **H = log₂(n)** : Tous les nœuds également probables (anonymat complet)

                ---

                #### 4. **Budget de Privacy (ε en Differential Privacy)**

                Pour les méthodes basées sur la privacy différentielle :

                - **ε < 0.1** : Privacy très forte, forte distorsion
                - **ε ∈ [0.1, 1]** : Privacy forte, distorsion modérée
                - **ε ∈ [1, 10]** : Privacy modérée, faible distorsion
                - **ε > 10** : Privacy faible, très faible distorsion

                **Garantie mathématique** : La probabilité qu'un attaquant distingue
                deux graphes voisins est bornée par e^ε.

                ---

                ### 🔄 Trade-off Privacy vs Utilité

                Il existe toujours un compromis :

                | Anonymisation | Privacy | Utilité |
                |---------------|---------|---------|
                | Très Faible   | ⭐      | ⭐⭐⭐⭐⭐ |
                | Faible        | ⭐⭐    | ⭐⭐⭐⭐   |
                | Moyenne       | ⭐⭐⭐  | ⭐⭐⭐    |
                | Forte         | ⭐⭐⭐⭐ | ⭐⭐     |
                | Très Forte    | ⭐⭐⭐⭐⭐| ⭐      |

                **Objectif** : Trouver le bon équilibre selon le cas d'usage !

                ---

                ### 📊 Comparaison des Méthodes

                | Méthode | Taux Modification | Incorrectness | Type de Garantie |
                |---------|------------------|---------------|------------------|
                | Random Add/Del | Moyen (20-30%) | Moyen | Aucune formelle |
                | Random Switch | Nul (0%) | Nul | Degrés préservés |
                | k-degree | Faible (10-20%) | Élevé | k-anonymity |
                | Généralisation | Très élevé | Très élevé | k-anonymity |
                | Probabiliste | Élevé (50%+) | Élevé | Entropie > seuil |
                | Differential Privacy | Variable | Variable | ε-DP |

                **Note** : Les valeurs exactes dépendent des paramètres choisis (k, ε, etc.)
                """)

        with tab5:
            st.markdown(f"## 🛡️ Attaques et Garanties : {current_method['name']}")

            method_details = ATTACKS_AND_GUARANTEES.get(st.session_state.method_key, {})

            if method_details:
                # Attaques protégées
                st.markdown("### ✅ Attaques contre lesquelles la méthode protège")
                attacks_protected = method_details.get('attacks_protected', [])
                for attack in attacks_protected:
                    with st.expander(f"🛡️ {attack['name']}", expanded=False):
                        st.markdown(attack['description'])

                # Attaques vulnérables
                st.markdown("---")
                st.markdown("### ⚠️ Vulnérabilités et Limitations")
                attacks_vulnerable = method_details.get('attacks_vulnerable', [])
                for attack in attacks_vulnerable:
                    with st.expander(f"🚨 {attack['name']}", expanded=False):
                        st.markdown(attack['description'])

                # Avantages
                st.markdown("---")
                st.markdown("### ✅ Avantages de la Méthode")
                advantages = method_details.get('advantages', [])
                for adv in advantages:
                    st.markdown(adv)

                # Inconvénients
                st.markdown("---")
                st.markdown("### ❌ Inconvénients et Limitations")
                disadvantages = method_details.get('disadvantages', [])
                for dis in disadvantages:
                    st.markdown(dis)

                # Exemple Karate
                st.markdown("---")
                st.markdown("### 🥋 Exemple Concret : Graphe Karate Club")
                karate_example = method_details.get('karate_example', '')
                if karate_example:
                    st.markdown(karate_example)
                else:
                    st.info("Exemple à venir pour cette méthode.")
            else:
                st.warning("Informations détaillées non disponibles pour cette méthode.")

    else:
        st.info("👈 Sélectionnez une méthode et cliquez sur 'Anonymiser le Graphe' pour commencer")

        # Afficher un aperçu du graphe original
        st.markdown("### 📊 Aperçu du Graphe Original")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nœuds", G.number_of_nodes())
        with col2:
            st.metric("Arêtes", G.number_of_edges())
        with col3:
            st.metric("Degré Moyen", f"{sum(d for n, d in G.degree()) / G.number_of_nodes():.2f}")

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.5, alpha=0.6)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
        ax.set_title('Graphe Original', fontsize=16, fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)


if __name__ == "__main__":
    main()
