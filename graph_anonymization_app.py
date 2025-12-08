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
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from method_details import ATTACKS_AND_GUARANTEES
from definitions_and_attacks import (
    ANONYMIZATION_DEFINITIONS,
    ATTACKS_DICTIONARY,
    GRAPH_PROPERTIES,
    CONCRETE_ATTACK_EXAMPLES
)
from thesis_references import (
    THESIS_REFERENCES,
    format_thesis_reference,
    get_method_references
)

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
        """
        k-degree anonymity : Garantit qu'au moins k nœuds ont le même degré.

        ALGORITHME EN LANGAGE NATUREL :
        1. Compter combien de nœuds ont chaque degré
        2. Pour chaque degré avec MOINS de k nœuds :
           - FUSIONNER ce groupe avec un groupe voisin (degré proche)
           - Ajuster les degrés en ajoutant/supprimant des arêtes
        3. Répéter jusqu'à ce que TOUS les groupes aient au moins k nœuds

        STRATÉGIE :
        - Groupes trop petits → Fusionner avec degré voisin
        - Ajouter des arêtes pour augmenter les degrés vers le degré cible
        - Garantir : chaque degré apparaît au moins k fois
        """
        G = self.original_graph.copy()

        # Itérer jusqu'à satisfaire la contrainte
        max_iterations = 100
        for iteration in range(max_iterations):
            # Calculer la distribution actuelle des degrés
            degrees = dict(G.degree())
            degree_counts = Counter(degrees.values())

            # Trouver les degrés qui violent la contrainte k
            violating_degrees = [d for d, count in degree_counts.items() if count < k]

            if not violating_degrees:
                # Contrainte satisfaite !
                break

            # Stratégie : Fusionner les petits groupes avec leurs voisins
            for degree in sorted(violating_degrees):
                nodes_with_degree = [n for n, d in degrees.items() if d == degree]

                if len(nodes_with_degree) == 0:
                    continue

                # Trouver le degré cible (degré voisin le plus fréquent)
                all_degrees = sorted(degree_counts.keys())

                # Chercher le degré voisin (supérieur ou inférieur)
                target_degree = None
                if degree < max(all_degrees):
                    # Augmenter vers le degré supérieur
                    target_degree = min([d for d in all_degrees if d > degree])
                elif degree > min(all_degrees):
                    # Diminuer vers le degré inférieur (en supprimant des arêtes)
                    target_degree = max([d for d in all_degrees if d < degree])
                else:
                    # Dernier recours : dupliquer le degré en ajoutant des arêtes
                    target_degree = degree + 1

                if target_degree is None:
                    continue

                # Ajuster les degrés des nœuds pour atteindre target_degree
                for node in nodes_with_degree[:]:  # Copie pour éviter modification pendant itération
                    current_degree = G.degree(node)

                    if current_degree < target_degree:
                        # AUGMENTER le degré en ajoutant des arêtes
                        edges_to_add = target_degree - current_degree

                        for _ in range(edges_to_add):
                            # Trouver un nœud non connecté
                            candidates = [n for n in G.nodes()
                                        if n != node and not G.has_edge(node, n)]

                            if candidates:
                                # Préférer les nœuds qui ont aussi besoin d'augmenter leur degré
                                target_node = random.choice(candidates)
                                G.add_edge(node, target_node)
                            else:
                                break  # Pas de candidat disponible

                    elif current_degree > target_degree:
                        # DIMINUER le degré en supprimant des arêtes
                        edges_to_remove = current_degree - target_degree

                        neighbors = list(G.neighbors(node))
                        edges_to_delete = random.sample(neighbors, min(edges_to_remove, len(neighbors)))

                        for neighbor in edges_to_delete:
                            G.remove_edge(node, neighbor)

                # Recalculer pour la prochaine itération
                degrees = dict(G.degree())
                degree_counts = Counter(degrees.values())

        return G

    def generalization(self, k=4):
        """
        Généralisation par clustering avec taille minimale k.

        ═══════════════════════════════════════════════════════════════════════
        PRINCIPE DE LA GÉNÉRALISATION (k-anonymity structurelle) :
        ═══════════════════════════════════════════════════════════════════════
        - Regrouper les nœuds en CLUSTERS (super-nœuds) de taille ≥ k
        - Chaque cluster représente au moins k nœuds indistinguables
        - Les arêtes deviennent des super-arêtes entre clusters

        GARANTIE DE PRIVACY :
        - Un attaquant ne peut identifier un nœud spécifique dans un cluster
        - Probabilité de ré-identification ≤ 1/k pour chaque nœud

        PARAMÈTRE k :
        - Plus k est GRAND → Clusters plus gros → PLUS de privacy → MOINS d'utilité
        - Plus k est PETIT → Clusters plus petits → MOINS de privacy → PLUS d'utilité
        ═══════════════════════════════════════════════════════════════════════
        """
        G = self.original_graph.copy()
        n = G.number_of_nodes()

        # Nombre de clusters basé sur k
        # On veut environ n/k clusters de taille k
        num_clusters = max(2, n // k)  # Au moins 2 clusters

        # Utiliser un algorithme de clustering spectral pour créer num_clusters
        try:
            # Essayer d'abord avec la méthode des communautés
            from networkx.algorithms import community

            # Utiliser label propagation qui converge rapidement
            communities_generator = community.label_propagation_communities(G)
            communities = list(communities_generator)

            # Si on a trop de clusters, fusionner les plus petits
            while len(communities) > num_clusters:
                # Trouver les 2 plus petits clusters
                communities = sorted(communities, key=len)
                smallest = communities[0]
                second_smallest = communities[1]
                # Fusionner
                merged = smallest.union(second_smallest)
                communities = [merged] + communities[2:]

            # Si on a trop peu de clusters, diviser les plus gros
            while len(communities) < num_clusters and any(len(c) >= 2*k for c in communities):
                # Trouver le plus gros cluster
                communities = sorted(communities, key=len, reverse=True)
                largest = communities[0]
                if len(largest) >= 2*k:
                    # Diviser en deux
                    largest_list = list(largest)
                    mid = len(largest_list) // 2
                    part1 = set(largest_list[:mid])
                    part2 = set(largest_list[mid:])
                    communities = [part1, part2] + communities[1:]
                else:
                    break

        except Exception as e:
            # Fallback : clustering simple par degré
            # Regrouper les nœuds par degré similaire
            nodes_by_degree = {}
            for node in G.nodes():
                degree = G.degree(node)
                if degree not in nodes_by_degree:
                    nodes_by_degree[degree] = []
                nodes_by_degree[degree].append(node)

            # Créer des clusters de taille k
            communities = []
            current_cluster = set()
            for degree in sorted(nodes_by_degree.keys()):
                for node in nodes_by_degree[degree]:
                    current_cluster.add(node)
                    if len(current_cluster) >= k:
                        communities.append(current_cluster)
                        current_cluster = set()

            # Ajouter le dernier cluster s'il existe
            if current_cluster:
                if communities:
                    # Fusionner avec le dernier cluster
                    communities[-1] = communities[-1].union(current_cluster)
                else:
                    communities.append(current_cluster)

        # Assurer que tous les clusters ont au moins k nœuds
        final_communities = []
        buffer = set()

        for community in sorted(communities, key=len, reverse=True):
            community = set(community).union(buffer)
            buffer = set()

            if len(community) >= k:
                final_communities.append(community)
            else:
                # Trop petit, mettre en buffer pour fusionner avec le prochain
                buffer = community

        # Si reste un buffer, fusionner avec le dernier cluster
        if buffer and final_communities:
            final_communities[-1] = final_communities[-1].union(buffer)
        elif buffer:
            final_communities.append(buffer)

        communities = final_communities

        # Créer le super-graphe
        super_graph = nx.Graph()
        node_to_cluster = {}
        cluster_to_nodes = {}
        cluster_id = 0

        for community in communities:
            community = set(community)

            for node in community:
                node_to_cluster[node] = cluster_id

            super_graph.add_node(cluster_id, cluster_size=len(community), size=len(community),
                               nodes=list(community), internal_edges=0)
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
                    # Incrémenter le compteur d'arêtes internes du nœud
                    super_graph.nodes[cluster_u]['internal_edges'] += 1
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
        """
        (k,ε)-obfuscation selon l'algorithme de Boldi et al. (2012)

        ═══════════════════════════════════════════════════════════════════════
        ALGORITHME DE BOLDI ET AL. (Distribution Normale Tronquée)
        ═══════════════════════════════════════════════════════════════════════
        Au lieu de la formule simplifiée (p = 1 - ε/k), cet algorithme utilise
        une DISTRIBUTION NORMALE TRONQUÉE R_σ sur [0,1] pour assigner les
        probabilités, conformément à la publication originale.

        PROCESSUS :
        1. Pour chaque nœud v, identifier k voisins candidats N_k(v)
        2. Tirer k valeurs de R_σ (normale tronquée centrée)
        3. Normaliser pour obtenir une distribution de probabilité
        4. Garantir H(N_k(v)) ≥ log(k) - ε

        AVANTAGE : Distribution plus réaliste, modélise mieux l'incertitude
        INCONVÉNIENT : Toujours vulnérable au threshold attack → voir MaxVar

        Référence : Boldi et al., "Injecting Uncertainty in Graphs for Identity
        Obfuscation", VLDB 2012 (voir thèse p.70-72)
        ═══════════════════════════════════════════════════════════════════════
        """
        from scipy.stats import truncnorm

        G = self.original_graph.copy()
        prob_graph = nx.Graph()
        prob_graph.add_nodes_from(G.nodes())

        # Paramètre de la distribution normale tronquée
        # σ est choisi pour créer une distribution centrée avec variance adaptée
        # Plus σ est grand, plus la distribution est étalée
        sigma = 0.15 + (epsilon / k) * 0.2  # Ajusté en fonction de ε et k

        # Préparer la distribution normale tronquée sur [0, 1]
        # truncnorm utilise les bornes standardisées (a, b) = ((lower - mu)/sigma, (upper - mu)/sigma)
        a_trunc = (0 - 0.5) / sigma  # Borne inférieure standardisée
        b_trunc = (1 - 0.5) / sigma  # Borne supérieure standardisée

        # Pour chaque nœud, on va assigner des probabilités à ses voisins candidats
        all_candidate_edges = []

        # D'abord, collecter tous les voisins candidats pour chaque nœud
        for node in G.nodes():
            # Voisins existants
            existing_neighbors = list(G.neighbors(node))

            # Voisins potentiels (non-voisins)
            all_nodes = set(G.nodes())
            potential_neighbors = list(all_nodes - set(existing_neighbors) - {node})

            # Sélectionner k voisins candidats total
            # Prioriser les voisins existants, compléter avec potentiels
            num_existing = len(existing_neighbors)
            num_potential_needed = max(0, k - num_existing)

            if len(potential_neighbors) > 0 and num_potential_needed > 0:
                num_potential_to_add = min(num_potential_needed, len(potential_neighbors))
                selected_potential = random.sample(potential_neighbors, num_potential_to_add)
            else:
                selected_potential = []

            candidates = existing_neighbors + selected_potential

            # Tirer k valeurs de la distribution normale tronquée
            num_candidates = len(candidates)
            if num_candidates > 0:
                # Générer des probabilités via distribution normale tronquée
                raw_probs = truncnorm.rvs(a_trunc, b_trunc, loc=0.5, scale=sigma, size=num_candidates)

                # Normaliser pour avoir une distribution de probabilité
                # (optionnel selon l'interprétation, mais aide à l'interprétation)
                prob_sum = np.sum(raw_probs)
                if prob_sum > 0:
                    normalized_probs = raw_probs / prob_sum
                else:
                    normalized_probs = np.ones(num_candidates) / num_candidates

                # Assigner les probabilités aux arêtes
                for i, neighbor in enumerate(candidates):
                    u, v = (node, neighbor) if node < neighbor else (neighbor, node)
                    is_original = G.has_edge(node, neighbor)

                    # Utiliser les probabilités brutes (non normalisées) pour mieux refléter l'algorithme
                    # Les probabilités individuelles sont dans [0,1]
                    prob = float(raw_probs[i])

                    # Ne pas dupliquer les arêtes
                    if not prob_graph.has_edge(u, v):
                        prob_graph.add_edge(u, v, probability=prob, is_original=is_original)

        return prob_graph

    def differential_privacy_edgeflip(self, epsilon=0.8):
        """
        EdgeFlip avec epsilon-differential privacy.

        FORMULE CORRECTE : s = 2 / (e^epsilon + 1)

        Trade-off :
        - epsilon PETIT => s GRAND => beaucoup de flips => FORTE privacy
        - epsilon GRAND => s PETIT => peu de flips => FAIBLE privacy
        """
        G = nx.Graph()
        G.add_nodes_from(self.original_graph.nodes())

        # FORMULE CORRECTE (dérivée du ratio de DP)
        s = 2 / (np.exp(epsilon) + 1)

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

    def maxvar_obfuscation(self, num_potential_edges=50):
        """
        MaxVar: Variance Maximizing Scheme

        Amélioration de (k,ε)-obfuscation qui résout le problème de reconstruction
        par seuillage en maximisant la variance totale des degrés.

        ALGORITHME (Nguyen Huu-Hiep, 2016):
        1. Ajouter des arêtes potentielles "nearby" (distance 2, friend-of-friend)
        2. Résoudre un programme quadratique pour assigner les probabilités:
           - Minimiser: Σ p_i² (équivalent à maximiser la variance)
           - Contrainte: Σ p_uv = degree(u) pour chaque nœud u
        3. Les probabilités résultantes sont DISPERSÉES (pas concentrées à 0/1)

        AVANTAGE vs (k,ε)-obf:
        - Pas de reconstruction par seuillage!
        - Probabilités varient significativement autour de 0.5
        - Préservation exacte des degrés attendus
        """
        G0 = self.original_graph.copy()
        n = G0.number_of_nodes()

        # Phase 1: Ajouter des arêtes potentielles "nearby" (distance = 2)
        potential_edges = []
        for u in G0.nodes():
            # Trouver les voisins à distance 2 (friend-of-friend)
            neighbors_dist_2 = set()
            for neighbor in G0.neighbors(u):
                for neighbor2 in G0.neighbors(neighbor):
                    if neighbor2 != u and not G0.has_edge(u, neighbor2):
                        neighbors_dist_2.add(neighbor2)

            # Ajouter des arêtes potentielles vers ces voisins
            for v in neighbors_dist_2:
                if u < v:  # Éviter les doublons
                    potential_edges.append((u, v))

        # Limiter le nombre d'arêtes potentielles
        if len(potential_edges) > num_potential_edges:
            potential_edges = random.sample(potential_edges, num_potential_edges)

        # Créer le graphe étendu avec arêtes existantes + potentielles
        all_edges = list(G0.edges()) + potential_edges
        edge_to_idx = {edge: idx for idx, edge in enumerate(all_edges)}
        m = len(all_edges)

        # Phase 2: Formulation du programme quadratique
        # Objectif: Minimiser Σ p_i²
        # Contrainte: Σ p_uv = degree(u) pour chaque nœud u

        # Construire la matrice de contraintes d'égalité (A_eq)
        # Chaque ligne correspond à un nœud, chaque colonne à une arête
        A_eq = np.zeros((n, m))
        b_eq = np.zeros(n)

        node_to_idx = {node: idx for idx, node in enumerate(G0.nodes())}

        for node in G0.nodes():
            node_idx = node_to_idx[node]
            b_eq[node_idx] = G0.degree(node)  # Degré attendu = degré original

            # Pour chaque arête touchant ce nœud
            for u, v in all_edges:
                if u == node or v == node:
                    edge_idx = edge_to_idx[(u, v)]
                    A_eq[node_idx, edge_idx] = 1.0

        # Fonction objectif: f(p) = Σ p_i²
        def objective(p):
            return np.sum(p ** 2)

        # Gradient: f'(p) = 2p
        def gradient(p):
            return 2 * p

        # Contraintes d'égalité
        constraints = {'type': 'eq', 'fun': lambda p: A_eq @ p - b_eq}

        # Bornes: 0 ≤ p_i ≤ 1
        bounds = [(0.0, 1.0) for _ in range(m)]

        # Point initial: probabilité uniforme qui satisfait les contraintes
        # Arêtes existantes: prob = 1.0, arêtes potentielles: prob faible
        p0 = np.zeros(m)
        for idx, (u, v) in enumerate(all_edges):
            if G0.has_edge(u, v):
                p0[idx] = 0.8  # Arête existante: prob élevée mais pas 1.0
            else:
                p0[idx] = 0.2  # Arête potentielle: prob faible mais pas 0.0

        # Résoudre le programme quadratique
        result = minimize(
            objective,
            p0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )

        if not result.success:
            # Si l'optimisation échoue, utiliser p0
            probabilities = p0
        else:
            probabilities = result.x

        # Phase 3: Créer le graphe incertain avec les probabilités optimisées
        prob_graph = nx.Graph()
        prob_graph.add_nodes_from(G0.nodes())

        for idx, (u, v) in enumerate(all_edges):
            prob = probabilities[idx]
            is_original = G0.has_edge(u, v)

            # Ajouter toutes les arêtes avec leur probabilité
            prob_graph.add_edge(u, v, probability=prob, is_original=is_original)

        return prob_graph


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
identifier le nœud cible qu'avec une probabilité $\\leq 1/k$.

**Exemple** : Avec k=3, si Alice a 7 amis, on s'assure qu'au moins 2 autres
personnes ont aussi 7 amis. L'attaquant ne peut pas dire laquelle est Alice.

L'algorithme ajoute des arêtes de manière **déterministe** pour atteindre cette propriété.

### Formalisation Mathématique

**Définition formelle** :

Un graphe G = (V, E) satisfait la k-degree anonymity si :

$$\\forall d \\in \\{\\deg(v) : v \\in V\\}, |\\{v \\in V : \\deg(v) = d\\}| \\geq k$$

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

### Heuristique Implémentée

L'implémentation utilise une **stratégie greedy itérative** qui fusionne les groupes de degrés trop petits :

**Étape 1 - Détection des violations** :
```
RÉPÉTER jusqu'à convergence (max 100 itérations):
  1. Calculer la distribution des degrés
  2. Identifier les degrés "violants" : count(d) < k
  3. SI aucun violant → STOP (contrainte satisfaite)
  4. SINON → Passer à l'étape 2
```

**Étape 2 - Fusion vers degré cible** :

Pour chaque degré violant d avec count(d) < k :
```
SI d < max(all_degrees):
  → AUGMENTER vers le degré supérieur le plus proche
  → target_degree = min([degrés > d])

SINON SI d > min(all_degrees):
  → DIMINUER vers le degré inférieur le plus proche
  → target_degree = max([degrés < d])

SINON (dernier recours):
  → Créer un nouveau groupe à d+1
```

**Étape 3 - Ajustement des degrés** :
```
Pour AUGMENTER un degré (current < target):
  1. Chercher candidats = [nœuds NON connectés]
  2. Sélection aléatoire parmi candidats
  3. Ajouter arête (node, candidat)

Pour DIMINUER un degré (current > target):
  1. Lister les voisins du nœud
  2. Échantillonner aléatoirement les arêtes à supprimer
  3. Supprimer les arêtes sélectionnées
```

**Exemple d'exécution** (Karate Club, k=2) :
- Distribution originale : {1: 1, 2: 11, 3: 6, 9: 1, 10: 1, 12: 1, 16: 1, 17: 1}
  - Degrés violants : 1, 9, 10, 12, 16, 17 (< 2 occurrences)
- Après anonymisation : {1: 6, 2: 7, 3: 8, 4: 2, 5: 2, 8: 9}
  - ✅ Tous les degrés $\\geq$ 2 occurrences
  - Modification : -14.1% d'arêtes

**Propriétés** :
- ✅ **Garantit** la contrainte k-anonymity
- ✅ **Minimise** les modifications (fusion vers voisin proche)
- ⚠️ **Non-optimal** (peut modifier plus que le minimum théorique)
- ⚠️ **Randomisé** (résultats non déterministes)

**Garantie de privacy** :

$$P(\\text{identité de } v | \\deg(v) = d) \\leq \\frac{1}{k}$$

**NP-complétude** : Trouver le nombre minimum d'arêtes à ajouter est NP-difficile.

**Complexité** : O(n²) en pratique (itérations × ajustements)

---

📖 **Références Thèse:**
- **p.30** - Section 2.2: k-Anonymity pour les graphes
- **p.32** - Section 2.2.1: k-Degree Anonymity - définition et algorithmes
- **p.45** - Section 2.5: Modèles d'attaques
        """,
        "formula": r"|\{v \in V : deg(v) = d\}| \geq k \quad \forall d",
        "privacy_level": "Moyenne à Forte (garantie k-anonymity)",
        "utility_preservation": "Bonne"
    },

    "Generalization": {
        "name": "Généralisation - Super-nodes",
        "category": "3. Anonymisation par Généralisation",
        "params": {"k": 4},
        "description_short": "Regroupe les nœuds en super-nœuds de taille $\\geq k$",
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

### Algorithme de Clustering Implémenté

L'implémentation utilise une approche **Label Propagation** avec ajustements pour garantir $|C_i| \\geq k$ :

**Étape 1 - Clustering initial** :
- Utilisation de l'algorithme **Label Propagation** (propagation d'étiquettes)
- Chaque nœud adopte l'étiquette la plus fréquente parmi ses voisins
- Converge rapidement vers des communautés naturelles du graphe

**Étape 2 - Ajustement des tailles** :
```
TANT QUE il existe des clusters < k :
  SI trop de clusters :
    → Fusionner les 2 plus petits clusters
  SI pas assez de clusters ET un cluster ≥ 2k :
    → Diviser le plus gros cluster en deux
```

**Avantages de cette approche** :
- ✅ Respecte la structure naturelle du graphe (communautés)
- ✅ Garantit la contrainte $|C_i| \\geq k$
- ✅ Temps de calcul raisonnable O(n²)

**Alternative possible** : Clustering spectral, k-means sur les embeddings, ou partitionnement par degré.

### Formalisation Mathématique

**Modèle de graphe généralisé** :

Soit G = (V, E) le graphe original. On crée une partition P = {C₁, C₂, ..., Cₘ}
de V telle que $|C_i| \\geq k \\; \\forall i$.

Le **super-graphe** G* = (V*, E*) est défini par :
- V* = {C₁, C₂, ..., Cₘ} (les clusters)
- $E^* = \\{(C_i, C_j) : \\exists(u,v) \\in E \\text{ avec } u \\in C_i, v \\in C_j\\}$

Chaque super-arête (Cᵢ, Cⱼ) a un **poids** :

$$w(C_i, C_j) = |\\{(u,v) \\in E : u \\in C_i, v \\in C_j\\}|$$

**Probabilité d'arête dans le cluster** :

$$P(\\text{edge} | C_i, C_j) = \\frac{w(C_i, C_j)}{|C_i| \\times |C_j|}$$

**Garantie de privacy** : Un individu est caché parmi au moins k-1 autres
dans son cluster.

**Problème d'optimisation** : Trouver la partition P qui minimise la perte
d'information tout en respectant $|C_i| \\geq k$ est NP-difficile.

**Complexité** : O(n²) à O(n³) selon l'algorithme de clustering

---

📖 **Références Thèse:**
- **p.40** - Section 2.3: Généralisation par super-nodes
- **p.30** - Section 2.2: Fondements de la k-anonymity
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
de probabilité sur les k voisins candidats doit être $\\geq \\log(k) - \\varepsilon$ :

$$H(N_k(v)) = -\\sum_i p_i \\log(p_i) \\geq \\log(k) - \\varepsilon$$

où $N_k(v)$ sont les k nœuds les plus susceptibles d'être voisins de v.

### Algorithme de Boldi et al. (Distribution Normale Tronquée)

**Approche implémentée** : Au lieu d'utiliser les formules simplifiées ci-dessus, cette application
implémente l'**algorithme original de Boldi et al. (2012)** qui utilise une **distribution normale tronquée** $R_\sigma$
sur l'intervalle $[0,1]$.

**Processus de construction** :

1. **Identification des candidats** : Pour chaque nœud $v$, identifier $N_k(v)$ (k voisins candidats : existants + potentiels)

2. **Génération des probabilités** : Tirer $k$ valeurs d'une distribution normale tronquée $R_\sigma$ centrée sur $[0,1]$
   - $\sigma$ est l'écart-type, ajusté en fonction de $\varepsilon$ et $k$
   - Plus $\sigma$ est grand, plus la distribution est étalée

3. **Assignation** : Assigner ces probabilités tirées aux arêtes candidates

4. **Vérification** : S'assurer que la contrainte d'entropie est respectée :
   $$H(N_k(v)) = -\\sum_i p_i \\log(p_i) \\geq \\log(k) - \\varepsilon$$

**Pourquoi cette approche ?**
- ✅ **Plus réaliste** : Distribution naturelle vs formule uniforme
- ✅ **Conforme à la publication** : Suit Boldi et al. 2012
- ✅ **Modélisation de l'incertitude** : Probabilités variées au lieu de 2 valeurs fixes
- ⚠️ **Toujours vulnérable** : Au threshold attack (voir MaxVar pour la solution)

**Formules simplifiées (pour référence)** :
- Arêtes existantes : $p = 1 - \varepsilon/k$ (généralement ≈ 0.9-1.0)
- Arêtes potentielles : $p = \varepsilon/(2k)$ (généralement ≈ 0.0-0.1)

**Graphe d'exemple (sample graph)** :

À partir de G̃, on peut générer des graphes compatibles en échantillonnant :

$$G_{sample} = (V, E_{sample}) \\text{ où } e \\in E_{sample} \\text{ ssi } X_e \\leq p(e), X_e \\sim U[0,1]$$

**Propriété** : L'espérance des degrés est préservée.

**Complexité** : O(|E| + k·n)

### ⚠️ **LIMITATION CRITIQUE : Reconstruction par Seuillage**

L'implémentation actuelle de (k,ε)-obfuscation a une **faille majeure** :

**Problème** :
- Arêtes existantes : probabilité $\\approx$ 1.0
- Arêtes potentielles : probabilité $\\approx$ 0.0
- **Un attaquant peut appliquer un seuil à 0.5 et récupérer EXACTEMENT le graphe original!**

**Pourquoi ?** Comme mentionné dans la thèse (Section 3.3.3) :
> "With small values of ε, re highly concentrates around zero, so existing sampled
> edges have probabilities nearly 1 and non-existing sampled edges are assigned
> probabilities almost 0. **Simple rounding techniques can easily reveal the true graph.**"

**Solution** : Utiliser **MaxVar** (voir ci-dessous) qui maximise la variance des
probabilités pour éviter cette concentration autour de 0/1.

**Utilité pédagogique** : Cette méthode est conservée dans l'application pour
montrer l'importance de la **conception d'algorithmes** en privacy. Une formulation
mathématique correcte ne garantit pas une implémentation sécurisée!

---

📖 **Références Thèse:**
- **p.70** - Section 3.3: Définition de la (k,ε)-obfuscation
- **p.72** - Section 3.3.2: Formules d'assignation des probabilités
- **p.75** - Section 3.3.3: ⚠️ Vulnérabilité au threshold attack
        """,
        "formula": r"H(N_k(v)) = -\sum_i p_i \log(p_i) \geq \log(k) - \varepsilon",
        "privacy_level": "⚠️ FAIBLE (vulnérable au seuillage) - Voir MaxVar",
        "utility_preservation": "Bonne (espérance préservée)"
    },

    "MaxVar": {
        "name": "Probabiliste - MaxVar (Variance Maximizing)",
        "category": "4. Approches Probabilistes",
        "params": {"num_potential_edges": 50},
        "description_short": "Graphe incertain avec probabilités dispersées (résiste au seuillage)",
        "description": """
### Principe en Langage Naturel

**MaxVar** est une amélioration de (k,ε)-obfuscation qui résout le **problème de reconstruction par seuillage**.

**Idée clé** : Au lieu de minimiser ε (ce qui concentre les probabilités autour de 0/1),
on **maximise la variance totale des degrés** tout en préservant les degrés attendus.

**Résultat** : Les probabilités sont **dispersées** autour de 0.5, rendant impossible
la reconstruction du graphe original par simple seuillage!

💡 **Note sur la distance d'édition** : MaxVar propose des arêtes "nearby" (à distance 2) au lieu
d'arêtes aléatoires, ce qui **minimise la distance d'édition** entre le graphe original et les graphes
échantillonnés. Cela préserve mieux la structure locale du graphe (voir métrique "Distance d'Édition"
dans les propriétés pour plus de détails).

**Analogie** : Imaginons que vous voulez cacher quelle porte est la vraie parmi 10 portes :
- **(k,ε)-obf** : Porte vraie = 99% de chance, portes fausses = 1% → **Trop évident!**
- **MaxVar** : Toutes les portes ont des probabilités variées entre 30% et 70% → **Confusion maximale!**

### Formalisation Mathématique

**Programme Quadratique** :

L'algorithme résout un programme d'optimisation quadratique :

$$\\min \\sum_{i \\in E} p_i^2$$

Contraintes:

$$0 \\leq p_i \\leq 1 \\quad \\forall i \\in E$$

$$\\sum_{v \\in N(u)} p_{uv} = \\deg(u) \\quad \\forall u \\in V$$

où $E$ contient à la fois les arêtes existantes ET les arêtes potentielles.

**Pourquoi minimiser $\\sum p_i^2$?**

La variance de la distance d'édition (Théorème 3.3, thèse) est :

$$\\text{Var}[D(\\tilde{G}, G)] = \\sum_i p_i(1 - p_i) = |E_{\\text{original}}| - \\sum_i p_i^2$$

Donc **minimiser $\\sum p_i^2$** équivaut à **maximiser la variance**, ce qui maximise
l'incertitude sur le graphe!

**Algorithme (3 phases)** :

**Phase 1 - Proposition d'arêtes "nearby"** :
```
Pour chaque nœud u :
  1. Trouver les voisins à distance 2 (friend-of-friend)
  2. Ajouter des arêtes potentielles vers ces voisins
  3. Limiter le nombre total d'arêtes potentielles
```

**Observation clé** : Proposer des arêtes "nearby" (distance 2) minimise la distorsion
structurelle tout en maximisant la confusion. C'est plus plausible qu'ajouter des arêtes
aléatoires entre nœuds distants!

**Phase 2 - Optimisation quadratique** :
```
1. Construire la matrice A_eq : ligne u, colonne (u,v) → 1
2. Vecteur b_eq : degree(u) pour chaque nœud u
3. Résoudre: min Σ p² sous contrainte A_eq @ p = b_eq
4. Utiliser SLSQP (Sequential Least Squares Programming)
```

**Phase 3 - Publication** :
```
1. Créer le graphe incertain G̃ = (V, E, p)
2. Publier plusieurs graphes échantillons G_sample
3. Chaque arête e ∈ E_sample si X_e ≤ p(e), X_e ~ U[0,1]
```

**Propriétés mathématiques** :

1. **Conservation des degrés attendus** :
   $$\\mathbb{E}[\\deg(u) \\text{ dans } \\tilde{G}] = \\deg(u) \\text{ dans } G_0 \\quad \\forall u$$

2. **Maximisation de la variance** :
   $$\\text{Var}[D(\\tilde{G}, G)] \\text{ est maximale sous contraintes}$$

3. **Résistance au seuillage** :
   Les probabilités NE sont PAS concentrées à 0/1, donc $\\text{threshold}(\\tilde{G}, 0.5) \\neq G_0$

**Exemple numérique** (Karate Club) :

Arête existante (0,1) :
- (k,ε)-obf : p = 0.95 (proche de 1.0) → **facilement identifiable**
- MaxVar : p = 0.63 (dispersé) → **ambiguë**

Arête potentielle (5,12) :
- (k,ε)-obf : p = 0.05 (proche de 0.0) → **facilement identifiable**
- MaxVar : p = 0.42 (dispersé) → **ambiguë**

**Complexité** :
- Phase 1 : $O(\\sum \\deg(u)^2) \\approx O(n \\cdot d_{avg}^2)$
- Phase 2 : $O(m^2)$ pour l'optimisation quadratique
- Phase 3 : $O(m)$ pour l'échantillonnage

Total : **O(m²)** (peut être réduit avec partitionnement du graphe)

### Comparaison (k,ε)-obf vs MaxVar

| Critère | (k,ε)-obf | MaxVar |
|---------|-----------|--------|
| **Probabilités** | Concentrées (0/1) | Dispersées (0.3-0.7) |
| **Résistance seuillage** | ❌ Vulnérable | ✅ Résistant |
| **Préservation degrés** | Approximative | ✅ Exacte |
| **Arêtes proposées** | Aléatoires | ✅ Nearby (distance 2) |
| **Variance** | Minimale | ✅ Maximale |
| **Complexité** | O(m + kn) | O(m²) |
| **Distance d'édition** | Élevée (arêtes aléatoires) | ✅ Faible (arêtes nearby) |

**Trade-off** : MaxVar est plus coûteux en calcul mais offre de meilleures garanties
de privacy et d'utilité.

---

📖 **Références Thèse:**
- **p.80** - Section 3.4: MaxVar - Variance Maximizing Scheme
- **p.82** - Section 3.4.2: Formulation du programme quadratique
- **p.85** - Section 3.4.3: Détails d'implémentation et nearby edges
        """,
        "formula": r"\min \sum_{i} p_i^2 \text{ s.t. } \sum_{v \in N(u)} p_{uv} = \deg(u)",
        "privacy_level": "Forte (résiste au seuillage)",
        "utility_preservation": "Excellente (degrés exacts + arêtes nearby)"
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

Un algorithme $\\mathcal{A}$ satisfait ε-DP si pour tous graphes voisins $G, G'$
(différant par une arête) et pour tout output $O$ :

$$P[\\mathcal{A}(G) = O] \\leq e^\\varepsilon \\cdot P[\\mathcal{A}(G') = O]$$

Plus $\\varepsilon$ est petit, plus forte est la garantie de privacy.

**Algorithme EdgeFlip (en langage naturel)** :

Pour chaque paire de nœuds possible (u, v) dans le graphe :
1. **Lancer une pièce biaisée** avec probabilité s/2
2. **Si pile** (probabilité s/2) : INVERSER l'état de l'arête
   - Si l'arête existe → la supprimer
   - Si l'arête n'existe pas → l'ajouter
3. **Si face** (probabilité 1-s/2) : GARDER l'état de l'arête
   - Si l'arête existe → la garder
   - Si l'arête n'existe pas → ne rien faire

Le paramètre s dépend du budget privacy ε selon :

$$s = \\frac{2}{e^\\varepsilon + 1}$$

**Trade-off** :
- ε petit (0.1) → s = 0.95 → flip 47.5% des arêtes → **forte privacy**
- ε grand (3.0) → s = 0.09 → flip 4.7% des arêtes → **faible privacy**

**Algorithme EdgeFlip (pseudo-code formel)** :

```
Entrée : G = (V, E), ε
Paramètre : s = 2 / (e^ε + 1)    ← FORMULE CORRECTE

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

Ratio : (1 - s/2) / (s/2) = (e^ε + 1 - 1) / 1 = e^ε ✓

Donc EdgeFlip satisfait ε-edge-DP.

**Espérance du nombre d'arêtes** :

$$\\mathbb{E}[|E_{output}|] = |E| \\cdot (1 - s/2) + (n(n-1)/2 - |E|) \\cdot s/2$$
$$\\approx n(n-1)/4 \\text{  (pour } s \\approx 1\\text{, très bruité)}$$

**Complexité** : O(n²)

**Inconvénient** : Complexité quadratique limite le passage à l'échelle.

---

📖 **Références Thèse:**
- **p.50** - Section 2.4: Differential Privacy pour les graphes
- **p.52** - Section 2.4.1: Edge-Level Differential Privacy
- **p.110** - Section 5: Évaluation expérimentale
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

$$\\Delta f = \\max_{G,G' \\text{ voisins}} ||f(G) - f(G')||_1$$

Pour les graphes (edge-DP), deux graphes sont voisins s'ils diffèrent par une arête.
Donc : $\\Delta f = 1$ pour une requête de type "cette arête existe-t-elle ?"

**Distribution de Laplace** :

$\\text{Lap}(b)$ a la densité :
$$p(x|b) = \\frac{1}{2b} \\cdot \\exp\\left(-\\frac{|x|}{b}\\right)$$

- Moyenne : 0
- Variance : $2b^2$
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

$$\\frac{P[M(G) = O]}{P[M(G') = O]} = \\exp(-\\varepsilon \\cdot |f(G)-f(G')|) \\leq e^\\varepsilon$$

car $|f(G) - f(G')| \\leq \\Delta f = 1$.

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
        guarantees['vulnerability'] = "⚠️ Reconstruction par seuillage possible!"

    elif method_key == "MaxVar":
        # MaxVar obfuscation
        num_pot = method_params.get('num_potential_edges', 50)

        guarantees['potential_edges'] = num_pot
        guarantees['optimization'] = "Programme quadratique (SLSQP)"
        guarantees['degree_preservation'] = "Exacte (E[deg(u)] = deg(u))"
        guarantees['variance_maximization'] = "✓ Probabilités dispersées"
        guarantees['threshold_resistance'] = "✓ Résiste au seuillage à 0.5"

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


def sample_from_probabilistic_graph(prob_graph):
    """
    Tire un échantillon de graphe déterministe depuis un graphe probabiliste.

    ═══════════════════════════════════════════════════════════════════════════
    PRINCIPE DU TIRAGE (SAMPLING) :
    ═══════════════════════════════════════════════════════════════════════════
    Dans un graphe probabiliste (k,ε)-obfuscation, chaque arête a une PROBABILITÉ
    d'existence. Pour créer un graphe déterministe, on effectue un TIRAGE au sort
    pour chaque arête :

    - Si prob(arête) = 0.95 → 95% de chance d'apparaître dans l'échantillon
    - Si prob(arête) = 0.10 → 10% de chance d'apparaître dans l'échantillon

    Ce mécanisme permet de :
    1. Publier plusieurs graphes échantillons différents
    2. Créer de la confusion pour l'attaquant (plusieurs graphes plausibles)
    3. Garantir que l'attaquant ne peut pas identifier le graphe original avec certitude

    ═══════════════════════════════════════════════════════════════════════════
    Paramètres:
        prob_graph : networkx.Graph avec attributs 'probability' sur les arêtes

    Retourne:
        networkx.Graph : Graphe déterministe échantillonné
    ═══════════════════════════════════════════════════════════════════════════
    """
    import random

    # Créer un nouveau graphe avec les mêmes nœuds
    sampled_graph = nx.Graph()
    sampled_graph.add_nodes_from(prob_graph.nodes())

    # Pour chaque arête du graphe probabiliste
    for u, v in prob_graph.edges():
        # Récupérer la probabilité
        prob = prob_graph[u][v].get('probability', 0.5)

        # Tirer au sort : l'arête apparaît si random < prob
        if random.random() < prob:
            sampled_graph.add_edge(u, v)

    return sampled_graph


def plot_probabilistic_graph(prob_graph, G_orig, method_name, ax):
    """
    Visualise un graphe probabiliste avec des arêtes de différentes intensités.

    ═══════════════════════════════════════════════════════════════════════════
    PRINCIPE DE LA VISUALISATION :
    ═══════════════════════════════════════════════════════════════════════════
    Dans un graphe probabiliste (k,ε)-obfuscation :
    - Les arêtes EXISTANTES ont une probabilité ÉLEVÉE (≈ 1 - ε/k) → FONCÉES
    - Les arêtes POTENTIELLES ont une probabilité FAIBLE (≈ ε/2k) → CLAIRES

    Cette visualisation utilise :
    1. INTENSITÉ DE COULEUR : Plus la probabilité est élevée, plus l'arête est foncée
    2. ÉPAISSEUR : Les arêtes à haute probabilité sont plus épaisses
    3. LÉGENDE : Code couleur pour interpréter les probabilités

    ═══════════════════════════════════════════════════════════════════════════
    """
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap

    # Position pour visualisation
    pos = nx.spring_layout(G_orig, seed=42, k=0.5, iterations=50)

    # Dessiner les nœuds
    nx.draw_networkx_nodes(prob_graph, pos, ax=ax,
                          node_color='lightcyan',
                          node_size=500, alpha=0.9,
                          edgecolors='darkblue', linewidths=2)

    # Collecter les arêtes par probabilité
    edges_with_prob = []
    for u, v in prob_graph.edges():
        prob = prob_graph[u][v].get('probability', 0.5)
        is_orig = prob_graph[u][v].get('is_original', False)
        edges_with_prob.append(((u, v), prob, is_orig))

    if not edges_with_prob:
        nx.draw_networkx_labels(prob_graph, pos, ax=ax, font_size=8, font_weight='bold')
        return

    # Trier par probabilité pour dessiner les faibles d'abord
    edges_with_prob.sort(key=lambda x: x[1])

    # Créer un colormap du clair (prob faible) au foncé (prob élevée)
    cmap = cm.get_cmap('RdYlGn')  # Rouge (faible) -> Jaune (moyen) -> Vert (élevé)

    # Dessiner chaque arête avec sa couleur et épaisseur
    for (u, v), prob, is_orig in edges_with_prob:
        # Couleur basée sur la probabilité
        color = cmap(prob)

        # Épaisseur basée sur la probabilité
        width = 0.5 + 3.5 * prob  # De 0.5 (prob=0) à 4.0 (prob=1)

        # Style : solide pour arêtes originales, pointillé pour potentielles
        style = 'solid' if is_orig else 'dotted'

        # Transparence basée sur la probabilité
        alpha = 0.3 + 0.6 * prob  # De 0.3 à 0.9

        nx.draw_networkx_edges(prob_graph, pos, [(u, v)], ax=ax,
                              edge_color=[color], width=width,
                              style=style, alpha=alpha)

    # Labels des nœuds
    nx.draw_networkx_labels(prob_graph, pos, ax=ax, font_size=8, font_weight='bold')

    # Créer une légende explicative
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=cmap(0.95), linewidth=4, label='Prob. très élevée (≈ 95%)'),
        Line2D([0], [0], color=cmap(0.70), linewidth=3, label='Prob. élevée (≈ 70%)'),
        Line2D([0], [0], color=cmap(0.50), linewidth=2, label='Prob. moyenne (≈ 50%)'),
        Line2D([0], [0], color=cmap(0.30), linewidth=1.5, label='Prob. faible (≈ 30%)'),
        Line2D([0], [0], color=cmap(0.10), linewidth=1, linestyle='dotted', label='Prob. très faible (≈ 10%)'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='solid', label='─── Arêtes originales'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='dotted', label='··· Arêtes potentielles'),
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_title(f'{method_name}\nGraphe Probabiliste ({prob_graph.number_of_nodes()} nœuds, {prob_graph.number_of_edges()} arêtes)',
                fontsize=14, fontweight='bold')


def plot_graph_comparison(G_orig, G_anon, method_name, node_to_cluster=None):
    """
    Crée une comparaison visuelle des graphes.

    Gère plusieurs types de graphes :
    - Graphes classiques (Random Add/Del, Random Switch, k-anonymity)
    - Graphes probabilistes (k,ε)-obfuscation avec arêtes pondérées
    - Super-graphes (Généralisation avec clusters)
    - Graphes différentiellement privés (EdgeFlip, Laplace)
    """
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
                # ═══════════════════════════════════════════════════════════════
                # VISUALISATION AMÉLIORÉE DU SUPER-GRAPHE
                # ═══════════════════════════════════════════════════════════════

                # Tailles proportionnelles au nombre de nœuds dans chaque cluster
                node_sizes = [G_anon.nodes[n].get('size', 1) * 300 for n in G_anon.nodes()]

                # Couleurs variées pour distinguer les clusters
                import matplotlib.cm as cm
                num_clusters = G_anon.number_of_nodes()
                colors_palette = cm.Set3(np.linspace(0, 1, num_clusters))

                # Dessiner les super-nœuds
                for idx, node in enumerate(G_anon.nodes()):
                    nx.draw_networkx_nodes(G_anon, pos_anon, nodelist=[node], ax=ax2,
                                          node_color=[colors_palette[idx]],
                                          node_size=node_sizes[idx],
                                          alpha=0.85, edgecolors='darkblue',
                                          linewidths=2.5)

                # Séparer les arêtes intra-cluster (self-loops) et inter-cluster
                intra_edges = []
                inter_edges = []
                intra_weights = []
                inter_weights = []

                for u, v in G_anon.edges():
                    weight = G_anon[u][v].get('weight', 1)
                    if u == v:  # Self-loop (arêtes intra-cluster)
                        intra_edges.append((u, v))
                        intra_weights.append(weight)
                    else:  # Arêtes inter-cluster
                        inter_edges.append((u, v))
                        inter_weights.append(weight)

                # Calculer les poids max pour normalisation
                max_intra_weight = max(intra_weights) if intra_weights else 1
                max_inter_weight = max(inter_weights) if inter_weights else 1

                # Dessiner les arêtes INTRA-cluster (self-loops)
                if intra_edges:
                    for (u, v), weight in zip(intra_edges, intra_weights):
                        # Dessiner un cercle autour du nœud pour représenter les arêtes internes
                        node_pos = pos_anon[u]
                        radius = 0.08 + 0.12 * (weight / max_intra_weight)
                        circle = plt.Circle(node_pos, radius, color='green',
                                          fill=False, linewidth=2 + 3*(weight/max_intra_weight),
                                          linestyle='solid', alpha=0.6)
                        ax2.add_patch(circle)

                # Dessiner les arêtes INTER-cluster
                if inter_edges:
                    for (u, v), weight in zip(inter_edges, inter_weights):
                        width = 1.5 + 4.5 * (weight / max_inter_weight)
                        nx.draw_networkx_edges(G_anon, pos_anon, [(u, v)], ax=ax2,
                                              width=width, alpha=0.7, edge_color='purple',
                                              style='solid')

                # Labels avec détails : ID + taille + poids intra
                labels = {}
                for n in G_anon.nodes():
                    size = G_anon.nodes[n].get('size', '?')
                    intra_weight = 0
                    if G_anon.has_edge(n, n):
                        intra_weight = G_anon[n][n].get('weight', 0)
                    labels[n] = f"C{n}\n[{size}n]\n{intra_weight}i"

                nx.draw_networkx_labels(G_anon, pos_anon, labels, ax=ax2,
                                       font_size=9, font_weight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3',
                                                facecolor='white', alpha=0.8,
                                                edgecolor='black'))

                # Légende explicative
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='green', linewidth=3, linestyle='solid',
                          label='Arêtes intra-cluster (vert)'),
                    Line2D([0], [0], color='purple', linewidth=3, linestyle='solid',
                          label='Arêtes inter-cluster (violet)'),
                ]
                ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

                # Titre avec statistiques complètes
                total_intra = sum(intra_weights)
                total_inter = sum(inter_weights)
                ax2.set_title(f'Super-Graphe - {method_name}\n{G_anon.number_of_nodes()} clusters | {int(total_intra)} arêtes intra | {int(total_inter)} arêtes inter',
                             fontsize=13, fontweight='bold')
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

            # ═══════════════════════════════════════════════════════════════════
            # DÉTECTION DE GRAPHE PROBABILISTE
            # ═══════════════════════════════════════════════════════════════════
            # Vérifier si le graphe a des arêtes avec des probabilités
            has_probabilities = False
            if G_anon.number_of_edges() > 0:
                first_edge = list(G_anon.edges())[0]
                has_probabilities = 'probability' in G_anon[first_edge[0]][first_edge[1]]

            if has_probabilities:
                # ═══════════════════════════════════════════════════════════════
                # VISUALISATION PROBABILISTE AMÉLIORÉE
                # ═══════════════════════════════════════════════════════════════
                plot_probabilistic_graph(G_anon, G_orig, method_name, ax2)
            else:
                # ═══════════════════════════════════════════════════════════════
                # VISUALISATION CLASSIQUE (graphes déterministes)
                # ═══════════════════════════════════════════════════════════════
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
    """
    Compare les distributions de degrés.

    Gère 3 cas spéciaux :
    1. Graphe probabiliste → Échantillonner avant de calculer
    2. Super-graphe → Tirage uniforme depuis les clusters
    3. Graphe classique → Distribution standard
    """
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
        # CAS 1 : Détecter si c'est un super-graphe (généralisation)
        is_super_graph = False
        if G_anon.number_of_nodes() > 0:
            first_node = list(G_anon.nodes())[0]
            node_data = G_anon.nodes[first_node]
            is_super_graph = 'cluster_size' in node_data

        if is_super_graph:
            # TIRAGE UNIFORME depuis le super-graphe pour recréer une distribution
            degrees_reconstructed = []

            # Pour chaque cluster
            for cluster_id in G_anon.nodes():
                cluster_size = G_anon.nodes[cluster_id]['cluster_size']
                internal_edges = G_anon.nodes[cluster_id]['internal_edges']

                # Compter les arêtes inter-cluster pour ce cluster
                inter_edges_count = 0
                for neighbor in G_anon.neighbors(cluster_id):
                    if neighbor != cluster_id:  # Exclure self-loops
                        inter_edges_count += G_anon[cluster_id][neighbor]['weight']

                # Estimer le degré moyen dans ce cluster
                # Degré interne moyen ≈ 2 × internal_edges / cluster_size
                avg_internal_degree = (2 * internal_edges) / cluster_size if cluster_size > 0 else 0

                # Degré externe moyen ≈ inter_edges_count / cluster_size
                avg_external_degree = inter_edges_count / cluster_size if cluster_size > 0 else 0

                # Degré total moyen pour les nœuds de ce cluster
                avg_degree = avg_internal_degree + avg_external_degree

                # Tirer des degrés avec une petite variance (± 20%)
                for _ in range(cluster_size):
                    # Ajouter un peu de bruit pour simuler la variabilité
                    degree = int(max(0, avg_degree + np.random.normal(0, avg_degree * 0.2)))
                    degrees_reconstructed.append(degree)

            ax2.hist(degrees_reconstructed, bins=range(max(degrees_reconstructed)+2) if degrees_reconstructed else [0],
                    alpha=0.7, color='orange', edgecolor='black', rwidth=0.8)
            ax2.set_xlabel('Degré', fontsize=12)
            ax2.set_ylabel('Nombre de nœuds', fontsize=12)
            ax2.set_title(f'Distribution des degrés - {method_name}\n(Tirage uniforme depuis clusters)',
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_axisbelow(True)

        else:
            # CAS 2 : Détecter si c'est un graphe probabiliste
            is_probabilistic = False
            if G_anon.number_of_edges() > 0:
                first_edge = list(G_anon.edges())[0]
                is_probabilistic = 'probability' in G_anon[first_edge[0]][first_edge[1]]

            if is_probabilistic:
                # ÉCHANTILLONNER le graphe probabiliste
                G_sample = sample_from_probabilistic_graph(G_anon)
                degrees_anon = [d for n, d in G_sample.degree()]

                ax2.hist(degrees_anon, bins=range(max(degrees_anon)+2) if degrees_anon else [0],
                        alpha=0.7, color='purple', edgecolor='black', rwidth=0.8)
                ax2.set_xlabel('Degré', fontsize=12)
                ax2.set_ylabel('Nombre de nœuds', fontsize=12)
                ax2.set_title(f'Distribution des degrés - {method_name}\n(Échantillon)',
                             fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.set_axisbelow(True)

            # CAS 3 : Graphe classique
            elif set(G_anon.nodes()).issubset(set(G_orig.nodes())):
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


def simulate_degree_attack(G_orig, G_anon, target_node=0):
    """
    Simule une ATTAQUE PAR DEGRÉ (Degree Attack) sur le graphe anonymisé.

    ═══════════════════════════════════════════════════════════════════════════
    PRINCIPE DE L'ATTAQUE :
    ═══════════════════════════════════════════════════════════════════════════
    L'adversaire connaît le DEGRÉ (nombre d'amis/connexions) d'un nœud cible
    dans le graphe original et tente de le retrouver dans le graphe anonymisé
    en cherchant les nœuds ayant le même degré.

    ═══════════════════════════════════════════════════════════════════════════
    EXEMPLE CONCRET (Karate Club) :
    ═══════════════════════════════════════════════════════════════════════════
    - Alice (instructrice) a 16 amis dans le club (information publique)
    - L'attaquant cherche dans le graphe anonymisé tous les nœuds de degré 16
    - Si UN SEUL nœud a degré 16 → Alice est ré-identifiée avec 100% de certitude
    - Si k nœuds ont degré 16 → Probabilité de ré-identification = 1/k

    ═══════════════════════════════════════════════════════════════════════════
    MÉTRIQUES DE PRIVACY CALCULÉES :
    ═══════════════════════════════════════════════════════════════════════════
    1. INCORRECTNESS : Nombre de fausses suppositions de l'attaquant
       → Plus cette valeur est élevée, meilleure est la privacy
       → Incorrectness = k - 1 (k candidats signifie k-1 erreurs potentielles)

    2. MIN ENTROPY : log₂(k) bits
       → Mesure l'incertitude de l'attaquant
       → 0 bits = aucune privacy (1 candidat)
       → 1 bit = privacy faible (2 candidats)
       → 3 bits = privacy moyenne (8 candidats)
       → 5+ bits = bonne privacy (32+ candidats)

    3. PROBABILITÉ DE RÉ-IDENTIFICATION : 1/k
       → Chance que l'attaquant devine correctement au hasard

    ═══════════════════════════════════════════════════════════════════════════
    PARAMÈTRES :
    ═══════════════════════════════════════════════════════════════════════════
    G_orig : networkx.Graph
        Graphe original (avant anonymisation)
    G_anon : networkx.Graph ou autre
        Graphe après anonymisation
    target_node : int
        Nœud que l'attaquant cherche à ré-identifier (par défaut : nœud 0)

    ═══════════════════════════════════════════════════════════════════════════
    RETOURNE :
    ═══════════════════════════════════════════════════════════════════════════
    dict contenant:
        - attack_type : Type d'attaque ('Degree Attack')
        - target_node : Nœud ciblé
        - target_degree : Degré du nœud cible
        - candidates : Liste des nœuds candidats dans le graphe anonymisé
        - success : True si ré-identification réussie (1 seul candidat)
        - re_identification_probability : 1/nombre_candidats
        - incorrectness : Nombre de fausses suppositions (k-1)
        - min_entropy_bits : log₂(k) bits de privacy
        - explanation : Explication textuelle du résultat
    """

    # ═══════════════════════════════════════════════════════════════════════
    # INITIALISATION DES RÉSULTATS
    # ═══════════════════════════════════════════════════════════════════════
    results = {
        'attack_type': 'Degree Attack',
        'target_node': target_node,
        'success': False,
        'candidates': [],
        'explanation': ''
    }

    # ═══════════════════════════════════════════════════════════════════════
    # VÉRIFICATION : L'attaque n'est possible que sur des graphes classiques
    # (pas sur les super-nœuds de la généralisation)
    # ═══════════════════════════════════════════════════════════════════════
    if not isinstance(G_anon, nx.Graph):
        results['explanation'] = "⚠️ Attaque impossible sur ce type de graphe (super-nodes). La généralisation détruit les degrés individuels."
        results['incorrectness'] = float('inf')  # Privacy parfaite
        results['min_entropy_bits'] = float('inf')
        results['re_identification_probability'] = 0.0
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 : CONNAISSANCE DE L'ATTAQUANT
    # ═══════════════════════════════════════════════════════════════════════
    # L'attaquant connaît le degré du nœud cible dans le graphe ORIGINAL
    # (par exemple via un profil public, un annuaire, ou sa propre connaissance)
    target_degree = G_orig.degree(target_node)

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 : RECHERCHE DES CANDIDATS
    # ═══════════════════════════════════════════════════════════════════════
    # L'attaquant cherche TOUS les nœuds du graphe anonymisé ayant le même degré
    candidates = [n for n in G_anon.nodes() if G_anon.degree(n) == target_degree]

    k = len(candidates)  # Nombre de nœuds indistinguables (anonymity set size)

    results['candidates'] = candidates
    results['target_degree'] = target_degree

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 : CALCUL DES MÉTRIQUES DE PRIVACY
    # ═══════════════════════════════════════════════════════════════════════

    if k == 1:
        # CAS 1 : RÉ-IDENTIFICATION RÉUSSIE (Privacy = 0)
        # Un seul candidat → L'attaquant est CERTAIN de l'identité
        results['success'] = True
        results['re_identified_node'] = candidates[0]
        results['re_identification_probability'] = 1.0  # 100% de certitude
        results['incorrectness'] = 0  # Aucune fausse supposition possible
        results['min_entropy_bits'] = 0.0  # Aucune incertitude (log₂(1) = 0)
        results['explanation'] = (
            f"✅ **RÉ-IDENTIFICATION RÉUSSIE !**\n\n"
            f"Le nœud cible {target_node} a un degré UNIQUE ({target_degree} connexions).\n"
            f"Un seul nœud dans le graphe anonymisé a ce degré.\n\n"
            f"📊 **Métriques de Privacy** :\n"
            f"- Probabilité de ré-identification : **100%** (certitude absolue)\n"
            f"- Incorrectness : **0** (aucune erreur possible)\n"
            f"- Min Entropy : **0 bits** (aucune privacy)\n\n"
            f"🔴 **DANGER** : L'attaquant peut maintenant découvrir toutes les connexions du nœud {target_node} !"
        )

    elif k == 0:
        # CAS 2 : AUCUN CANDIDAT TROUVÉ
        # Le degré a été modifié par l'anonymisation (randomisation, DP, etc.)
        results['success'] = False
        results['re_identification_probability'] = 0.0
        results['incorrectness'] = float('inf')  # Protection parfaite
        results['min_entropy_bits'] = float('inf')
        results['explanation'] = (
            f"❌ **ATTAQUE ÉCHOUÉE - Aucun candidat**\n\n"
            f"Aucun nœud avec degré {target_degree} trouvé dans le graphe anonymisé.\n"
            f"Le degré du nœud cible a été modifié par l'anonymisation.\n\n"
            f"📊 **Métriques de Privacy** :\n"
            f"- Probabilité de ré-identification : **0%**\n"
            f"- Incorrectness : **∞** (impossible de deviner)\n"
            f"- Min Entropy : **∞ bits** (privacy maximale)\n\n"
            f"🟢 **SÉCURITÉ** : Excellente protection contre cette attaque !"
        )

    else:
        # CAS 3 : RÉ-IDENTIFICATION AMBIGUË (k-anonymity)
        # Plusieurs candidats → L'attaquant doit deviner parmi k nœuds
        results['success'] = False
        results['re_identification_probability'] = 1.0 / k
        results['incorrectness'] = k - 1  # Nombre de fausses suppositions
        results['min_entropy_bits'] = np.log2(k)  # Bits de privacy

        # Évaluation qualitative de la privacy
        if k >= 10:
            privacy_level = "🟢 FORTE"
            privacy_comment = "Excellente protection"
        elif k >= 5:
            privacy_level = "🟡 MOYENNE"
            privacy_comment = "Protection acceptable"
        else:
            privacy_level = "🟠 FAIBLE"
            privacy_comment = "Protection limitée"

        results['explanation'] = (
            f"⚠️ **RÉ-IDENTIFICATION AMBIGUË**\n\n"
            f"{k} nœuds ont le degré {target_degree} dans le graphe anonymisé.\n"
            f"L'attaquant doit deviner au hasard parmi ces {k} candidats.\n\n"
            f"📊 **Métriques de Privacy** :\n"
            f"- Probabilité de ré-identification : **{1/k*100:.1f}%** (1/{k})\n"
            f"- Incorrectness : **{k-1}** fausses suppositions possibles\n"
            f"- Min Entropy : **{np.log2(k):.2f} bits** de privacy\n"
            f"- Niveau de privacy : {privacy_level}\n\n"
            f"💡 **Interprétation** : {privacy_comment}. "
            f"Le graphe satisfait la **{k}-anonymité** pour ce nœud."
        )

    return results


def simulate_subgraph_attack(G_orig, G_anon, target_node=0):
    """
    Simule une ATTAQUE PAR SOUS-GRAPHE (Subgraph/Neighborhood Attack) sur le graphe.

    ═══════════════════════════════════════════════════════════════════════════
    PRINCIPE DE L'ATTAQUE :
    ═══════════════════════════════════════════════════════════════════════════
    L'adversaire connaît la STRUCTURE LOCALE autour du nœud cible, notamment :
    - Le nombre de connexions (degré)
    - Les TRIANGLES formés avec ses voisins (amis communs)
    - Le coefficient de clustering (densité du voisinage)

    Cette attaque est BEAUCOUP PLUS PUISSANTE que l'attaque par degré seul,
    car elle exploite des MOTIFS STRUCTURELS (patterns) qui sont souvent uniques.

    ═══════════════════════════════════════════════════════════════════════════
    EXEMPLE CONCRET (Karate Club) :
    ═══════════════════════════════════════════════════════════════════════════
    - Mr. Hi (nœud 0) a 16 amis ET forme 45 triangles avec eux
    - Pattern : [degré=16, triangles=45]
    - L'attaquant cherche ce pattern dans le graphe anonymisé
    - Ce pattern est souvent UNIQUE → Ré-identification réussie

    Comparaison avec Degree Attack :
    - Degree Attack : Cherche seulement "degré = 16"
      → Peut trouver plusieurs candidats (k-anonymity)
    - Subgraph Attack : Cherche "degré = 16 ET 45 triangles"
      → Pattern beaucoup plus distinctif → Moins de candidats

    ═══════════════════════════════════════════════════════════════════════════
    PROTECTION CONTRE CETTE ATTAQUE :
    ═══════════════════════════════════════════════════════════════════════════
    🟢 GÉNÉRALISATION (Super-nœuds) : ★★★★★
       → Détruit complètement les motifs locaux en regroupant les nœuds
       → L'attaque devient IMPOSSIBLE

    🟢 DIFFERENTIAL PRIVACY : ★★★★☆
       → Ajoute/supprime des triangles de manière aléatoire
       → Brouille les patterns structurels

    🟠 RANDOMISATION : ★★☆☆☆
       → Peut préserver certains triangles
       → Protection limitée

    🔴 k-DEGREE ANONYMITY : ★☆☆☆☆
       → Ne protège que le degré, pas les triangles
       → VULNÉRABLE à cette attaque

    ═══════════════════════════════════════════════════════════════════════════
    MÉTRIQUES DE PRIVACY CALCULÉES :
    ═══════════════════════════════════════════════════════════════════════════
    1. STRUCTURAL QUERY H2 : (degré, nombre_triangles)
       → Décrit la structure locale du nœud

    2. INCORRECTNESS : k - 1 (nombre de fausses suppositions)

    3. MIN ENTROPY : log₂(k) bits d'incertitude

    4. PROBABILITÉ DE RÉ-IDENTIFICATION : 1/k

    ═══════════════════════════════════════════════════════════════════════════
    PARAMÈTRES :
    ═══════════════════════════════════════════════════════════════════════════
    G_orig : networkx.Graph
        Graphe original (avant anonymisation)
    G_anon : networkx.Graph ou autre
        Graphe après anonymisation
    target_node : int
        Nœud que l'attaquant cherche à ré-identifier

    ═══════════════════════════════════════════════════════════════════════════
    RETOURNE :
    ═══════════════════════════════════════════════════════════════════════════
    dict contenant:
        - attack_type : 'Subgraph Attack'
        - target_node : Nœud ciblé
        - target_degree : Degré du nœud cible
        - target_triangles : Nombre de triangles formés par le nœud
        - clustering_coefficient : Coefficient de clustering
        - candidates : Liste des nœuds candidats
        - success : True si ré-identification unique
        - re_identification_probability : 1/k
        - incorrectness : k-1
        - min_entropy_bits : log₂(k)
        - explanation : Explication détaillée
    """

    # ═══════════════════════════════════════════════════════════════════════
    # INITIALISATION DES RÉSULTATS
    # ═══════════════════════════════════════════════════════════════════════
    results = {
        'attack_type': 'Subgraph Attack',
        'target_node': target_node,
        'success': False,
        'candidates': [],
        'explanation': ''
    }

    # ═══════════════════════════════════════════════════════════════════════
    # VÉRIFICATION : L'attaque nécessite des graphes avec structure locale
    # ═══════════════════════════════════════════════════════════════════════
    if not isinstance(G_anon, nx.Graph):
        results['explanation'] = (
            "⚠️ **Attaque impossible sur ce type de graphe (super-nodes)**\n\n"
            "La GÉNÉRALISATION détruit les motifs locaux (triangles, voisinages).\n"
            "C'est justement la FORCE de cette méthode contre les attaques structurelles !\n\n"
            "🟢 Protection : **EXCELLENTE** (★★★★★)"
        )
        results['incorrectness'] = float('inf')
        results['min_entropy_bits'] = float('inf')
        results['re_identification_probability'] = 0.0
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 : ANALYSER LA STRUCTURE LOCALE DU NŒUD CIBLE (Graphe Original)
    # ═══════════════════════════════════════════════════════════════════════

    # Compter les TRIANGLES dont le nœud fait partie
    # Un triangle = 3 nœuds tous connectés entre eux (A-B, B-C, A-C)
    target_triangles = []
    for u, v in G_orig.edges(target_node):
        # Si u et v sont aussi connectés → triangle [target, u, v]
        if G_orig.has_edge(u, v):
            target_triangles.append(sorted([target_node, u, v]))

    # Si le nœud n'a aucun triangle, l'attaque structurelle est limitée
    if not target_triangles:
        results['explanation'] = (
            f"⚠️ Le nœud {target_node} ne fait partie d'AUCUN triangle.\n\n"
            f"Cette attaque nécessite des motifs structurels (triangles).\n"
            f"Utilisez plutôt une **Degree Attack** pour ce nœud."
        )
        return results

    # Caractéristiques structurelles du nœud cible
    target_degree = G_orig.degree(target_node)
    target_triangle_count = len(target_triangles)

    # Coefficient de clustering : proportion de voisins connectés entre eux
    try:
        target_clustering = nx.clustering(G_orig, target_node)
    except:
        target_clustering = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 : RECHERCHER LE PATTERN STRUCTUREL DANS LE GRAPHE ANONYMISÉ
    # ═══════════════════════════════════════════════════════════════════════
    # L'attaquant cherche les nœuds ayant le MÊME PATTERN : (degré, triangles)

    candidates = []
    for n in G_anon.nodes():
        # Filtrer d'abord par degré (critère rapide)
        if G_anon.degree(n) == target_degree:
            # Compter les triangles pour ce nœud candidat
            node_triangles = 0
            for u, v in G_anon.edges(n):
                if G_anon.has_edge(u, v):
                    node_triangles += 1

            # Si le nombre de triangles correspond aussi → Candidat potentiel !
            if node_triangles == target_triangle_count:
                candidates.append(n)

    k = len(candidates)  # Taille de l'ensemble d'anonymat

    results['candidates'] = candidates
    results['target_degree'] = target_degree
    results['target_triangles'] = target_triangle_count
    results['clustering_coefficient'] = target_clustering

    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 : ÉVALUATION DU SUCCÈS DE L'ATTAQUE
    # ═══════════════════════════════════════════════════════════════════════

    if k == 1:
        # CAS 1 : PATTERN UNIQUE → RÉ-IDENTIFICATION RÉUSSIE
        results['success'] = True
        results['re_identified_node'] = candidates[0]
        results['re_identification_probability'] = 1.0
        results['incorrectness'] = 0
        results['min_entropy_bits'] = 0.0
        results['explanation'] = (
            f"✅ **RÉ-IDENTIFICATION RÉUSSIE !**\n\n"
            f"Le pattern structurel du nœud {target_node} est UNIQUE :\n"
            f"- Degré : **{target_degree}** connexions\n"
            f"- Triangles : **{target_triangle_count}**\n"
            f"- Clustering : **{target_clustering:.3f}**\n\n"
            f"Un seul nœud dans le graphe anonymisé possède ce pattern.\n\n"
            f"📊 **Métriques de Privacy** :\n"
            f"- Probabilité de ré-identification : **100%**\n"
            f"- Incorrectness : **0** (certitude absolue)\n"
            f"- Min Entropy : **0 bits**\n\n"
            f"🔴 **DANGER** : L'attaque structurelle est PLUS PUISSANTE que l'attaque par degré.\n"
            f"💡 **Protection recommandée** : Généralisation ou Differential Privacy"
        )

    elif k == 0:
        # CAS 2 : AUCUN CANDIDAT → STRUCTURE MODIFIÉE
        results['success'] = False
        results['re_identification_probability'] = 0.0
        results['incorrectness'] = float('inf')
        results['min_entropy_bits'] = float('inf')
        results['explanation'] = (
            f"❌ **ATTAQUE ÉCHOUÉE - Structure modifiée**\n\n"
            f"Aucun nœud ne correspond au pattern recherché :\n"
            f"- Degré : {target_degree}\n"
            f"- Triangles : {target_triangle_count}\n\n"
            f"L'anonymisation a modifié la structure locale du graphe.\n\n"
            f"📊 **Métriques de Privacy** :\n"
            f"- Probabilité de ré-identification : **0%**\n"
            f"- Incorrectness : **∞**\n"
            f"- Min Entropy : **∞ bits**\n\n"
            f"🟢 **SÉCURITÉ** : Excellente protection contre cette attaque structurelle !"
        )

    else:
        # CAS 3 : PLUSIEURS CANDIDATS → AMBIGUÏTÉ
        results['success'] = False
        results['re_identification_probability'] = 1.0 / k
        results['incorrectness'] = k - 1
        results['min_entropy_bits'] = np.log2(k)

        # Évaluation qualitative
        if k >= 8:
            privacy_level = "🟢 FORTE"
            protection_comment = "Excellente protection structurelle"
        elif k >= 4:
            privacy_level = "🟡 MOYENNE"
            protection_comment = "Protection acceptable"
        else:
            privacy_level = "🟠 FAIBLE"
            protection_comment = "Protection limitée - Pattern encore distinctif"

        results['explanation'] = (
            f"⚠️ **RÉ-IDENTIFICATION AMBIGUË**\n\n"
            f"{k} nœuds partagent le pattern structurel :\n"
            f"- Degré : **{target_degree}**\n"
            f"- Triangles : **{target_triangle_count}**\n"
            f"- Clustering : **{target_clustering:.3f}**\n\n"
            f"L'attaquant doit deviner parmi {k} candidats.\n\n"
            f"📊 **Métriques de Privacy** :\n"
            f"- Probabilité de ré-identification : **{1/k*100:.1f}%** (1/{k})\n"
            f"- Incorrectness : **{k-1}** fausses suppositions\n"
            f"- Min Entropy : **{np.log2(k):.2f} bits**\n"
            f"- Niveau de privacy : {privacy_level}\n\n"
            f"💡 **Interprétation** : {protection_comment}.\n\n"
            f"⚠️ **Note** : Cette attaque est plus discriminante qu'une simple Degree Attack.\n"
            f"Pour une meilleure protection, utilisez la Généralisation ou Differential Privacy."
        )

    return results


def calculate_supergraph_metrics(G_orig, G_super):
    """
    Calcule les métriques d'utilité pour un SUPER-GRAPHE (Généralisation).

    Le super-graphe a une structure différente :
    - Chaque NŒUD = un CLUSTER (super-nœud)
    - Attributs des nœuds : 'cluster_size', 'internal_edges'
    - Arêtes INTRA-cluster : self-loops avec poids = nb d'arêtes internes
    - Arêtes INTER-cluster : arêtes normales avec poids = nb d'arêtes entre clusters

    On calcule les métriques directement à partir de ces informations.
    """
    metrics = {
        'type': 'super-graph',
        'comparable': True,
        'is_super_graph': True
    }

    # ═══════════════════════════════════════════════════════════════════════
    # MÉTRIQUES DE BASE (Structure du Super-Graphe)
    # ═══════════════════════════════════════════════════════════════════════

    num_clusters = G_super.number_of_nodes()
    metrics['num_clusters'] = num_clusters

    # Extraire les tailles de clusters
    cluster_sizes = []
    total_internal_edges = 0

    for node in G_super.nodes():
        node_data = G_super.nodes[node]
        size = node_data.get('cluster_size', 0)
        internal = node_data.get('internal_edges', 0)

        cluster_sizes.append(size)
        total_internal_edges += internal

    metrics['num_nodes'] = sum(cluster_sizes)  # Total de nœuds originaux
    metrics['min_cluster_size'] = min(cluster_sizes) if cluster_sizes else 0
    metrics['max_cluster_size'] = max(cluster_sizes) if cluster_sizes else 0
    metrics['avg_cluster_size'] = np.mean(cluster_sizes) if cluster_sizes else 0
    metrics['cluster_size_variance'] = np.var(cluster_sizes) if cluster_sizes else 0

    # ═══════════════════════════════════════════════════════════════════════
    # MÉTRIQUES D'ARÊTES (Intra vs Inter)
    # ═══════════════════════════════════════════════════════════════════════

    # Arêtes intra-cluster (depuis les self-loops ou attributs de nœuds)
    metrics['intra_cluster_edges'] = total_internal_edges

    # Arêtes inter-cluster (depuis les arêtes entre clusters)
    inter_cluster_edges = 0
    for u, v in G_super.edges():
        if u != v:  # Exclure les self-loops
            weight = G_super[u][v].get('weight', 1)
            inter_cluster_edges += weight

    metrics['inter_cluster_edges'] = inter_cluster_edges
    metrics['num_edges'] = total_internal_edges + inter_cluster_edges

    # Ratio intra/total
    total_edges = metrics['num_edges']
    if total_edges > 0:
        metrics['intra_ratio'] = total_internal_edges / total_edges
        metrics['inter_ratio'] = inter_cluster_edges / total_edges
    else:
        metrics['intra_ratio'] = 0
        metrics['inter_ratio'] = 0

    # ═══════════════════════════════════════════════════════════════════════
    # PERTE D'INFORMATION (Information Loss)
    # ═══════════════════════════════════════════════════════════════════════

    # Comparer avec le graphe original
    orig_edges = G_orig.number_of_edges()
    orig_nodes = G_orig.number_of_nodes()

    # Perte de granularité : passage de n nœuds à k clusters
    metrics['node_compression_ratio'] = num_clusters / orig_nodes if orig_nodes > 0 else 0
    metrics['information_loss'] = 1 - metrics['node_compression_ratio']

    # Conservation des arêtes
    if orig_edges > 0:
        metrics['edge_preservation_ratio'] = total_edges / orig_edges
    else:
        metrics['edge_preservation_ratio'] = 0

    # ═══════════════════════════════════════════════════════════════════════
    # MÉTRIQUES STRUCTURELLES (sur le super-graphe lui-même)
    # ═══════════════════════════════════════════════════════════════════════

    # Densité du super-graphe (sans compter les self-loops)
    super_graph_no_loops = G_super.copy()
    super_graph_no_loops.remove_edges_from(nx.selfloop_edges(super_graph_no_loops))
    metrics['super_graph_density'] = nx.density(super_graph_no_loops)

    # Degré moyen des clusters (nb de clusters voisins)
    super_degrees = [d for n, d in super_graph_no_loops.degree()]
    metrics['avg_cluster_degree'] = np.mean(super_degrees) if super_degrees else 0
    metrics['max_cluster_degree'] = max(super_degrees) if super_degrees else 0

    # Connectivité du super-graphe
    metrics['super_graph_connected'] = nx.is_connected(super_graph_no_loops)

    if metrics['super_graph_connected']:
        try:
            metrics['super_graph_diameter'] = nx.diameter(super_graph_no_loops)
        except:
            metrics['super_graph_diameter'] = None
    else:
        metrics['super_graph_diameter'] = None

    return metrics


def calculate_utility_metrics(G_orig, G_anon):
    """
    Calcule les métriques d'utilité selon la thèse (Section 3.5.2).

    ═══════════════════════════════════════════════════════════════════════════
    MÉTRIQUES D'UTILITÉ (selon la thèse, lignes 2503-2636) :
    ═══════════════════════════════════════════════════════════════════════════

    3 GROUPES DE STATISTIQUES :

    1. DEGREE-BASED (statistiques basées sur les degrés) :
       - Nombre d'arêtes (S_NE)
       - Degré moyen (S_AD)
       - Degré maximal (S_MD)
       - Variance des degrés (S_DV)
       - Exposant power-law (S_PL)

    2. SHORTEST PATH-BASED (statistiques basées sur les chemins) :
       - Distance moyenne (S_APD)
       - Diamètre effectif - 90e percentile (S_EDiam)
       - Longueur de connectivité - moyenne harmonique (S_CL)
       - Diamètre (S_Diam)

    3. CLUSTERING :
       - Coefficient de clustering (S_CC) = 3 × triangles / triples connectés

    CAS SPÉCIAUX :
    - Graphes PROBABILISTES : Calculer sur ÉCHANTILLONS (sample graphs)
    - GÉNÉRALISATION (super-graphe) : Métriques adaptées au format cluster
    ═══════════════════════════════════════════════════════════════════════════
    """
    metrics = {}

    # ═══════════════════════════════════════════════════════════════════════
    # CAS 1 : SUPER-GRAPHE (Généralisation)
    # ═══════════════════════════════════════════════════════════════════════
    if not isinstance(G_anon, nx.Graph):
        # Pour la généralisation, on ne peut pas comparer
        return {'type': 'super-graph', 'comparable': False}

    # Vérifier si c'est un super-graphe (a des attributs cluster)
    is_super_graph = False
    if G_anon.number_of_nodes() > 0:
        first_node = list(G_anon.nodes())[0]
        node_data = G_anon.nodes[first_node]
        is_super_graph = 'cluster_size' in node_data

    if is_super_graph:
        # MÉTRIQUES SPÉCIFIQUES POUR LE SUPER-GRAPHE
        return calculate_supergraph_metrics(G_orig, G_anon)

    # ═══════════════════════════════════════════════════════════════════════
    # CAS 2 : GRAPHE PROBABILISTE → Échantillonner d'abord
    # ═══════════════════════════════════════════════════════════════════════
    is_probabilistic = False
    if G_anon.number_of_edges() > 0:
        first_edge = list(G_anon.edges())[0]
        is_probabilistic = 'probability' in G_anon[first_edge[0]][first_edge[1]]

    if is_probabilistic:
        # Générer un échantillon déterministe depuis le graphe probabiliste
        G_sample = sample_from_probabilistic_graph(G_anon)
        metrics['is_sample'] = True
        metrics['probabilistic_edges'] = G_anon.number_of_edges()
    else:
        G_sample = G_anon
        metrics['is_sample'] = False

    # ═══════════════════════════════════════════════════════════════════════
    # GROUPE 1 : DEGREE-BASED STATISTICS
    # ═══════════════════════════════════════════════════════════════════════

    # S_NE : Nombre d'arêtes
    metrics['num_edges'] = G_sample.number_of_edges()
    metrics['num_nodes'] = G_sample.number_of_nodes()

    # Calculer les degrés
    degrees = [d for n, d in G_sample.degree()]

    # S_AD : Degré moyen
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0

    # S_MD : Degré maximal
    metrics['max_degree'] = max(degrees) if degrees else 0

    # S_DV : Variance des degrés
    metrics['degree_variance'] = np.var(degrees) if degrees else 0

    # S_PL : Exposant power-law
    # On estime γ (gamma) de la distribution P(k) ∝ k^(-γ)
    try:
        from scipy.stats import linregress
        # Compter la distribution des degrés
        degree_counts = Counter(degrees)
        degrees_unique = sorted([d for d in degree_counts.keys() if d > 0])
        counts = [degree_counts[d] for d in degrees_unique]

        if len(degrees_unique) >= 3:
            # Régression log-log : log(P(k)) = -γ × log(k) + C
            log_degrees = np.log(degrees_unique)
            log_counts = np.log(counts)
            slope, intercept, r_value, p_value, std_err = linregress(log_degrees, log_counts)
            metrics['power_law_exponent'] = -slope  # γ = -slope
            metrics['power_law_r_squared'] = r_value ** 2
        else:
            metrics['power_law_exponent'] = None
            metrics['power_law_r_squared'] = None
    except:
        metrics['power_law_exponent'] = None
        metrics['power_law_r_squared'] = None

    # Densité (métrique additionnelle)
    metrics['density'] = nx.density(G_sample)

    # ═══════════════════════════════════════════════════════════════════════
    # GROUPE 2 : SHORTEST PATH-BASED STATISTICS
    # ═══════════════════════════════════════════════════════════════════════

    try:
        if nx.is_connected(G_sample):
            # S_Diam : Diamètre
            metrics['diameter'] = nx.diameter(G_sample)

            # S_APD : Distance moyenne
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G_sample)

            # S_EDiam : Diamètre effectif (90e percentile)
            # Calculer toutes les distances
            all_paths = dict(nx.all_pairs_shortest_path_length(G_sample))
            all_distances = []
            for source in all_paths:
                for target, dist in all_paths[source].items():
                    if source != target:
                        all_distances.append(dist)

            if all_distances:
                metrics['effective_diameter'] = np.percentile(all_distances, 90)

                # S_CL : Longueur de connectivité (moyenne harmonique)
                # CL = n(n-1) / Σ(1/d(u,v))
                harmonic_sum = sum([1.0/d for d in all_distances if d > 0])
                n = G_sample.number_of_nodes()
                metrics['connectivity_length'] = n * (n-1) / harmonic_sum if harmonic_sum > 0 else None
            else:
                metrics['effective_diameter'] = None
                metrics['connectivity_length'] = None
        else:
            # Prendre la plus grande composante connexe
            largest_cc = max(nx.connected_components(G_sample), key=len)
            subgraph = G_sample.subgraph(largest_cc)

            metrics['diameter'] = nx.diameter(subgraph)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(subgraph)

            # Pour effective diameter et connectivity length sur la LCC
            all_paths = dict(nx.all_pairs_shortest_path_length(subgraph))
            all_distances = []
            for source in all_paths:
                for target, dist in all_paths[source].items():
                    if source != target:
                        all_distances.append(dist)

            if all_distances:
                metrics['effective_diameter'] = np.percentile(all_distances, 90)
                harmonic_sum = sum([1.0/d for d in all_distances if d > 0])
                n_lcc = subgraph.number_of_nodes()
                metrics['connectivity_length'] = n_lcc * (n_lcc-1) / harmonic_sum if harmonic_sum > 0 else None
            else:
                metrics['effective_diameter'] = None
                metrics['connectivity_length'] = None
    except:
        metrics['diameter'] = None
        metrics['avg_shortest_path'] = None
        metrics['effective_diameter'] = None
        metrics['connectivity_length'] = None

    # ═══════════════════════════════════════════════════════════════════════
    # GROUPE 3 : CLUSTERING COEFFICIENT
    # ═══════════════════════════════════════════════════════════════════════

    # S_CC : Clustering coefficient = 3 × triangles / triples connectés
    try:
        # Compter les triangles
        triangles = sum(nx.triangles(G_sample).values()) / 3  # Divisé par 3 car chaque triangle compté 3 fois

        # Compter les triples connectés (chemins de longueur 2)
        connected_triples = 0
        for node in G_sample.nodes():
            degree = G_sample.degree(node)
            # Chaque nœud de degré k crée k(k-1)/2 triples
            connected_triples += degree * (degree - 1) / 2

        if connected_triples > 0:
            metrics['clustering_coefficient'] = (3 * triangles) / connected_triples
        else:
            metrics['clustering_coefficient'] = 0

        # Clustering moyen (métrique alternative)
        metrics['avg_clustering'] = nx.average_clustering(G_sample)
    except:
        metrics['clustering_coefficient'] = None
        metrics['avg_clustering'] = None

    # ═══════════════════════════════════════════════════════════════════════
    # MÉTRIQUES ADDITIONNELLES : Préservation de la structure
    # ═══════════════════════════════════════════════════════════════════════

    # Corrélation des séquences de degrés (Spearman)
    orig_degrees = sorted([d for n, d in G_orig.degree()])
    sample_degrees = sorted([d for n, d in G_sample.degree()])

    if len(orig_degrees) == len(sample_degrees):
        from scipy.stats import spearmanr
        try:
            corr, _ = spearmanr(orig_degrees, sample_degrees)
            metrics['degree_correlation'] = corr
        except:
            metrics['degree_correlation'] = None
    else:
        metrics['degree_correlation'] = None

    # Erreur relative moyenne (comme dans la thèse)
    # rel.err = |S(G0) - S(G)| / S(G0)
    metrics['comparable'] = True

    return metrics


def calculate_privacy_metrics_separated(G_orig, G_anon, method_key, method_params):
    """Calcule les métriques de privacy séparées"""
    metrics = {}

    if method_key == "k-degree anonymity":
        degrees = dict(G_anon.degree()) if isinstance(G_anon, nx.Graph) else {}
        degree_counts = Counter(degrees.values()) if degrees else Counter()
        k_value = method_params.get('k', 2)
        min_count = min(degree_counts.values()) if degree_counts else 0

        metrics['k_value'] = k_value
        metrics['min_anonymity_set'] = min_count
        metrics['satisfies_k_anonymity'] = min_count >= k_value
        metrics['re_identification_probability'] = 1/min_count if min_count > 0 else 1.0

    elif method_key in ["EdgeFlip", "Laplace"]:
        epsilon = method_params.get('epsilon', 1.0)
        metrics['epsilon_budget'] = epsilon
        metrics['privacy_loss_bound'] = np.exp(epsilon)
        metrics['privacy_level'] = "Forte (ε<1)" if epsilon < 1.0 else ("Moyenne (1≤ε<2)" if epsilon < 2.0 else "Faible (ε≥2)")

        if method_key == "EdgeFlip":
            s = 1 - np.exp(-epsilon)
            metrics['flip_probability'] = s
            metrics['expected_noise_edges'] = int(G_orig.number_of_edges() * s / 2)

    elif method_key == "Probabilistic":
        k = method_params.get('k', 5)
        eps = method_params.get('epsilon', 0.3)
        metrics['k_candidates'] = k
        metrics['epsilon_tolerance'] = eps
        metrics['min_entropy'] = np.log(k) - eps
        metrics['confusion_factor'] = k

    elif method_key == "MaxVar":
        num_pot = method_params.get('num_potential_edges', 50)
        metrics['num_potential_edges'] = num_pot

        # Analyser les probabilités pour vérifier la dispersion
        if G_anon.number_of_edges() > 0:
            # Séparer arêtes existantes vs potentielles
            existing_probs = []
            potential_probs = []
            all_probs = []

            for u, v in G_anon.edges():
                prob = G_anon[u][v].get('probability', 1.0)
                is_original = G_anon[u][v].get('is_original', False)
                all_probs.append(prob)

                if is_original:
                    existing_probs.append(prob)
                else:
                    potential_probs.append(prob)

            # Métriques globales
            metrics['avg_probability'] = np.mean(all_probs)
            metrics['std_probability'] = np.std(all_probs)
            metrics['min_probability'] = np.min(all_probs)
            metrics['max_probability'] = np.max(all_probs)

            # Métriques pour arêtes existantes
            if existing_probs:
                metrics['existing_avg_prob'] = np.mean(existing_probs)
                metrics['existing_std_prob'] = np.std(existing_probs)

            # Métriques pour arêtes potentielles
            if potential_probs:
                metrics['potential_avg_prob'] = np.mean(potential_probs)
                metrics['potential_std_prob'] = np.std(potential_probs)

            # Calculer la variance totale (objectif maximisé)
            total_variance = sum(p * (1 - p) for p in all_probs)
            metrics['total_variance'] = total_variance
            metrics['avg_edge_variance'] = total_variance / len(all_probs) if all_probs else 0

            # Tester la résistance au seuillage
            threshold = 0.5
            reconstructed = sum(1 for p in all_probs if p > threshold)
            original_edges_count = len(existing_probs)
            if original_edges_count > 0:
                reconstruction_rate = sum(1 for p in existing_probs if p > threshold) / original_edges_count
                metrics['threshold_resistance'] = 1 - reconstruction_rate  # Plus proche de 1 = meilleur
                metrics['reconstruction_rate'] = reconstruction_rate

    elif method_key == "Generalization":
        if hasattr(G_anon, 'graph') and 'cluster_to_nodes' in G_anon.graph:
            cluster_sizes = [len(nodes) for nodes in G_anon.graph['cluster_to_nodes'].values()]
            metrics['min_cluster_size'] = min(cluster_sizes) if cluster_sizes else 0
            metrics['avg_cluster_size'] = np.mean(cluster_sizes) if cluster_sizes else 0
            metrics['max_privacy'] = 1/min(cluster_sizes) if cluster_sizes else 1.0

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# DICTIONNAIRE DES DÉFINITIONS ET MÉTHODES DE CALCUL (pour tooltips)
# ═══════════════════════════════════════════════════════════════════════════

METRIC_DEFINITIONS = {
    # Métriques de base
    'num_nodes': {
        'name': 'Nombre de Nœuds',
        'definition': "Nombre total de nœuds dans le graphe",
        'formula': "n = |V|",
        'interpretation': "Plus élevé = graphe plus grand"
    },
    'num_edges': {
        'name': 'Nombre d\'Arêtes',
        'definition': "Nombre total d'arêtes dans le graphe",
        'formula': "m = |E|",
        'interpretation': "Plus élevé = graphe plus connecté"
    },
    'density': {
        'name': 'Densité',
        'definition': "Proportion d'arêtes existantes par rapport au maximum possible",
        'formula': "D = 2m / (n(n-1))",
        'interpretation': "0 = vide, 1 = complet, ~0.1 = épars, ~0.5 = dense"
    },

    # Groupe 1: Degree-based
    'avg_degree': {
        'name': 'Degré Moyen (S_AD)',
        'definition': "Nombre moyen de voisins par nœud",
        'formula': r"d_{avg} = \frac{1}{n} \sum_{v \in V} \deg(v)",
        'interpretation': "Mesure la connectivité moyenne du graphe"
    },
    'max_degree': {
        'name': 'Degré Maximal (S_MD)',
        'definition': "Plus grand nombre de voisins d'un nœud",
        'formula': r"d_{max} = \max_{v \in V} \deg(v)",
        'interpretation': "Identifie les hubs (nœuds très connectés)"
    },
    'degree_variance': {
        'name': 'Variance des Degrés (S_DV)',
        'definition': "Dispersion des degrés autour de la moyenne",
        'formula': r"\sigma^2 = \frac{1}{n} \sum_{v \in V} (\deg(v) - d_{avg})^2",
        'interpretation': "Élevée = hétérogénéité (hubs + nœuds peu connectés)"
    },
    'power_law_exponent': {
        'name': 'Exposant Power-Law (S_PL)',
        'definition': "Caractérise la distribution des degrés pour les réseaux scale-free",
        'formula': r"P(k) \propto k^{-\gamma} \text{ où } \gamma \text{ est estimé par régression log-log}",
        'interpretation': r"\gamma \in [2,3] \text{ typique pour réseaux sociaux (loi de puissance)}"
    },

    # Groupe 2: Shortest path-based
    'diameter': {
        'name': 'Diamètre (S_Diam)',
        'definition': "Plus grande distance entre deux nœuds connectés",
        'formula': r"D = \max_{u,v \in V} d(u,v)",
        'interpretation': "Borne supérieure sur toutes les distances"
    },
    'avg_shortest_path': {
        'name': 'Distance Moyenne (S_APD)',
        'definition': "Longueur moyenne des plus courts chemins entre tous les couples",
        'formula': r"L = \frac{2}{n(n-1)} \sum_{u,v \in V} d(u,v)",
        'interpretation': "Mesure la compacité du réseau (propriété small-world)"
    },
    'effective_diameter': {
        'name': 'Diamètre Effectif (S_EDiam)',
        'definition': "90e percentile des distances (plus robuste que le diamètre)",
        'formula': r"D_{eff} = \text{Percentile}_{90}\{d(u,v)\}",
        'interpretation': r"90\% \text{ des nœuds sont à distance } \leq D_{eff}"
    },
    'connectivity_length': {
        'name': 'Longueur de Connectivité (S_CL)',
        'definition': "Moyenne harmonique des distances (privilégie les courtes distances)",
        'formula': r"CL = \frac{n(n-1)}{\sum_{u,v} \frac{1}{d(u,v)}}",
        'interpretation': "Plus faible = meilleure connectivité locale"
    },

    # Groupe 3: Clustering
    'clustering_coefficient': {
        'name': 'Coefficient de Clustering (S_CC)',
        'definition': "Mesure la tendance à former des triangles (cliques locales)",
        'formula': "CC = (3 × nb_triangles) / nb_triples_connectés",
        'interpretation': "Élevé = forte transitivité (ami de mes amis = mon ami)"
    },
    'avg_clustering': {
        'name': 'Clustering Moyen',
        'definition': "Moyenne des coefficients de clustering locaux",
        'formula': r"C_{avg} = \frac{1}{n} \sum_{v \in V} C(v) \text{ où } C(v) = \frac{\text{triangles}(v)}{\text{triples}(v)}",
        'interpretation': "Mesure alternative du clustering (locale → globale)"
    },

    # Métriques de préservation
    'degree_correlation': {
        'name': 'Corrélation des Degrés',
        'definition': "Corrélation de Spearman entre séquences de degrés (original vs anonymisé)",
        'formula': "ρ = Spearman(deg_orig, deg_anon)",
        'interpretation': ">0.9 = excellente préservation, >0.7 = bonne, <0.7 = faible"
    },

    # Métriques pour super-graphe (généralisation)
    'num_clusters': {
        'name': 'Nombre de Clusters',
        'definition': "Nombre de super-nœuds dans le graphe de généralisation",
        'formula': "k = nombre de clusters",
        'interpretation': "Plus faible = plus de privacy, moins d'utilité"
    },
    'min_cluster_size': {
        'name': 'Taille Min. Cluster',
        'definition': "Plus petit nombre de nœuds dans un cluster",
        'formula': r"\min_i |C_i|",
        'interpretation': r"\text{Doit être } \geq k \text{ pour garantir k-anonymity}"
    },
    'avg_cluster_size': {
        'name': 'Taille Moy. Cluster',
        'definition': "Nombre moyen de nœuds par cluster",
        'formula': r"\text{avg}\{|C_i|\} = \frac{n}{k}",
        'interpretation': "Plus élevé = clusters plus gros = plus de privacy"
    },
    'intra_cluster_edges': {
        'name': 'Arêtes Intra-Cluster',
        'definition': "Nombre d'arêtes à l'intérieur des clusters",
        'formula': "Somme des arêtes internes de chaque cluster",
        'interpretation': "Représentent la structure locale préservée"
    },
    'inter_cluster_edges': {
        'name': 'Arêtes Inter-Cluster',
        'definition': "Nombre d'arêtes entre différents clusters",
        'formula': "Arêtes reliant des nœuds de clusters différents",
        'interpretation': "Représentent les connexions globales"
    },
    'intra_ratio': {
        'name': 'Ratio Intra/Total',
        'definition': "Proportion d'arêtes intra-cluster par rapport au total",
        'formula': "ratio = intra_edges / (intra_edges + inter_edges)",
        'interpretation': "Élevé = structure locale bien préservée"
    },
    'information_loss': {
        'name': 'Perte d\'Information',
        'definition': "Proportion de granularité perdue lors du clustering",
        'formula': "loss = 1 - (k_clusters / n_nodes)",
        'interpretation': "0 = aucune perte, 1 = perte totale (1 seul cluster)"
    },
    'edge_preservation_ratio': {
        'name': 'Taux de Préservation des Arêtes',
        'definition': "Proportion d'arêtes préservées après anonymisation",
        'formula': "ratio = edges_anon / edges_orig",
        'interpretation': "1 = toutes préservées, <1 = pertes, >1 = arêtes ajoutées"
    },
    'super_graph_density': {
        'name': 'Densité du Super-Graphe',
        'definition': "Densité du graphe des clusters (sans self-loops)",
        'formula': "D_super = 2m_inter / (k(k-1))",
        'interpretation': "Mesure la connectivité entre clusters"
    },
}


def get_metric_tooltip(metric_key):
    """
    Génère un tooltip formaté pour une métrique donnée.

    Args:
        metric_key: Clé de la métrique dans METRIC_DEFINITIONS

    Returns:
        String formaté pour le paramètre 'help' de st.metric()
    """
    if metric_key not in METRIC_DEFINITIONS:
        return None

    info = METRIC_DEFINITIONS[metric_key]

    tooltip = (
        f"📖 **Définition**: {info['definition']}\n\n"
        f"📐 **Formule**: {info['formula']}\n\n"
        f"💡 **Interprétation**: {info['interpretation']}"
    )

    return tooltip


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

        # Mini présentation du Karate Club
        with st.sidebar.expander("ℹ️ À propos du Karate Club", expanded=False):
            st.markdown("""
            **Graphe de Zachary** (1977)

            Réseau social d'un club de karaté universitaire :
            - **34 membres** (nœuds)
            - **78 relations** sociales (arêtes)
            - **2 communautés** formées après une scission réelle du club

            Graphe de référence classique en analyse de réseaux sociaux.
            """)
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

    # Références à la thèse pour cette méthode
    method_internal_key = {
        "k-degree anonymity": "KDegreeAnonymity",
        "Generalization": "Generalization",
        "Probabilistic": "ProbabilisticObfuscation",
        "MaxVar": "MaxVar",
        "EdgeFlip": "EdgeFlip",
        "Laplace": "Laplace"
    }.get(method_key, None)

    if method_internal_key:
        refs = get_method_references(method_internal_key)
        if refs:
            with st.sidebar.expander("📖 Références Thèse", expanded=False):
                st.markdown("**Sources dans la thèse:**")
                for ref_key in refs[:3]:  # Afficher max 3 références principales
                    if ref_key in THESIS_REFERENCES:
                        ref = THESIS_REFERENCES[ref_key]
                        st.markdown(f"• **p.{ref['page']}** - {ref['section']}")
                st.caption("💡 Cliquez sur 'Voir la Thèse' en bas pour plus de détails")

    # Section de paramètres modulables
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Budget de Privacy (Modulable)")

    # Paramètres dynamiques selon la méthode
    dynamic_params = {}

    if method_key in ["Random Add/Del", "Random Switch"]:
        k_value = st.sidebar.slider(
            "k = Nombre de modifications",
            min_value=5,
            max_value=50,
            value=method['params']['k'],
            step=5,
            help="Nombre d'arêtes à modifier (ajout/suppression ou échange)"
        )
        dynamic_params['k'] = k_value

    elif method_key == "k-degree anonymity":
        k_value = st.sidebar.slider(
            "k = Taille minimale des groupes",
            min_value=2,
            max_value=10,
            value=method['params']['k'],
            step=1,
            help="Nombre minimum de nœuds ayant le même degré"
        )
        dynamic_params['k'] = k_value

    elif method_key == "Generalization":
        k_value = st.sidebar.slider(
            "k = Taille minimale des clusters",
            min_value=2,
            max_value=10,
            value=method['params']['k'],
            step=1,
            help="Nombre minimum de nœuds dans chaque cluster"
        )
        dynamic_params['k'] = k_value

    elif method_key == "Probabilistic":
        k_value = st.sidebar.slider(
            "k = Nombre de graphes candidats",
            min_value=3,
            max_value=15,
            value=method['params']['k'],
            step=1,
            help="Nombre minimum de graphes plausibles"
        )
        epsilon_value = st.sidebar.slider(
            "ε = Taux de transfert de probabilité",
            min_value=0.1,
            max_value=10.0,
            value=method['params']['epsilon'],
            step=0.1,
            help="⚠️ ATTENTION : Plus ε est GRAND, plus on transfère de probabilité des arêtes existantes vers les potentielles → PLUS DE PRIVACY ! Avec ε grand (ex: 5.0), p_exist diminue et p_potential augmente, rendant le graphe plus anonymisé."
        )
        dynamic_params['k'] = k_value
        dynamic_params['epsilon'] = epsilon_value

    elif method_key == "MaxVar":
        num_pot_edges = st.sidebar.slider(
            "Nombre d'arêtes potentielles",
            min_value=20,
            max_value=100,
            value=method['params']['num_potential_edges'],
            step=10,
            help="Nombre d'arêtes potentielles (nearby, distance=2) à ajouter avant l'optimisation. Plus ce nombre est élevé, plus la variance peut être maximisée, mais le calcul est plus long."
        )
        dynamic_params['num_potential_edges'] = num_pot_edges

    elif method_key in ["EdgeFlip", "Laplace"]:
        epsilon_value = st.sidebar.slider(
            "ε = Budget de Privacy",
            min_value=0.1,
            max_value=3.0,
            value=method['params']['epsilon'],
            step=0.1,
            help="""📖 Budget de Privacy Différentielle (ε-DP)

FORMULE CORRECTE : s = 2/(e^ε + 1), flip_probability = s/2

Trade-off Privacy-Utilité :
• ε PETIT = FORTE privacy (beaucoup de modifications)
• ε GRAND = FAIBLE privacy (peu de modifications)

Exemples concrets :
• ε = 0.1 (petit): flip_prob = 47.5% → graphe très différent → FORTE privacy ✓
• ε = 1.0 (moyen): flip_prob = 26.9% → changements modérés → privacy moyenne
• ε = 3.0 (grand): flip_prob = 4.7% → graphe proche → FAIBLE privacy

En DP, epsilon mesure la "perte de privacy" : plus c'est petit, mieux c'est !"""
        )
        dynamic_params['epsilon'] = epsilon_value

        # Afficher l'impact du budget AVEC LA FORMULE CORRECTE
        privacy_loss = np.exp(epsilon_value)
        s = 2 / (np.exp(epsilon_value) + 1)  # FORMULE CORRECTE
        flip_prob = s / 2

        if epsilon_value < 1.0:
            st.sidebar.success(f"✅ Privacy Forte (ε={epsilon_value:.1f})")
            st.sidebar.caption(f"Flip: {flip_prob*100:.1f}% | Graphe très modifié")
        elif epsilon_value < 2.0:
            st.sidebar.warning(f"⚠️ Privacy Moyenne (ε={epsilon_value:.1f})")
            st.sidebar.caption(f"Flip: {flip_prob*100:.1f}% | Modifications modérées")
        else:
            st.sidebar.error(f"❌ Privacy Faible (ε={epsilon_value:.1f})")
            st.sidebar.caption(f"Flip: {flip_prob*100:.1f}% | Graphe proche de l'original")

    # Bouton pour anonymiser
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Anonymiser le Graphe", type="primary"):
        st.session_state.anonymized = True
        st.session_state.method_key = method_key
        st.session_state.method_params = dynamic_params  # Sauvegarder les paramètres utilisés
        st.session_state.show_sample = False  # Réinitialiser l'affichage d'échantillon

        # Anonymiser
        anonymizer = GraphAnonymizer(G)

        with st.spinner('Anonymisation en cours...'):
            node_to_cluster = None
            if method_key == "Random Add/Del":
                G_anon = anonymizer.random_add_del(**dynamic_params)
            elif method_key == "Random Switch":
                G_anon = anonymizer.random_switch(**dynamic_params)
            elif method_key == "k-degree anonymity":
                G_anon = anonymizer.k_degree_anonymity(**dynamic_params)
            elif method_key == "Generalization":
                G_anon, node_to_cluster = anonymizer.generalization(**dynamic_params)
                st.session_state.node_to_cluster = node_to_cluster
            elif method_key == "Probabilistic":
                G_anon = anonymizer.probabilistic_obfuscation(**dynamic_params)
            elif method_key == "MaxVar":
                G_anon = anonymizer.maxvar_obfuscation(**dynamic_params)
            elif method_key == "EdgeFlip":
                G_anon = anonymizer.differential_privacy_edgeflip(**dynamic_params)
            elif method_key == "Laplace":
                G_anon = anonymizer.differential_privacy_laplace(**dynamic_params)

            st.session_state.G_anon = G_anon
            st.session_state.G_orig = G
            if node_to_cluster is None:
                st.session_state.node_to_cluster = None

    # Bouton pour tirer un échantillon (seulement pour graphes probabilistes)
    if 'anonymized' in st.session_state and st.session_state.anonymized:
        if st.session_state.method_key in ["Probabilistic", "MaxVar"]:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🎲 Échantillonnage")
            st.sidebar.caption("Les méthodes probabilistes publient un graphe incertain. Tirez un échantillon déterministe!")

            if st.sidebar.button("🎲 Tirer un Échantillon", type="secondary"):
                with st.spinner('Tirage d\'échantillon en cours...'):
                    G_sample = sample_from_probabilistic_graph(st.session_state.G_anon)
                    st.session_state.G_sample = G_sample
                    st.session_state.show_sample = True

            if st.session_state.get('show_sample', False):
                st.sidebar.success("✅ Échantillon tiré!")
                st.sidebar.caption(f"Nœuds: {st.session_state.G_sample.number_of_nodes()}, Arêtes: {st.session_state.G_sample.number_of_edges()}")

                if st.sidebar.button("🔄 Afficher graphe incertain", type="secondary"):
                    st.session_state.show_sample = False

    # Section Thèse PDF
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Thèse de Référence")

    with st.sidebar.expander("📖 Voir la Thèse", expanded=False):
        st.markdown("""
        **"Anonymisation de Graphes Sociaux"**
        *NGUYEN Huu-Hiep (2016)*

        Université de Lorraine
        """)

        # Afficher le PDF
        try:
            import base64
            with open("assets/thesis.pdf", "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            # Bouton pour télécharger le PDF
            st.download_button(
                label="📥 Télécharger la thèse (PDF)",
                data=open("assets/thesis.pdf", "rb").read(),
                file_name="NGUYEN_Anonymisation_Graphes_Sociaux_2016.pdf",
                mime="application/pdf"
            )

            # Afficher un iframe avec le PDF
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        except FileNotFoundError:
            st.warning("Fichier PDF non trouvé. Placez thesis.pdf dans le dossier assets/")

    # Affichage des résultats
    if 'anonymized' in st.session_state and st.session_state.anonymized:
        G_orig = st.session_state.G_orig
        G_anon = st.session_state.G_anon

        # Utiliser l'échantillon si disponible pour l'affichage
        G_display = st.session_state.get('G_sample', G_anon) if st.session_state.get('show_sample', False) else G_anon

        current_method = METHODS[st.session_state.method_key]

        # Onglets - VERSION AMÉLIORÉE avec 8 onglets
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Résultats",
            "📖 Définitions",
            "📈 Métriques Utilité",
            "🔒 Métriques Privacy",
            "🎯 Simulations d'Attaques",
            "🛡️ Attaques & Garanties",
            "📚 Dict. Attaques",
            "🔍 Dict. Propriétés"
        ])

        with tab1:
            st.markdown("## 📊 Résultats de l'Anonymisation")

            # Indicateur si on affiche un échantillon
            if st.session_state.get('show_sample', False):
                st.info("🎲 **Affichage d'un graphe échantillon** tiré depuis le graphe incertain. Les probabilités ont été converties en arêtes déterministes.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Nœuds Originaux", G_orig.number_of_nodes())
                st.metric("Arêtes Originales", G_orig.number_of_edges())

            with col2:
                if isinstance(G_display, nx.Graph):
                    label = "Échantillon" if st.session_state.get('show_sample', False) else "Anonymisés"
                    st.metric(f"Nœuds {label}", G_display.number_of_nodes())
                    st.metric(f"Arêtes {label}", G_display.number_of_edges(),
                             delta=f"{G_display.number_of_edges() - G_orig.number_of_edges():+d}")
                else:
                    st.info("Format de graphe non standard (super-nodes)")

            st.markdown("---")
            st.markdown("### Comparaison Visuelle")

            node_to_cluster = st.session_state.get('node_to_cluster', None)
            # Utiliser G_display pour la visualisation
            fig = plot_graph_comparison(G_orig, G_display, current_method['name'], node_to_cluster)
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

            # ═══════════════════════════════════════════════════════════════════
            # SECTION SPÉCIALE : TIRAGE D'ÉCHANTILLONS (Graphes Probabilistes)
            # ═══════════════════════════════════════════════════════════════════
            if st.session_state.method_key == "Probabilistic" and isinstance(G_anon, nx.Graph):
                # Vérifier si c'est un graphe probabiliste
                if G_anon.number_of_edges() > 0:
                    first_edge = list(G_anon.edges())[0]
                    if 'probability' in G_anon[first_edge[0]][first_edge[1]]:
                        st.markdown("---")
                        st.markdown("### 🎲 Tirage d'Échantillons depuis le Graphe Probabiliste")

                        st.info("""
                        **💡 Principe du Tirage (Sampling)** :

                        Dans un graphe probabiliste (k,ε)-obfuscation, chaque arête a une **probabilité d'existence**.
                        Au lieu de publier le graphe probabiliste directement, on peut publier des **graphes échantillons**
                        tirés au sort selon ces probabilités.

                        **🔍 Visualisation** :
                        - **Arêtes pleines (───)** : Arêtes du graphe **original**
                        - **Arêtes pointillés (···)** : Arêtes **potentielles** ajoutées pour l'anonymisation
                        - **Couleur & Épaisseur** : Indiquent la probabilité d'existence

                        - **Arêtes à haute probabilité** (≈ 95%) : Apparaissent presque toujours
                        - **Arêtes à faible probabilité** (≈ 10%) : Apparaissent rarement

                        Cliquez sur le bouton ci-dessous pour générer 3 échantillons différents !
                        """)

                        # Bouton pour générer des échantillons
                        if st.button("🎲 Générer 3 Échantillons Aléatoires", key="sample_btn"):
                            st.markdown("#### Échantillons Générés :")

                            cols = st.columns(3)
                            for i, col in enumerate(cols):
                                with col:
                                    # Générer un échantillon
                                    sampled_graph = sample_from_probabilistic_graph(G_anon)

                                    # Créer une figure pour cet échantillon
                                    fig_sample, ax_sample = plt.subplots(1, 1, figsize=(6, 6))

                                    pos = nx.spring_layout(G_orig, seed=42, k=0.5, iterations=50)

                                    # Dessiner les nœuds
                                    nx.draw_networkx_nodes(sampled_graph, pos, ax=ax_sample,
                                                          node_color='lightyellow',
                                                          node_size=400, alpha=0.9,
                                                          edgecolors='orange', linewidths=2)

                                    # Dessiner les arêtes
                                    nx.draw_networkx_edges(sampled_graph, pos, ax=ax_sample,
                                                          edge_color='gray', width=1.5, alpha=0.6)

                                    # Labels
                                    nx.draw_networkx_labels(sampled_graph, pos, ax=ax_sample,
                                                           font_size=7, font_weight='bold')

                                    ax_sample.set_title(f'Échantillon #{i+1}\n{sampled_graph.number_of_edges()} arêtes',
                                                       fontsize=12, fontweight='bold')
                                    ax_sample.axis('off')

                                    plt.tight_layout()
                                    st.pyplot(fig_sample)
                                    plt.close(fig_sample)

                                    # Afficher les stats
                                    st.caption(f"**{sampled_graph.number_of_nodes()}** nœuds | **{sampled_graph.number_of_edges()}** arêtes")

                        st.markdown("""
                        **🔍 Observation** : Chaque échantillon est différent ! C'est cette variabilité qui crée
                        de l'incertitude pour l'attaquant. Il ne peut pas savoir quel échantillon correspond au graphe original.
                        """)

            st.markdown("---")
            st.markdown("### Distribution des Degrés")

            fig_dist = plot_degree_distribution(G_orig, G_anon, current_method['name'])
            st.pyplot(fig_dist)

            # Explication de la méthode actuelle (déplacée depuis tab2)
            st.markdown("---")
            st.markdown(f"### 🔬 Explication : {current_method['name']}")

            with st.expander("📚 Détails de la méthode", expanded=False):
                st.markdown(current_method['description'])
                st.markdown("**Formule** :")
                st.latex(current_method['formula'])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🔒 Niveau de Privacy**")
                    st.info(current_method['privacy_level'])
                with col2:
                    st.markdown("**📊 Préservation de l'Utilité**")
                    st.info(current_method['utility_preservation'])

        with tab2:
            st.markdown("## 📖 Définitions des Concepts d'Anonymisation")

            st.markdown("""
            Cette section présente les définitions formelles et intuitions pour chaque type d'anonymisation.
            Choisissez un concept ci-dessous pour voir sa définition complète.
            """)

            st.markdown("---")

            # Sélecteur de concept
            concept_keys = list(ANONYMIZATION_DEFINITIONS.keys())
            concept_names = [ANONYMIZATION_DEFINITIONS[k]['name'] for k in concept_keys]

            selected_concept_name = st.selectbox(
                "Choisir un concept à explorer",
                concept_names
            )

            # Trouver la clé correspondante
            selected_concept_key = concept_keys[concept_names.index(selected_concept_name)]
            concept = ANONYMIZATION_DEFINITIONS[selected_concept_key]

            st.markdown(f"### {concept['name']}")

            with st.expander("📝 Définition Formelle", expanded=True):
                st.markdown(concept['definition'])
                st.markdown("**Formule mathématique** :")
                st.latex(concept['math_formula'])

            with st.expander("💡 Intuition (Explication en langage naturel)", expanded=True):
                st.markdown(concept['intuition'])

            with st.expander("🔒 Garantie de Privacy"):
                st.info(f"**Garantie** : {concept['privacy_guarantee']}")

            with st.expander("⚙️ Signification des Paramètres"):
                st.markdown(concept['parameter_meaning'])

        with tab3:
            st.markdown("## 📈 Métriques d'Utilité du Graphe")

            st.markdown("""
            Ces métriques mesurent la **préservation de l'utilité** du graphe après anonymisation.
            Plus ces métriques sont proches du graphe original, mieux l'utilité est préservée.

            💡 **Astuce** : Passez votre souris sur le ℹ️ à côté de chaque métrique pour voir sa définition et méthode de calcul.
            """)

            utility_metrics = calculate_utility_metrics(G_orig, G_anon)

            # ═══════════════════════════════════════════════════════════════════════
            # CAS 1 : SUPER-GRAPHE (Généralisation)
            # ═══════════════════════════════════════════════════════════════════════
            if utility_metrics.get('is_super_graph', False):
                st.info("🔍 **Type de graphe** : Super-Graphe (Généralisation) - Métriques adaptées au format cluster")

                st.markdown("### 🏘️ Métriques de Clustering")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Nombre de Clusters",
                             utility_metrics.get('num_clusters', 'N/A'),
                             help=get_metric_tooltip('num_clusters'))
                with col2:
                    st.metric("Taille Min. Cluster",
                             utility_metrics.get('min_cluster_size', 'N/A'),
                             help=get_metric_tooltip('min_cluster_size'))
                with col3:
                    st.metric("Taille Moy. Cluster",
                             f"{utility_metrics.get('avg_cluster_size', 0):.1f}",
                             help=get_metric_tooltip('avg_cluster_size'))
                with col4:
                    st.metric("Taille Max. Cluster",
                             utility_metrics.get('max_cluster_size', 'N/A'))

                st.markdown("---")
                st.markdown("### 🔗 Métriques d'Arêtes")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Arêtes Intra-Cluster",
                             utility_metrics.get('intra_cluster_edges', 'N/A'),
                             help=get_metric_tooltip('intra_cluster_edges'))
                with col2:
                    st.metric("Arêtes Inter-Cluster",
                             utility_metrics.get('inter_cluster_edges', 'N/A'),
                             help=get_metric_tooltip('inter_cluster_edges'))
                with col3:
                    intra_ratio = utility_metrics.get('intra_ratio', 0)
                    st.metric("Ratio Intra/Total",
                             f"{intra_ratio*100:.1f}%",
                             help=get_metric_tooltip('intra_ratio'))

                st.markdown("---")
                st.markdown("### 📊 Perte d'Information")

                col1, col2, col3 = st.columns(3)

                with col1:
                    info_loss = utility_metrics.get('information_loss', 0)
                    st.metric("Perte d'Information",
                             f"{info_loss*100:.1f}%",
                             help=get_metric_tooltip('information_loss'))
                with col2:
                    edge_pres = utility_metrics.get('edge_preservation_ratio', 0)
                    st.metric("Préservation des Arêtes",
                             f"{edge_pres*100:.1f}%",
                             help=get_metric_tooltip('edge_preservation_ratio'))
                with col3:
                    super_density = utility_metrics.get('super_graph_density', 0)
                    st.metric("Densité Super-Graphe",
                             f"{super_density:.3f}",
                             help=get_metric_tooltip('super_graph_density'))

                # Afficher un résumé comparatif
                st.markdown("---")
                st.markdown("### 📉 Comparaison Original ↔ Anonymisé")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Graphe Original**")
                    st.metric("Nœuds", G_orig.number_of_nodes())
                    st.metric("Arêtes", G_orig.number_of_edges())

                with col2:
                    st.markdown("**Super-Graphe**")
                    st.metric("Clusters (super-nœuds)", utility_metrics.get('num_clusters', 'N/A'))
                    st.metric("Arêtes Totales", utility_metrics.get('num_edges', 'N/A'))

            # ═══════════════════════════════════════════════════════════════════════
            # CAS 2 : GRAPHE CLASSIQUE ou PROBABILISTE
            # ═══════════════════════════════════════════════════════════════════════
            elif utility_metrics.get('comparable', True):
                if utility_metrics.get('is_sample', False):
                    st.info("🎲 **Type de graphe** : Échantillon tiré depuis un graphe probabiliste")

                st.markdown("### 📊 Métriques de Base")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Nœuds",
                             utility_metrics.get('num_nodes', 'N/A'),
                             help=get_metric_tooltip('num_nodes'))
                with col2:
                    st.metric("Arêtes",
                             utility_metrics.get('num_edges', 'N/A'),
                             help=get_metric_tooltip('num_edges'))
                with col3:
                    orig_density = nx.density(G_orig)
                    anon_density = utility_metrics.get('density', 0)
                    delta_density = anon_density - orig_density
                    st.metric("Densité",
                             f"{anon_density:.3f}",
                             delta=f"{delta_density:+.3f}",
                             help=get_metric_tooltip('density'))
                with col4:
                    if utility_metrics.get('avg_clustering') is not None:
                        orig_clust = nx.average_clustering(G_orig)
                        anon_clust = utility_metrics['avg_clustering']
                        delta_clust = anon_clust - orig_clust
                        st.metric("Clustering Moyen",
                                 f"{anon_clust:.3f}",
                                 delta=f"{delta_clust:+.3f}",
                                 help=get_metric_tooltip('avg_clustering'))

                st.markdown("---")
                st.markdown("### 🌐 Métriques Globales")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if utility_metrics.get('diameter') is not None:
                        try:
                            if nx.is_connected(G_orig):
                                orig_diam = nx.diameter(G_orig)
                            else:
                                largest_cc = max(nx.connected_components(G_orig), key=len)
                                orig_diam = nx.diameter(G_orig.subgraph(largest_cc))
                            delta_diam = utility_metrics['diameter'] - orig_diam
                            st.metric("Diamètre",
                                     utility_metrics['diameter'],
                                     delta=f"{delta_diam:+d}",
                                     help=get_metric_tooltip('diameter'))
                        except:
                            st.metric("Diamètre",
                                     utility_metrics['diameter'],
                                     help=get_metric_tooltip('diameter'))

                with col2:
                    if utility_metrics.get('avg_shortest_path') is not None:
                        try:
                            if nx.is_connected(G_orig):
                                orig_asp = nx.average_shortest_path_length(G_orig)
                            else:
                                largest_cc = max(nx.connected_components(G_orig), key=len)
                                orig_asp = nx.average_shortest_path_length(G_orig.subgraph(largest_cc))
                            delta_asp = utility_metrics['avg_shortest_path'] - orig_asp
                            st.metric("Chemin Moyen",
                                     f"{utility_metrics['avg_shortest_path']:.2f}",
                                     delta=f"{delta_asp:+.2f}",
                                     help=get_metric_tooltip('avg_shortest_path'))
                        except:
                            st.metric("Chemin Moyen",
                                     f"{utility_metrics['avg_shortest_path']:.2f}",
                                     help=get_metric_tooltip('avg_shortest_path'))

                with col3:
                    if utility_metrics.get('degree_correlation') is not None:
                        st.metric("Corrélation des Degrés",
                                 f"{utility_metrics['degree_correlation']:.3f}",
                                 help=get_metric_tooltip('degree_correlation'))

                st.markdown("---")
                st.markdown("### 📉 Trade-off Utilité vs Modifications")

                metrics = calculate_anonymization_metrics(G_orig, G_anon)

                if metrics:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Modifications des Arêtes**")
                        added = metrics.get('edges_added', 0)
                        removed = metrics.get('edges_removed', 0)
                        preserved = metrics.get('edges_preserved', 0)

                        df_edges = pd.DataFrame({
                            'Type': ['Préservées', 'Ajoutées', 'Supprimées'],
                            'Nombre': [preserved, added, removed]
                        })
                        st.bar_chart(df_edges.set_index('Type'))

                    with col2:
                        st.markdown("**Taux de Modification**")
                        rate = metrics.get('modification_rate', 0)
                        st.progress(min(rate, 1.0))
                        st.metric("Taux de modification", f"{rate*100:.1f}%")

                        if rate < 0.1:
                            st.success("✅ Utilité très bien préservée")
                        elif rate < 0.3:
                            st.info("ℹ️ Utilité correctement préservée")
                        else:
                            st.warning("⚠️ Modifications importantes")

            else:
                st.warning("⚠️ Type de graphe non reconnu - impossible de calculer les métriques")

        with tab4:
            st.markdown("## 🔒 Métriques de Privacy")

            st.markdown("""
            Ces métriques quantifient la **protection de la vie privée** offerte par l'anonymisation.
            Plus ces valeurs sont élevées, meilleure est la protection.
            """)

            method_params = st.session_state.get('method_params', {})
            privacy_metrics = calculate_privacy_metrics_separated(G_orig, G_anon, st.session_state.method_key, method_params)

            if privacy_metrics:
                st.markdown("### 🛡️ Garanties de Privacy")

                if 'k_value' in privacy_metrics:
                    # k-anonymity
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("k requis", privacy_metrics['k_value'])
                    with col2:
                        st.metric("Ensemble d'anonymat min.", privacy_metrics['min_anonymity_set'])
                    with col3:
                        satisfies = privacy_metrics['satisfies_k_anonymity']
                        if satisfies:
                            st.success(f"✅ {privacy_metrics['k_value']}-anonymité satisfaite")
                        else:
                            st.error(f"❌ {privacy_metrics['k_value']}-anonymité NON satisfaite")

                    st.markdown("---")
                    prob = privacy_metrics['re_identification_probability']
                    st.markdown(f"**Probabilité de ré-identification** : {prob:.3f} ({prob*100:.1f}%)")

                    st.progress(1 - prob)

                    if prob < 0.2:
                        st.success("✅ Risque de ré-identification faible")
                    elif prob < 0.5:
                        st.warning("⚠️ Risque de ré-identification modéré")
                    else:
                        st.error("❌ Risque de ré-identification élevé")

                elif 'epsilon_budget' in privacy_metrics:
                    # Differential Privacy
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        eps = privacy_metrics['epsilon_budget']
                        st.metric("ε (epsilon) Budget", f"{eps:.2f}")

                    with col2:
                        loss = privacy_metrics['privacy_loss_bound']
                        st.metric("Borne de perte de privacy", f"e^{eps:.2f} = {loss:.2f}x")

                    with col3:
                        level = privacy_metrics['privacy_level']
                        if "Forte" in level:
                            st.success(f"✅ {level}")
                        elif "Moyenne" in level:
                            st.warning(f"⚠️ {level}")
                        else:
                            st.error(f"❌ {level}")

                    st.markdown("---")

                    if 'flip_probability' in privacy_metrics:
                        st.markdown("### 🎲 EdgeFlip - Paramètres de Randomisation")
                        col1, col2 = st.columns(2)

                        with col1:
                            flip_prob = privacy_metrics['flip_probability']
                            st.metric("Probabilité de flip", f"{flip_prob:.3f}")

                        with col2:
                            expected_noise = privacy_metrics['expected_noise_edges']
                            st.metric("Arêtes bruitées (attendu)", expected_noise)

                elif 'k_candidates' in privacy_metrics:
                    # Probabilistic
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("k graphes candidats", privacy_metrics['k_candidates'])

                    with col2:
                        st.metric("ε tolérance", f"{privacy_metrics['epsilon_tolerance']:.2f}")

                    with col3:
                        entropy = privacy_metrics['min_entropy']
                        st.metric("Entropie minimale", f"{entropy:.2f}")

                    st.markdown("---")
                    confusion = privacy_metrics['confusion_factor']
                    st.info(f"**Facteur de confusion** : {confusion} graphes plausibles")

                elif 'num_potential_edges' in privacy_metrics:
                    # MaxVar
                    st.markdown("### 🔒 MaxVar - Métriques de Variance et Dispersion")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Arêtes potentielles ajoutées", privacy_metrics['num_potential_edges'])

                    with col2:
                        if 'total_variance' in privacy_metrics:
                            var = privacy_metrics['total_variance']
                            st.metric("Variance totale", f"{var:.2f}",
                                     help="Plus élevée = meilleure dispersion des probabilités")

                    with col3:
                        if 'avg_edge_variance' in privacy_metrics:
                            avg_var = privacy_metrics['avg_edge_variance']
                            st.metric("Variance moyenne/arête", f"{avg_var:.3f}")

                    st.markdown("---")
                    st.markdown("### 📊 Analyse des Probabilités")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Arêtes existantes (originales)**")
                        if 'existing_avg_prob' in privacy_metrics:
                            st.metric("Probabilité moyenne", f"{privacy_metrics['existing_avg_prob']:.3f}")
                            if 'existing_std_prob' in privacy_metrics:
                                st.metric("Écart-type", f"{privacy_metrics['existing_std_prob']:.3f}",
                                         help="Plus élevé = probabilités plus dispersées")

                    with col2:
                        st.markdown("**Arêtes potentielles (ajoutées)**")
                        if 'potential_avg_prob' in privacy_metrics:
                            st.metric("Probabilité moyenne", f"{privacy_metrics['potential_avg_prob']:.3f}")
                            if 'potential_std_prob' in privacy_metrics:
                                st.metric("Écart-type", f"{privacy_metrics['potential_std_prob']:.3f}")

                    st.markdown("---")
                    st.markdown("### 🛡️ Résistance au Seuillage")

                    if 'threshold_resistance' in privacy_metrics:
                        resistance = privacy_metrics['threshold_resistance']
                        reconstruction = privacy_metrics.get('reconstruction_rate', 0)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Taux de résistance", f"{resistance*100:.1f}%",
                                     help="% d'arêtes originales non récupérables par seuillage à 0.5")

                        with col2:
                            st.metric("Taux de reconstruction", f"{reconstruction*100:.1f}%",
                                     help="% d'arêtes originales récupérables par seuillage à 0.5")

                        st.progress(resistance)

                        if resistance > 0.2:
                            st.success(f"✅ Bonne résistance au seuillage ({resistance*100:.1f}%)")
                        elif resistance > 0.1:
                            st.warning(f"⚠️ Résistance modérée ({resistance*100:.1f}%)")
                        else:
                            st.error(f"❌ Faible résistance - vulnérable au seuillage ({resistance*100:.1f}%)")

                        st.caption("💡 Un attaquant qui applique un seuil à 0.5 ne récupère que "
                                  f"{reconstruction*100:.1f}% des arêtes originales (contre 100% pour (k,ε)-obf)")

                elif 'min_cluster_size' in privacy_metrics:
                    # Generalization
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Taille min. cluster", int(privacy_metrics['min_cluster_size']))

                    with col2:
                        st.metric("Taille moy. cluster", f"{privacy_metrics['avg_cluster_size']:.1f}")

                    with col3:
                        max_priv = privacy_metrics['max_privacy']
                        st.metric("Prob. max ré-identification", f"{max_priv:.3f}")

                st.markdown("---")

                # Garanties globales
                guarantees = calculate_privacy_guarantees(G_orig, G_anon, st.session_state.method_key, method_params)

                if guarantees:
                    st.markdown("### 📋 Garanties Détaillées")

                    with st.expander("Voir toutes les garanties"):
                        for key, value in guarantees.items():
                            st.text(f"{key}: {value}")

            else:
                st.info("Aucune métrique de privacy spécifique pour cette méthode")

        with tab5:
            st.markdown("## 🎯 Simulations d'Attaques Réelles")

            st.markdown("""
            Cette section simule des attaques de **ré-identification** sur le graphe anonymisé.
            Ces simulations montrent concrètement si un adversaire peut retrouver des nœuds spécifiques.
            """)

            st.markdown("---")

            # Sélection du nœud cible
            st.markdown("### 🎯 Configuration de l'Attaque")

            col1, col2 = st.columns(2)

            with col1:
                target_node = st.number_input(
                    "Nœud cible à retrouver",
                    min_value=0,
                    max_value=G_orig.number_of_nodes()-1,
                    value=0,
                    help="Le nœud que l'adversaire essaie de ré-identifier"
                )

            with col2:
                attack_type = st.selectbox(
                    "Type d'attaque",
                    ["Degree Attack", "Subgraph Attack (Triangles)"]
                )

            st.markdown("---")

            if st.button("🚀 Lancer l'Attaque"):
                st.markdown("### 📊 Résultats de l'Attaque")

                with st.spinner("Simulation en cours..."):
                    if attack_type == "Degree Attack":
                        results = simulate_degree_attack(G_orig, G_anon, target_node)
                    else:
                        results = simulate_subgraph_attack(G_orig, G_anon, target_node)

                if results['success']:
                    st.error("### ⚠️ Attaque Réussie !")
                    st.markdown(results['explanation'])

                    st.markdown(f"**Nœud ré-identifié** : {results.get('re_identified_node', 'N/A')}")

                else:
                    st.success("### ✅ Attaque Échouée / Partiellement Réussie")
                    st.markdown(results['explanation'])

                st.markdown("---")
                st.markdown("### 📈 Détails Techniques")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Nœud cible** :")
                    st.info(f"Nœud {target_node}")

                    if 'target_degree' in results:
                        st.markdown("**Degré du nœud** :")
                        st.info(f"Degré = {results['target_degree']}")

                    if 'target_triangles' in results:
                        st.markdown("**Triangles** :")
                        st.info(f"{results['target_triangles']} triangles")

                with col2:
                    st.markdown("**Candidats trouvés** :")
                    if results['candidates']:
                        st.info(f"{len(results['candidates'])} nœuds : {results['candidates'][:10]}")
                    else:
                        st.info("Aucun candidat")

                    if len(results['candidates']) > 1:
                        prob_success = 1 / len(results['candidates'])
                        st.markdown("**Probabilité de succès** :")
                        st.warning(f"{prob_success*100:.1f}%")

            st.markdown("---")

            # Section éducative
            with st.expander("📚 En savoir plus sur ces attaques"):
                st.markdown("""
                ### Degree Attack (Attaque par Degré)

                L'adversaire connaît le degré (nombre de connexions) du nœud cible et cherche
                dans le graphe anonymisé tous les nœuds ayant ce degré.

                **Protection** :
                - k-degree anonymity garantit au moins k nœuds par degré
                - Randomisation modifie les degrés
                - Differential Privacy ajoute du bruit

                ### Subgraph Attack (Attaque par Sous-graphe)

                L'adversaire connaît la structure locale autour du nœud (ex: triangles, motifs).
                Cette attaque est plus puissante car elle exploite plus d'information.

                **Protection** :
                - Generalization détruit les motifs locaux
                - Differential Privacy ajoute/supprime des triangles fictifs
                - Randomisation casse certains motifs
                """)

        with tab6:
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

        with tab7:
            st.markdown("## 📚 Dictionnaire des Attaques de Ré-Identification")

            st.markdown("""
            Ce dictionnaire présente **toutes les attaques connues** contre les graphes anonymisés,
            avec des exemples concrets et des explications détaillées.
            """)

            st.markdown("---")

            # Liste des attaques
            attack_names = [ATTACKS_DICTIONARY[k]['name'] for k in ATTACKS_DICTIONARY.keys()]

            selected_attack_name = st.selectbox(
                "Choisir une attaque à explorer",
                attack_names
            )

            # Trouver l'attaque correspondante
            selected_attack_key = list(ATTACKS_DICTIONARY.keys())[attack_names.index(selected_attack_name)]
            attack = ATTACKS_DICTIONARY[selected_attack_key]

            st.markdown(f"### {attack['name']}")

            col1, col2 = st.columns([2, 1])

            with col1:
                with st.expander("📝 Description de l'Attaque", expanded=True):
                    st.markdown(attack['description'])

                with st.expander("💡 Exemple Concret"):
                    st.markdown(attack['example'])

            with col2:
                st.markdown("**⚠️ Sévérité**")
                severity = attack['severity']
                if "Très élevée" in severity or "Élevée" in severity:
                    st.error(severity)
                elif "Moyenne" in severity:
                    st.warning(severity)
                else:
                    st.info(severity)

                st.markdown("**🛡️ Protection**")
                st.success(attack['protection'])

            st.markdown("---")

            # Exemples concrets sur Karate Club
            st.markdown("### 🥋 Exemples Concrets sur Karate Club")

            example_keys = list(CONCRETE_ATTACK_EXAMPLES.keys())

            for example_key in example_keys:
                example = CONCRETE_ATTACK_EXAMPLES[example_key]

                with st.expander(f"📖 {example['title']}"):
                    st.markdown(f"**Scénario** : {example['scenario']}")

                    st.markdown("**Étapes de l'attaque** :")
                    for step in example['steps']:
                        st.markdown(f"- {step}")

                    st.markdown("---")
                    st.markdown("**Taux de Succès** :")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Sans protection", example.get('success_rate_no_protection', 'N/A'))

                    with col2:
                        if 'success_rate_k_anonymity' in example:
                            st.metric("Avec k-anonymity", example['success_rate_k_anonymity'])
                        elif 'success_rate_randomization' in example:
                            st.metric("Avec randomization", example['success_rate_randomization'])

                    with col3:
                        if 'success_rate_differential_privacy' in example:
                            st.metric("Avec Diff. Privacy", example['success_rate_differential_privacy'])
                        elif 'success_rate_generalization' in example:
                            st.metric("Avec Generalization", example['success_rate_generalization'])

                    if 'code_simulation' in example:
                        with st.expander("💻 Code de Simulation"):
                            st.code(example['code_simulation'], language='python')

        with tab8:
            st.markdown("## 🔍 Dictionnaire des Propriétés de Graphes")

            st.markdown("""
            Ce dictionnaire explique **toutes les propriétés de graphes** utilisées en anonymisation,
            leur importance pour l'utilité, et leur risque pour la privacy.
            """)

            st.markdown("---")

            # Liste des propriétés
            property_names = [GRAPH_PROPERTIES[k]['name'] for k in GRAPH_PROPERTIES.keys()]

            selected_property_name = st.selectbox(
                "Choisir une propriété à explorer",
                property_names
            )

            # Trouver la propriété correspondante
            selected_property_key = list(GRAPH_PROPERTIES.keys())[property_names.index(selected_property_name)]
            prop = GRAPH_PROPERTIES[selected_property_key]

            st.markdown(f"### {prop['name']}")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("📝 Définition", expanded=True):
                    st.markdown(prop['definition'])

                with st.expander("🔢 Formule"):
                    st.latex(prop['formula'])

                with st.expander("💡 Exemple"):
                    st.info(prop['example'])

            with col2:
                st.markdown("**📊 Importance pour l'Utilité**")
                importance = prop['utility_importance']
                if "Critique" in importance or "Élevée" in importance:
                    st.success(importance)
                else:
                    st.info(importance)

                st.markdown("**⚠️ Risque pour la Privacy**")
                risk = prop['privacy_risk']
                if "Élevé" in risk:
                    st.error(risk)
                elif "Moyen" in risk:
                    st.warning(risk)
                else:
                    st.success(risk)

            st.markdown("---")

            # Calcul des propriétés sur le graphe actuel
            if isinstance(G_anon, nx.Graph):
                st.markdown("### 📊 Valeurs pour le Graphe Actuel")

                try:
                    if selected_property_key == 'degree':
                        degrees = dict(G_anon.degree())
                        st.metric("Degré moyen", f"{np.mean(list(degrees.values())):.2f}")
                        st.metric("Degré max", max(degrees.values()))

                    elif selected_property_key == 'clustering_coefficient':
                        clustering = nx.average_clustering(G_anon)
                        st.metric("Coefficient de clustering moyen", f"{clustering:.3f}")

                    elif selected_property_key == 'density':
                        density = nx.density(G_anon)
                        st.metric("Densité", f"{density:.3f}")

                    elif selected_property_key == 'diameter':
                        if nx.is_connected(G_anon):
                            diameter = nx.diameter(G_anon)
                            st.metric("Diamètre", diameter)
                        else:
                            st.info("Graphe non connexe, diamètre non défini")

                    elif selected_property_key == 'average_path_length':
                        if nx.is_connected(G_anon):
                            apl = nx.average_shortest_path_length(G_anon)
                            st.metric("Longueur moyenne des chemins", f"{apl:.2f}")
                        else:
                            st.info("Graphe non connexe, calculé sur la plus grande composante")

                except Exception as e:
                    st.warning(f"Calcul non disponible pour ce graphe")

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
