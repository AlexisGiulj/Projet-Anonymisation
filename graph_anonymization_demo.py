"""
Application de démonstration des 5 types de méthodes d'anonymisation de graphes
Basée sur la thèse "Anonymisation de Graphes Sociaux" par NGUYEN Huu-Hiep

Les 5 types de méthodes:
1. Anonymisation par Randomisation (Random Add/Del, Random Switch)
2. K-anonymisation (k-degree anonymity)
3. Anonymisation par Généralisation (super-nodes)
4. Approches Probabilistes ((k,ε)-obfuscation)
5. Privacy Différentielle (EdgeFlip, Laplace mechanism)
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random
from copy import deepcopy

# Configuration pour l'affichage
plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('seaborn-v0_8-darkgrid')


class GraphAnonymizer:
    """Classe pour anonymiser des graphes sociaux avec différentes méthodes"""

    def __init__(self, graph):
        """
        Args:
            graph: Un graphe NetworkX
        """
        self.original_graph = graph.copy()
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()

    # ========================================================================
    # MÉTHODE 1: ANONYMISATION PAR RANDOMISATION
    # ========================================================================

    def random_add_del(self, k):
        """
        Random Add/Del: Ajoute k fausses arêtes puis supprime k vraies arêtes

        Args:
            k: nombre d'arêtes à modifier

        Returns:
            Graphe anonymisé
        """
        G = self.original_graph.copy()

        # Ajouter k fausses arêtes
        added = 0
        attempts = 0
        max_attempts = k * 100

        while added < k and attempts < max_attempts:
            u, v = random.sample(list(G.nodes()), 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1
            attempts += 1

        # Supprimer k vraies arêtes
        edges = list(self.original_graph.edges())
        if len(edges) >= k:
            edges_to_remove = random.sample(edges, k)
            for u, v in edges_to_remove:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)

        return G

    def random_switch(self, k):
        """
        Random Switch: Échange k paires d'arêtes pour préserver les degrés

        Args:
            k: nombre d'échanges à effectuer

        Returns:
            Graphe anonymisé
        """
        G = self.original_graph.copy()

        for _ in range(k):
            edges = list(G.edges())
            if len(edges) < 2:
                break

            # Choisir deux arêtes aléatoires (u,w) et (v,x)
            (u, w), (v, x) = random.sample(edges, 2)

            # Vérifier si on peut faire l'échange
            if u != v and u != x and w != v and w != x:
                if not G.has_edge(u, v) and not G.has_edge(w, x):
                    # Effectuer l'échange
                    G.remove_edge(u, w)
                    G.remove_edge(v, x)
                    G.add_edge(u, v)
                    G.add_edge(w, x)

        return G

    # ========================================================================
    # MÉTHODE 2: K-ANONYMISATION
    # ========================================================================

    def k_degree_anonymity(self, k):
        """
        k-degree anonymity: Assure que chaque degré apparaît au moins k fois

        Args:
            k: paramètre d'anonymité

        Returns:
            Graphe anonymisé
        """
        G = self.original_graph.copy()

        # Calculer la distribution des degrés
        degrees = dict(G.degree())
        degree_counts = Counter(degrees.values())

        # Trouver les degrés qui n'apparaissent pas k fois
        for degree, count in degree_counts.items():
            if count < k:
                # Trouver les nœuds avec ce degré
                nodes_with_degree = [n for n, d in degrees.items() if d == degree]

                # Ajouter des arêtes pour augmenter leur degré
                for node in nodes_with_degree:
                    needed = k - count
                    for _ in range(needed):
                        # Trouver un nœud non connecté
                        candidates = [n for n in G.nodes() if n != node and not G.has_edge(node, n)]
                        if candidates:
                            target = random.choice(candidates)
                            G.add_edge(node, target)

        return G

    # ========================================================================
    # MÉTHODE 3: ANONYMISATION PAR GÉNÉRALISATION
    # ========================================================================

    def generalization(self, k):
        """
        Généralisation: Groupe les nœuds en super-nœuds de taille >= k

        Args:
            k: taille minimale des super-nœuds

        Returns:
            Graphe généralisé et mapping des clusters
        """
        G = self.original_graph.copy()

        # Clustering simple basé sur les communautés
        communities = list(nx.community.greedy_modularity_communities(G))

        # Créer un graphe de super-nœuds
        super_graph = nx.Graph()
        node_to_cluster = {}

        cluster_id = 0
        for community in communities:
            # Si la communauté est trop petite, on la fusionne avec une autre
            if len(community) < k:
                # Fusionner avec la plus grande communauté
                if communities:
                    largest = max(communities, key=len)
                    community = community.union(largest)

            # Assigner les nœuds au cluster
            for node in community:
                node_to_cluster[node] = cluster_id

            super_graph.add_node(cluster_id, size=len(community))
            cluster_id += 1

        # Ajouter les super-arêtes
        for u, v in G.edges():
            cluster_u = node_to_cluster.get(u)
            cluster_v = node_to_cluster.get(v)

            if cluster_u is not None and cluster_v is not None:
                if cluster_u != cluster_v:
                    if super_graph.has_edge(cluster_u, cluster_v):
                        super_graph[cluster_u][cluster_v]['weight'] += 1
                    else:
                        super_graph.add_edge(cluster_u, cluster_v, weight=1)

        return super_graph, node_to_cluster

    # ========================================================================
    # MÉTHODE 4: APPROCHES PROBABILISTES
    # ========================================================================

    def probabilistic_obfuscation(self, k, epsilon):
        """
        (k,ε)-obfuscation: Ajoute des arêtes potentielles avec des probabilités

        Args:
            k: niveau d'anonymisation
            epsilon: paramètre de tolérance

        Returns:
            Graphe avec probabilités sur les arêtes
        """
        G = self.original_graph.copy()

        # Convertir en graphe probabiliste
        prob_graph = nx.Graph()
        prob_graph.add_nodes_from(G.nodes())

        # Les arêtes existantes ont probabilité élevée
        for u, v in G.edges():
            prob_graph.add_edge(u, v, probability=1.0 - epsilon/k)

        # Ajouter des arêtes potentielles avec probabilités plus faibles
        for u in G.nodes():
            # Trouver les k plus proches voisins
            neighbors = set(G.neighbors(u))
            non_neighbors = set(G.nodes()) - neighbors - {u}

            # Ajouter des arêtes potentielles
            for v in non_neighbors:
                if random.random() < 0.3:  # 30% des non-voisins
                    prob = epsilon / (2 * k)
                    prob_graph.add_edge(u, v, probability=prob)

        return prob_graph

    # ========================================================================
    # MÉTHODE 5: PRIVACY DIFFÉRENTIELLE
    # ========================================================================

    def differential_privacy_edgeflip(self, epsilon):
        """
        EdgeFlip: Applique le Randomized Response Technique
        Chaque arête est inversée avec probabilité s/2

        Args:
            epsilon: budget de privacy

        Returns:
            Graphe anonymisé avec ε-differential privacy
        """
        G = nx.Graph()
        G.add_nodes_from(self.original_graph.nodes())

        # Calculer s pour satisfaire ε-DP
        s = 1 - np.exp(-epsilon)

        # Pour chaque paire de nœuds possible
        for u in self.original_graph.nodes():
            for v in self.original_graph.nodes():
                if u < v:  # Éviter les doublons
                    exists = self.original_graph.has_edge(u, v)

                    # Appliquer le randomized response
                    if random.random() < s/2:
                        # Inverser l'arête
                        if exists:
                            pass  # Ne pas ajouter
                        else:
                            G.add_edge(u, v)
                    else:
                        # Garder l'état original
                        if exists:
                            G.add_edge(u, v)

        return G

    def differential_privacy_laplace(self, epsilon):
        """
        Mécanisme de Laplace: Ajoute du bruit Laplacien aux arêtes

        Args:
            epsilon: budget de privacy

        Returns:
            Graphe anonymisé
        """
        G = self.original_graph.copy()

        # Sensibilité globale = 1 (ajouter/supprimer une arête)
        sensitivity = 1
        scale = sensitivity / epsilon

        # Pour chaque arête potentielle, décider avec bruit
        new_graph = nx.Graph()
        new_graph.add_nodes_from(G.nodes())

        for u in G.nodes():
            for v in G.nodes():
                if u < v:
                    # Valeur vraie: 1 si arête existe, 0 sinon
                    true_value = 1 if G.has_edge(u, v) else 0

                    # Ajouter du bruit Laplacien
                    noise = np.random.laplace(0, scale)
                    noisy_value = true_value + noise

                    # Décision: ajouter l'arête si valeur bruitée > 0.5
                    if noisy_value > 0.5:
                        new_graph.add_edge(u, v)

        return new_graph


class GraphVisualizer:
    """Classe pour visualiser les graphes et leurs propriétés"""

    @staticmethod
    def plot_graph_comparison(original, anonymized, method_name, ax=None):
        """
        Compare deux graphes côte à côte

        Args:
            original: graphe original
            anonymized: graphe anonymisé
            method_name: nom de la méthode
            ax: axes matplotlib (optionnel)
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            ax1, ax2 = ax

        # Graphe original
        pos = nx.spring_layout(original, seed=42)
        nx.draw(original, pos, ax=ax1, with_labels=True,
                node_color='lightblue', node_size=500,
                font_size=10, font_weight='bold')
        ax1.set_title(f'Graphe Original\n{original.number_of_nodes()} nœuds, {original.number_of_edges()} arêtes')

        # Graphe anonymisé
        if isinstance(anonymized, nx.Graph):
            # Utiliser la même disposition
            if set(anonymized.nodes()).issubset(set(original.nodes())):
                pos_anon = pos
            else:
                pos_anon = nx.spring_layout(anonymized, seed=42)

            # Colorer différemment selon le type d'arête
            edge_colors = []
            for u, v in anonymized.edges():
                if original.has_edge(u, v):
                    edge_colors.append('blue')  # Arête originale
                else:
                    edge_colors.append('red')  # Nouvelle arête

            if not edge_colors:
                edge_colors = ['blue']

            nx.draw(anonymized, pos_anon, ax=ax2, with_labels=True,
                   node_color='lightgreen', node_size=500,
                   font_size=10, font_weight='bold',
                   edge_color=edge_colors, width=2)
            ax2.set_title(f'Graphe Anonymisé - {method_name}\n{anonymized.number_of_nodes()} nœuds, {anonymized.number_of_edges()} arêtes')

        plt.tight_layout()

    @staticmethod
    def plot_degree_distribution(original, anonymized, method_name):
        """
        Compare les distributions de degrés

        Args:
            original: graphe original
            anonymized: graphe anonymisé
            method_name: nom de la méthode
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Distribution originale
        degrees_orig = [d for n, d in original.degree()]
        ax1.hist(degrees_orig, bins=range(max(degrees_orig)+2),
                alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Degré')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution des degrés - Original')
        ax1.grid(True, alpha=0.3)

        # Distribution anonymisée
        if isinstance(anonymized, nx.Graph):
            degrees_anon = [d for n, d in anonymized.degree()]
            ax2.hist(degrees_anon, bins=range(max(degrees_anon)+2),
                    alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Degré')
            ax2.set_ylabel('Fréquence')
            ax2.set_title(f'Distribution des degrés - {method_name}')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    @staticmethod
    def plot_metrics_comparison(original, anonymized_graphs, method_names):
        """
        Compare plusieurs métriques pour toutes les méthodes

        Args:
            original: graphe original
            anonymized_graphs: liste de graphes anonymisés
            method_names: liste des noms de méthodes
        """
        metrics = {
            'Nb arêtes': [],
            'Degré moyen': [],
            'Clustering coeff': [],
            'Densité': []
        }

        # Métriques du graphe original
        orig_metrics = {
            'Nb arêtes': original.number_of_edges(),
            'Degré moyen': sum(d for n, d in original.degree()) / original.number_of_nodes(),
            'Clustering coeff': nx.average_clustering(original),
            'Densité': nx.density(original)
        }

        # Calculer les métriques pour chaque graphe anonymisé
        for G in anonymized_graphs:
            if isinstance(G, nx.Graph):
                metrics['Nb arêtes'].append(G.number_of_edges())
                metrics['Degré moyen'].append(sum(d for n, d in G.degree()) / G.number_of_nodes())
                metrics['Clustering coeff'].append(nx.average_clustering(G))
                metrics['Densité'].append(nx.density(G))
            else:
                # Valeurs par défaut si pas un graphe NetworkX standard
                metrics['Nb arêtes'].append(0)
                metrics['Degré moyen'].append(0)
                metrics['Clustering coeff'].append(0)
                metrics['Densité'].append(0)

        # Créer les graphiques
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]

            # Ajouter la valeur originale
            x = ['Original'] + method_names
            y = [orig_metrics[metric_name]] + values

            colors = ['blue'] + ['green'] * len(method_names)
            bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black')

            ax.set_ylabel(metric_name)
            ax.set_title(f'Comparaison: {metric_name}')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()


def main():
    """Fonction principale de démonstration"""

    print("=" * 80)
    print("DEMONSTRATION DES 5 TYPES DE METHODES D'ANONYMISATION DE GRAPHES")
    print("Base sur la these: Anonymisation de Graphes Sociaux - NGUYEN Huu-Hiep")
    print("=" * 80)
    print()

    # Charger le graphe Karate Club
    print("Chargement du graphe Karate Club de Zachary...")
    G = nx.karate_club_graph()
    print(f"Graphe charge: {G.number_of_nodes()} noeuds, {G.number_of_edges()} aretes")
    print()

    # Creer l'anonymizer
    anonymizer = GraphAnonymizer(G)
    visualizer = GraphVisualizer()

    # Liste pour stocker tous les graphes anonymises
    anonymized_graphs = []
    method_names = []

    # ========================================================================
    print("1. METHODE DE RANDOMISATION")
    print("-" * 80)

    # Random Add/Del
    print("   a) Random Add/Del (ajoute 5 aretes, supprime 5 aretes)")
    G_rand_add_del = anonymizer.random_add_del(k=5)
    anonymized_graphs.append(G_rand_add_del)
    method_names.append("Random\nAdd/Del")
    print(f"      Resultat: {G_rand_add_del.number_of_edges()} aretes")

    # Random Switch
    print("   b) Random Switch (echange 10 paires d'aretes)")
    G_rand_switch = anonymizer.random_switch(k=10)
    anonymized_graphs.append(G_rand_switch)
    method_names.append("Random\nSwitch")
    print(f"      Resultat: {G_rand_switch.number_of_edges()} aretes (degres preserves)")
    print()

    # ========================================================================
    print("2. K-ANONYMISATION")
    print("-" * 80)
    print("   k-degree anonymity (k=3)")
    G_k_anon = anonymizer.k_degree_anonymity(k=3)
    anonymized_graphs.append(G_k_anon)
    method_names.append("k-degree\nanonymity")
    print(f"   Resultat: {G_k_anon.number_of_edges()} aretes")
    print()

    # ========================================================================
    print("3. GENERALISATION")
    print("-" * 80)
    print("   Super-nodes avec k=3")
    G_generalized, mapping = anonymizer.generalization(k=3)
    anonymized_graphs.append(G_generalized)
    method_names.append("Generalisation\n(super-nodes)")
    print(f"   Resultat: {G_generalized.number_of_nodes()} super-noeuds")
    print()

    # ========================================================================
    print("4. APPROCHES PROBABILISTES")
    print("-" * 80)
    print("   (k,epsilon)-obfuscation avec k=3, epsilon=0.1")
    G_prob = anonymizer.probabilistic_obfuscation(k=3, epsilon=0.1)
    anonymized_graphs.append(G_prob)
    method_names.append("Probabiliste\n(k,eps)-obf")
    print(f"   Resultat: {G_prob.number_of_edges()} aretes (avec probabilites)")
    print()

    # ========================================================================
    print("5. PRIVACY DIFFERENTIELLE")
    print("-" * 80)

    # EdgeFlip
    print("   a) EdgeFlip avec epsilon=1.0")
    G_edgeflip = anonymizer.differential_privacy_edgeflip(epsilon=1.0)
    anonymized_graphs.append(G_edgeflip)
    method_names.append("EdgeFlip\n(eps=1.0)")
    print(f"      Resultat: {G_edgeflip.number_of_edges()} aretes")

    # Laplace
    print("   b) Mecanisme de Laplace avec epsilon=0.5")
    G_laplace = anonymizer.differential_privacy_laplace(epsilon=0.5)
    anonymized_graphs.append(G_laplace)
    method_names.append("Laplace\n(eps=0.5)")
    print(f"      Resultat: {G_laplace.number_of_edges()} aretes")
    print()

    # ========================================================================
    print("VISUALISATIONS")
    print("=" * 80)
    print("Generation des visualisations...")

    # Créer une grande figure avec toutes les comparaisons
    fig = plt.figure(figsize=(20, 24))

    # Grille 7x2 pour 7 méthodes (2 colonnes: original vs anonymisé)
    for idx, (G_anon, name) in enumerate(zip(anonymized_graphs, method_names)):
        ax1 = plt.subplot(7, 2, 2*idx + 1)
        ax2 = plt.subplot(7, 2, 2*idx + 2)

        visualizer.plot_graph_comparison(G, G_anon, name, ax=(ax1, ax2))

    plt.suptitle('Comparaison des 5 Types de Methodes d\'Anonymisation de Graphes',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('graph_anonymization_comparison.png', dpi=300, bbox_inches='tight')
    print("   [OK] Graphique de comparaison sauvegarde: graph_anonymization_comparison.png")

    # Comparer les distributions de degrés
    fig2 = plt.figure(figsize=(20, 24))
    for idx, (G_anon, name) in enumerate(zip(anonymized_graphs, method_names)):
        ax1 = plt.subplot(7, 2, 2*idx + 1)
        ax2 = plt.subplot(7, 2, 2*idx + 2)

        # Distribution originale
        degrees_orig = [d for n, d in G.degree()]
        ax1.hist(degrees_orig, bins=range(max(degrees_orig)+2),
                alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Degré')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Original')
        ax1.grid(True, alpha=0.3)

        # Distribution anonymisée
        if isinstance(G_anon, nx.Graph) and G_anon.number_of_nodes() > 0:
            degrees_anon = [d for n, d in G_anon.degree()]
            if degrees_anon:
                ax2.hist(degrees_anon, bins=range(max(degrees_anon)+2),
                        alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Degré')
            ax2.set_ylabel('Fréquence')
            ax2.set_title(f'{name}')
            ax2.grid(True, alpha=0.3)

    plt.suptitle('Comparaison des Distributions de Degres',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('degree_distributions.png', dpi=300, bbox_inches='tight')
    print("   [OK] Distributions de degres sauvegardees: degree_distributions.png")

    # Comparaison des metriques
    visualizer.plot_metrics_comparison(G, anonymized_graphs, method_names)
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("   [OK] Comparaison des metriques sauvegardee: metrics_comparison.png")

    print()
    print("=" * 80)
    print("TERMINE!")
    print("Les visualisations ont ete sauvegardees dans le repertoire courant.")
    print("=" * 80)

    # plt.show()  # Desactive pour ne pas bloquer en mode batch


if __name__ == "__main__":
    main()
