"""
Test de la correction de la generalisation
Verifie que le parametre k influence le nombre de clusters
"""

import networkx as nx
from graph_anonymization_app import GraphAnonymizer

# Creer un graphe de test
G = nx.karate_club_graph()
anonymizer = GraphAnonymizer(G)

print("=" * 80)
print("TEST DE LA CORRECTION : Generalisation avec parametre k")
print("=" * 80)
print(f"Graphe original : {G.number_of_nodes()} noeuds, {G.number_of_edges()} aretes")
print()

# Tester avec differentes valeurs de k
test_cases = [
    (2, "k=2: Clusters de taille >= 2"),
    (4, "k=4: Clusters de taille >= 4"),
    (7, "k=7: Clusters de taille >= 7"),
    (10, "k=10: Clusters de taille >= 10"),
]

for k, label in test_cases:
    print(f"Test : {label}")
    print("-" * 60)

    super_graph, node_to_cluster = anonymizer.generalization(k=k)

    # Analyser les clusters
    cluster_sizes = {}
    for node, cluster_id in node_to_cluster.items():
        if cluster_id not in cluster_sizes:
            cluster_sizes[cluster_id] = 0
        cluster_sizes[cluster_id] += 1

    num_clusters = len(cluster_sizes)
    sizes = list(cluster_sizes.values())
    min_size = min(sizes)
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)

    print(f"  Nombre de clusters : {num_clusters}")
    print(f"  Tailles des clusters :")
    print(f"    Min  : {min_size} noeuds")
    print(f"    Moy  : {avg_size:.1f} noeuds")
    print(f"    Max  : {max_size} noeuds")
    print(f"  Detail : {sorted(sizes)}")

    # Verifier la contrainte k-anonymity
    if min_size >= k:
        print(f"  [OK] Contrainte k-anonymity respectee (min_size >= {k})")
    else:
        print(f"  [WARNING] Contrainte violee: min_size={min_size} < k={k}")

    # Statistiques du super-graphe
    total_intra = super_graph.graph.get('intra_edges', 0)
    total_inter = super_graph.graph.get('inter_edges', 0)
    print(f"  Aretes intra-cluster : {total_intra}")
    print(f"  Aretes inter-cluster : {total_inter}")

    print()

print("=" * 80)
print("OBSERVATION ATTENDUE :")
print("=" * 80)
print("Quand k augmente de 2 a 10 :")
print("  [OK] Nombre de clusters DIMINUE (clusters plus gros)")
print("  [OK] Taille minimale >= k (contrainte k-anonymity)")
print("  [OK] Plus k grand -> Moins de clusters -> Plus de privacy")
print("=" * 80)
