"""
Script de test pour vérifier les améliorations du simulateur d'attaque
"""

import networkx as nx
from graph_anonymization_app import simulate_degree_attack, simulate_subgraph_attack

# Créer un graphe de test simple (Karate Club)
G_orig = nx.karate_club_graph()

# Tester l'attaque par degré sur le graphe original (pas d'anonymisation)
print("=" * 80)
print("TEST 1 : Degree Attack sur graphe non-anonymisé")
print("=" * 80)
result_degree = simulate_degree_attack(G_orig, G_orig, target_node=0)
print(f"Target node: {result_degree['target_node']}")
print(f"Target degree: {result_degree['target_degree']}")
print(f"Candidates: {result_degree['candidates']}")
print(f"Success: {result_degree['success']}")
if 're_identification_probability' in result_degree:
    print(f"Re-ID probability: {result_degree['re_identification_probability']:.2%}")
if 'min_entropy_bits' in result_degree:
    print(f"Min Entropy: {result_degree['min_entropy_bits']:.2f} bits")
if 'incorrectness' in result_degree:
    print(f"Incorrectness: {result_degree['incorrectness']}")
print()

# Tester l'attaque par sous-graphe
print("=" * 80)
print("TEST 2 : Subgraph Attack sur graphe non-anonymisé")
print("=" * 80)
result_subgraph = simulate_subgraph_attack(G_orig, G_orig, target_node=0)
print(f"Target node: {result_subgraph['target_node']}")
print(f"Target degree: {result_subgraph.get('target_degree', 'N/A')}")
print(f"Target triangles: {result_subgraph.get('target_triangles', 'N/A')}")
print(f"Clustering coefficient: {result_subgraph.get('clustering_coefficient', 'N/A')}")
print(f"Candidates: {result_subgraph['candidates']}")
print(f"Success: {result_subgraph['success']}")
if 're_identification_probability' in result_subgraph:
    print(f"Re-ID probability: {result_subgraph['re_identification_probability']:.2%}")
if 'min_entropy_bits' in result_subgraph:
    print(f"Min Entropy: {result_subgraph['min_entropy_bits']:.2f} bits")

print("\n" + "=" * 80)
print("TESTS TERMINÉS AVEC SUCCÈS !")
print("=" * 80)
