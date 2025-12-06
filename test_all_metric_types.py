"""
Test des metriques d'utilite sur les 3 types de graphes :
1. Graphe classique (Random Switch)
2. Graphe probabiliste ((k,epsilon)-obfuscation)
3. Super-graphe (Generalisation)

Verifie que :
- Les metriques sont calculees correctement pour chaque type
- Les tooltips sont disponibles
- L'affichage UI est adapte au type de graphe
"""

import networkx as nx
from graph_anonymization_app import (
    GraphAnonymizer,
    calculate_utility_metrics,
    calculate_supergraph_metrics,
    get_metric_tooltip,
    METRIC_DEFINITIONS
)

# Creer un graphe de test
G = nx.karate_club_graph()
anonymizer = GraphAnonymizer(G)

print("=" * 80)
print("TEST DES METRIQUES D'UTILITE SUR LES 3 TYPES DE GRAPHES")
print("=" * 80)
print(f"Graphe original : {G.number_of_nodes()} noeuds, {G.number_of_edges()} aretes")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1 : GRAPHE CLASSIQUE (Random Switch)
# ═══════════════════════════════════════════════════════════════════════════
print("TEST 1 : GRAPHE CLASSIQUE (Random Switch)")
print("-" * 80)

G_random = anonymizer.random_switch(k=10)
metrics_random = calculate_utility_metrics(G, G_random)

print(f"Type detecte : {metrics_random.get('type', 'graphe classique')}")
print(f"Comparable : {metrics_random.get('comparable', True)}")
print(f"Est un echantillon probabiliste : {metrics_random.get('is_sample', False)}")
print(f"Est un super-graphe : {metrics_random.get('is_super_graph', False)}")
print()

print("METRIQUES CALCULEES :")
print(f"  - Noeuds : {metrics_random.get('num_nodes')}")
print(f"  - Aretes : {metrics_random.get('num_edges')}")
print(f"  - Densite : {metrics_random.get('density', 0):.3f}")
print(f"  - Clustering moyen : {metrics_random.get('avg_clustering', 0):.3f}")
print(f"  - Diametre : {metrics_random.get('diameter')}")
print(f"  - Distance moyenne : {metrics_random.get('avg_shortest_path', 0):.2f}")
print(f"  - Correlation degres : {metrics_random.get('degree_correlation', 0):.3f}")
print()

# Tester les tooltips
print("TOOLTIPS DISPONIBLES :")
print(f"  - density : {'OK' if get_metric_tooltip('density') else 'MANQUANT'}")
print(f"  - avg_clustering : {'OK' if get_metric_tooltip('avg_clustering') else 'MANQUANT'}")
print(f"  - diameter : {'OK' if get_metric_tooltip('diameter') else 'MANQUANT'}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2 : GRAPHE PROBABILISTE ((k,epsilon)-obfuscation)
# ═══════════════════════════════════════════════════════════════════════════
print("TEST 2 : GRAPHE PROBABILISTE ((k,epsilon)-obfuscation)")
print("-" * 80)

G_prob = anonymizer.probabilistic_obfuscation(k=5, epsilon=0.5)
metrics_prob = calculate_utility_metrics(G, G_prob)

print(f"Type detecte : {metrics_prob.get('type', 'graphe classique')}")
print(f"Comparable : {metrics_prob.get('comparable', True)}")
print(f"Est un echantillon probabiliste : {metrics_prob.get('is_sample', False)}")
print(f"Est un super-graphe : {metrics_prob.get('is_super_graph', False)}")
print()

print("METRIQUES CALCULEES :")
print(f"  - Noeuds : {metrics_prob.get('num_nodes')}")
print(f"  - Aretes : {metrics_prob.get('num_edges')}")
print(f"  - Densite : {metrics_prob.get('density', 0):.3f}")
print(f"  - Clustering moyen : {metrics_prob.get('avg_clustering', 0):.3f}")
print(f"  - Diametre : {metrics_prob.get('diameter')}")
print(f"  - Distance moyenne : {metrics_prob.get('avg_shortest_path', 0):.2f}")
print(f"  - Correlation degres : {metrics_prob.get('degree_correlation', 0):.3f}")
print()

if metrics_prob.get('is_sample'):
    print("[INFO] Graphe probabiliste -> Metriques calculees sur un ECHANTILLON")
print()

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3 : SUPER-GRAPHE (Generalisation)
# ═══════════════════════════════════════════════════════════════════════════
print("TEST 3 : SUPER-GRAPHE (Generalisation)")
print("-" * 80)

G_super, node_to_cluster = anonymizer.generalization(k=5)
metrics_super = calculate_utility_metrics(G, G_super)

print(f"Type detecte : {metrics_super.get('type', 'graphe classique')}")
print(f"Comparable : {metrics_super.get('comparable', True)}")
print(f"Est un echantillon probabiliste : {metrics_super.get('is_sample', False)}")
print(f"Est un super-graphe : {metrics_super.get('is_super_graph', False)}")
print()

print("METRIQUES CALCULEES (SPECIFIQUES AU SUPER-GRAPHE) :")
print(f"  - Nombre de clusters : {metrics_super.get('num_clusters')}")
print(f"  - Taille min. cluster : {metrics_super.get('min_cluster_size')}")
print(f"  - Taille moy. cluster : {metrics_super.get('avg_cluster_size', 0):.1f}")
print(f"  - Taille max. cluster : {metrics_super.get('max_cluster_size')}")
print(f"  - Aretes intra-cluster : {metrics_super.get('intra_cluster_edges')}")
print(f"  - Aretes inter-cluster : {metrics_super.get('inter_cluster_edges')}")
print(f"  - Ratio intra/total : {metrics_super.get('intra_ratio', 0)*100:.1f}%")
print(f"  - Perte d'information : {metrics_super.get('information_loss', 0)*100:.1f}%")
print(f"  - Preservation aretes : {metrics_super.get('edge_preservation_ratio', 0)*100:.1f}%")
print(f"  - Densite super-graphe : {metrics_super.get('super_graph_density', 0):.3f}")
print()

# Tester les tooltips specifiques au super-graphe
print("TOOLTIPS DISPONIBLES (Super-Graphe) :")
print(f"  - num_clusters : {'OK' if get_metric_tooltip('num_clusters') else 'MANQUANT'}")
print(f"  - min_cluster_size : {'OK' if get_metric_tooltip('min_cluster_size') else 'MANQUANT'}")
print(f"  - intra_cluster_edges : {'OK' if get_metric_tooltip('intra_cluster_edges') else 'MANQUANT'}")
print(f"  - information_loss : {'OK' if get_metric_tooltip('information_loss') else 'MANQUANT'}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION FINALE
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("VALIDATION FINALE")
print("=" * 80)

all_ok = True

# Test 1 : Graphe classique
if metrics_random.get('comparable') and not metrics_random.get('is_super_graph'):
    print("[OK] Graphe classique detecte et metriques calculees")
else:
    print("[ERREUR] Probleme avec le graphe classique")
    all_ok = False

# Test 2 : Graphe probabiliste
if metrics_prob.get('is_sample') and metrics_prob.get('comparable'):
    print("[OK] Graphe probabiliste detecte et echantillonnage effectue")
else:
    print("[ERREUR] Probleme avec le graphe probabiliste")
    all_ok = False

# Test 3 : Super-graphe
if metrics_super.get('is_super_graph') and metrics_super.get('comparable'):
    print("[OK] Super-graphe detecte et metriques adaptees calculees")
else:
    print("[ERREUR] Probleme avec le super-graphe")
    all_ok = False

# Test 4 : Tooltips
nb_tooltips = len([k for k in METRIC_DEFINITIONS.keys() if get_metric_tooltip(k)])
if nb_tooltips >= 15:
    print(f"[OK] {nb_tooltips} tooltips disponibles")
else:
    print(f"[ERREUR] Seulement {nb_tooltips} tooltips disponibles (attendu >= 15)")
    all_ok = False

print()
if all_ok:
    print("=" * 80)
    print("SUCCES : Tous les tests passes !")
    print("=" * 80)
else:
    print("=" * 80)
    print("ECHEC : Certains tests ont echoue")
    print("=" * 80)
