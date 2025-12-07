"""
Test de la methode k-degree anonymity CORRIGEE.
Verifie que la contrainte k est RESPECTEE : chaque degre apparait au moins k fois.
"""

import networkx as nx
from collections import Counter
from graph_anonymization_app import GraphAnonymizer

# Creer un graphe de test
G = nx.karate_club_graph()
anonymizer = GraphAnonymizer(G)

print("=" * 80)
print("TEST k-DEGREE ANONYMITY - VERIFICATION DE LA CONTRAINTE")
print("=" * 80)
print(f"Graphe original : {G.number_of_nodes()} noeuds, {G.number_of_edges()} aretes")
print()

# Distribution originale
orig_degrees = dict(G.degree())
orig_degree_counts = Counter(orig_degrees.values())

print("DISTRIBUTION ORIGINALE DES DEGRES :")
print("-" * 80)
for degree in sorted(orig_degree_counts.keys()):
    count = orig_degree_counts[degree]
    print(f"  Degre {degree:2d} : {count:2d} noeuds")
print()

# Tester avec differentes valeurs de k
test_cases = [2, 3, 4, 5]

for k in test_cases:
    print(f"TEST k = {k}")
    print("-" * 80)

    # Anonymiser
    G_anon = anonymizer.k_degree_anonymity(k=k)

    # Calculer la distribution
    anon_degrees = dict(G_anon.degree())
    anon_degree_counts = Counter(anon_degrees.values())

    print(f"  Graphe anonymise : {G_anon.number_of_nodes()} noeuds, {G_anon.number_of_edges()} aretes")
    print(f"  Distribution des degres :")

    all_satisfy = True
    for degree in sorted(anon_degree_counts.keys()):
        count = anon_degree_counts[degree]
        satisfies = count >= k

        status = "OK" if satisfies else "VIOLATION"
        symbol = "+" if satisfies else "X"

        print(f"    Degre {degree:2d} : {count:2d} noeuds  [{symbol} {status}]")

        if not satisfies:
            all_satisfy = False

    print()

    if all_satisfy:
        print(f"  [OK] CONTRAINTE k-anonymity RESPECTEE : tous les degres >= {k} occurrences")
    else:
        print(f"  [ERREUR] CONTRAINTE VIOLEE : certains degres < {k} occurrences")

    # Statistiques de modification
    orig_edges = G.number_of_edges()
    anon_edges = G_anon.number_of_edges()
    diff = anon_edges - orig_edges

    print(f"  Modifications : {diff:+d} aretes ({abs(diff)/orig_edges*100:.1f}% de changement)")
    print()

print("=" * 80)
print("VALIDATION FINALE")
print("=" * 80)
print("[INFO] Verifiez visuellement l'histogramme dans l'application")
print("[INFO] Chaque barre devrait avoir une hauteur >= k")
print("=" * 80)
