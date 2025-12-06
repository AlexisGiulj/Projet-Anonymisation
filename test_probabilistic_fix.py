"""
Test de la correction du gradient (k,ε)-obfuscation
Vérifie que ε GRAND = PLUS DE PRIVACY
"""

import networkx as nx
from graph_anonymization_app import GraphAnonymizer

# Créer un graphe de test
G = nx.karate_club_graph()
anonymizer = GraphAnonymizer(G)

print("=" * 80)
print("TEST DE LA CORRECTION : epsilon GRAND = PLUS DE PRIVACY")
print("=" * 80)
print()

# Tester avec differentes valeurs de epsilon
test_cases = [
    (5, 0.1, "FAIBLE privacy (epsilon petit)"),
    (5, 0.5, "MOYENNE privacy"),
    (5, 1.0, "FORTE privacy (epsilon grand)"),
]

for k, eps, label in test_cases:
    print(f"Test : k={k}, epsilon={eps} -> {label}")
    print("-" * 60)

    prob_graph = anonymizer.probabilistic_obfuscation(k=k, epsilon=eps)

    # Analyser les probabilités
    existing_probs = []
    potential_probs = []

    for u, v in prob_graph.edges():
        prob = prob_graph[u][v]['probability']
        is_orig = prob_graph[u][v]['is_original']

        if is_orig:
            existing_probs.append(prob)
        else:
            potential_probs.append(prob)

    if existing_probs:
        avg_existing = sum(existing_probs) / len(existing_probs)
        min_existing = min(existing_probs)
        max_existing = max(existing_probs)
        print(f"  Arêtes EXISTANTES ({len(existing_probs)}) :")
        print(f"    Min  : {min_existing:.3f}")
        print(f"    Moy  : {avg_existing:.3f}")
        print(f"    Max  : {max_existing:.3f}")

    if potential_probs:
        avg_potential = sum(potential_probs) / len(potential_probs)
        min_potential = min(potential_probs)
        max_potential = max(potential_probs)
        print(f"  Arêtes POTENTIELLES ({len(potential_probs)}) :")
        print(f"    Min  : {min_potential:.3f}")
        print(f"    Moy  : {avg_potential:.3f}")
        print(f"    Max  : {max_potential:.3f}")

    # Calculer l'écart (gradient)
    if existing_probs and potential_probs:
        gradient = avg_existing - avg_potential
        print(f"  GRADIENT (existantes - potentielles) : {gradient:.3f}")
        print(f"  -> Plus le gradient est FAIBLE, plus la privacy est ELEVEE")

    print()

print("=" * 80)
print("OBSERVATION ATTENDUE :")
print("=" * 80)
print("Quand epsilon augmente de 0.1 a 1.0 :")
print("  [OK] Prob. aretes EXISTANTES diminue (vert -> jaune -> rouge)")
print("  [OK] Prob. aretes POTENTIELLES augmente (rouge -> jaune -> vert)")
print("  [OK] GRADIENT diminue -> Graphe plus melange -> Plus de privacy")
print("=" * 80)
