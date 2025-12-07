"""
Test du comportement epsilon pour les methodes de privacy differentielle.
Verifie la relation entre epsilon et les modifications du graphe.
"""

import networkx as nx
import numpy as np
from graph_anonymization_app import GraphAnonymizer

# Creer un graphe de test
G = nx.karate_club_graph()
orig_edges = G.number_of_edges()
anonymizer = GraphAnonymizer(G)

print("=" * 80)
print("TEST EPSILON vs MODIFICATIONS DU GRAPHE")
print("=" * 80)
print(f"Graphe original : {G.number_of_nodes()} noeuds, {orig_edges} aretes")
print()

# Test EdgeFlip
print("TEST 1 : EdgeFlip")
print("-" * 80)

epsilon_values = [0.1, 0.5, 1.0, 2.0, 3.0]

for eps in epsilon_values:
    G_anon = anonymizer.differential_privacy_edgeflip(epsilon=eps)

    # Compter les modifications
    added = len(set(G_anon.edges()) - set(G.edges()))
    removed = len(set(G.edges()) - set(G_anon.edges()))
    total_changes = added + removed
    change_rate = total_changes / (2 * orig_edges)  # Normalise

    s = 1 - np.exp(-eps)

    print(f"epsilon = {eps:.1f}")
    print(f"  s = {s:.3f} (probabilite de flip)")
    print(f"  Aretes ajoutees   : {added}")
    print(f"  Aretes supprimees : {removed}")
    print(f"  Total changes     : {total_changes}")
    print(f"  Taux de modif     : {change_rate*100:.1f}%")
    print()

print("=" * 80)
print("OBSERVATION :")
print("=" * 80)
print("epsilon PETIT (0.1) => PEU de modifications  => graphe PROCHE de l'original")
print("epsilon GRAND (3.0) => BEAUCOUP de modifs   => graphe DIFFERENT de l'original")
print()
print("MAIS en Differential Privacy :")
print("epsilon PETIT => FORTE privacy (indistinguishability)")
print("epsilon GRAND => FAIBLE privacy")
print()
print("⚠️ PROBLEME : C'est CONTRE-INTUITIF pour l'utilisateur !")
print("=" * 80)
print()
print("SOLUTION PROPOSEE :")
print("=" * 80)
print("Option 1 : Inverser epsilon dans l'UI (afficher 1/epsilon)")
print("  - Slider affiche 'Budget Privacy' de 0.33 a 10")
print("  - Plus le budget est GRAND, plus de privacy (epsilon effectif PETIT)")
print()
print("Option 2 : Afficher 'Niveau de Bruit' au lieu de 'Budget Privacy'")
print("  - Slider affiche 'Niveau de Bruit' de 0.1 a 3.0")
print("  - Plus le bruit est ELEVE, plus le graphe change")
print("  - Mais expliquer que bruit FAIBLE = privacy FORTE (paradoxe DP)")
print()
print("Option 3 : Garder epsilon mais AMELIORER le help text")
print("  - Expliquer clairement le paradoxe")
print("  - Montrer s = 1 - e^(-epsilon) et le taux de flip")
print("=" * 80)
