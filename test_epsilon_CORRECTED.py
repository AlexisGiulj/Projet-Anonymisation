"""
Test du comportement CORRIGE d'epsilon pour EdgeFlip.
Utilise la FORMULE CORRECTE : s = 2/(e^epsilon + 1)
"""

import networkx as nx
import numpy as np
from graph_anonymization_app import GraphAnonymizer

# Creer un graphe de test
G = nx.karate_club_graph()
orig_edges = G.number_of_edges()
anonymizer = GraphAnonymizer(G)

print("=" * 80)
print("TEST EPSILON CORRIGE - EdgeFlip avec FORMULE CORRECTE")
print("=" * 80)
print(f"Graphe original : {G.number_of_nodes()} noeuds, {orig_edges} aretes")
print()
print("FORMULE CORRECTE : s = 2 / (e^epsilon + 1)")
print()

epsilon_values = [0.1, 0.5, 1.0, 2.0, 3.0]

for eps in epsilon_values:
    # Utilise la methode corrigee
    G_anon = anonymizer.differential_privacy_edgeflip(epsilon=eps)

    # Compter les modifications
    added = len(set(G_anon.edges()) - set(G.edges()))
    removed = len(set(G.edges()) - set(G_anon.edges()))
    total_changes = added + removed
    change_rate = total_changes / (2 * orig_edges)

    # Calculer s avec la formule CORRECTE
    s = 2 / (np.exp(eps) + 1)
    flip_prob = s / 2

    print(f"epsilon = {eps:.1f}")
    print(f"  s = {s:.3f} (parametre EdgeFlip)")
    print(f"  flip_prob = {flip_prob:.3f} ({flip_prob*100:.1f}%)")
    print(f"  Aretes ajoutees   : {added}")
    print(f"  Aretes supprimees : {removed}")
    print(f"  Total changes     : {total_changes}")
    print(f"  Taux de modif     : {change_rate*100:.1f}%")

    # Determiner le niveau de privacy
    if eps < 1.0:
        level = "FORTE privacy"
    elif eps < 2.0:
        level = "privacy MOYENNE"
    else:
        level = "FAIBLE privacy"

    print(f"  => {level}")
    print()

print("=" * 80)
print("OBSERVATION CORRECTE :")
print("=" * 80)
print("epsilon PETIT (0.1) => s GRAND (0.95) => flip 47.5% => BEAUCOUP de modifs => FORTE privacy")
print("epsilon GRAND (3.0) => s PETIT (0.09) => flip 4.7% => PEU de modifs => FAIBLE privacy")
print()
print("Trade-off Privacy-Utilite LOGIQUE :")
print("  - epsilon petit => FORTE privacy mais FAIBLE utilite (graphe tres different)")
print("  - epsilon grand => FAIBLE privacy mais FORTE utilite (graphe proche)")
print()
print("C'est le comportement ATTENDU en Differential Privacy !")
print("=" * 80)
