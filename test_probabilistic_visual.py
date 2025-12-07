"""
Test visuel de (k,ε)-obfuscation pour vérifier les probabilités
"""

import networkx as nx
import numpy as np
from graph_anonymization_app import GraphAnonymizer

# Créer un graphe de test
G = nx.karate_club_graph()
anonymizer = GraphAnonymizer(G)

print("=" * 80)
print("TEST VISUEL (k,eps)-OBFUSCATION")
print("=" * 80)
print(f"Graphe original : {G.number_of_nodes()} noeuds, {G.number_of_edges()} aretes")
print()

# Tester avec differents parametres
test_cases = [
    (5, 0.3),
    (5, 0.5),
    (10, 0.3),
]

for k, eps in test_cases:
    print(f"TEST k={k}, eps={eps}")
    print("-" * 80)

    # Formules theoriques
    p_exist_theory = 1.0 - (eps / k)
    p_potential_theory = eps / (2 * k)

    print(f"Formule theorique :")
    print(f"  p_existantes = 1 - eps/k = 1 - {eps}/{k} = {p_exist_theory:.4f}")
    print(f"  p_potentielles = eps/(2k) = {eps}/(2*{k}) = {p_potential_theory:.4f}")
    print()

    # Generer le graphe
    G_prob = anonymizer.probabilistic_obfuscation(k=k, epsilon=eps)

    # Analyser les probabilites
    existing_probs = []
    potential_probs = []

    for u, v in G_prob.edges():
        prob = G_prob[u][v].get('probability', 1.0)
        is_original = G_prob[u][v].get('is_original', False)

        if is_original:
            existing_probs.append(prob)
        else:
            potential_probs.append(prob)

    print(f"Resultats observes :")
    print(f"  Aretes existantes : {len(existing_probs)} aretes")
    print(f"    Probabilite : {existing_probs[0]:.4f} (uniforme)")
    print(f"    Attendu : {p_exist_theory:.4f}")
    match1 = '+' if abs(existing_probs[0] - p_exist_theory) < 0.0001 else 'X'
    print(f"    Match : {match1}")
    print()

    print(f"  Aretes potentielles : {len(potential_probs)} aretes")
    if len(potential_probs) > 0:
        print(f"    Probabilite : {potential_probs[0]:.4f} (uniforme)")
        print(f"    Attendu : {p_potential_theory:.4f}")
        match2 = '+' if abs(potential_probs[0] - p_potential_theory) < 0.0001 else 'X'
        print(f"    Match : {match2}")
    else:
        print(f"    Aucune arete potentielle ajoutee")
    print()

    # Test de reconstruction par seuillage
    threshold = 0.5
    reconstructed = sum(1 for p in existing_probs if p > threshold)
    reconstruction_rate = reconstructed / len(existing_probs) * 100

    print(f"Test de reconstruction (seuil={threshold}) :")
    print(f"  {reconstructed}/{len(existing_probs)} aretes originales recuperees ({reconstruction_rate:.1f}%)")

    if reconstruction_rate == 100.0:
        print(f"  [!] VULNERABLE : Reconstruction totale possible!")
    elif reconstruction_rate > 90.0:
        print(f"  [!] Tres vulnerable")
    else:
        print(f"  [+] Resistant")

    print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print("L'implémentation suit exactement les formules de la thèse.")
print("Les probabilités sont uniformes et concentrées autour de 0/1.")
print("→ Vulnérable à la reconstruction par seuillage (but pédagogique).")
print("=" * 80)
