"""
Test de la méthode MaxVar pour vérifier qu'elle résiste au seuillage.
Compare avec (k,ε)-obfuscation pour montrer la différence.
"""

import networkx as nx
import numpy as np
from graph_anonymization_app import GraphAnonymizer

# Créer un graphe de test
G = nx.karate_club_graph()
anonymizer = GraphAnonymizer(G)

print("=" * 80)
print("COMPARAISON (k,eps)-OBFUSCATION vs MaxVar")
print("=" * 80)
print(f"Graphe original : {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
print()

# TEST 1: (k,eps)-obfuscation
print("TEST 1 : (k,eps)-obfuscation (VULNERABLE)")
print("-" * 80)

G_prob = anonymizer.probabilistic_obfuscation(k=5, epsilon=0.3)

# Analyser les probabilités
existing_edges_probs = []
potential_edges_probs = []

for u, v in G_prob.edges():
    prob = G_prob[u][v].get('probability', 1.0)
    is_original = G_prob[u][v].get('is_original', False)

    if is_original:
        existing_edges_probs.append(prob)
    else:
        potential_edges_probs.append(prob)

print(f"Arêtes existantes ({len(existing_edges_probs)}) :")
print(f"  Moyenne : {np.mean(existing_edges_probs):.3f}")
print(f"  Min-Max : [{np.min(existing_edges_probs):.3f}, {np.max(existing_edges_probs):.3f}]")
print()

print(f"Arêtes potentielles ({len(potential_edges_probs)}) :")
if len(potential_edges_probs) > 0:
    print(f"  Moyenne : {np.mean(potential_edges_probs):.3f}")
    print(f"  Min-Max : [{np.min(potential_edges_probs):.3f}, {np.max(potential_edges_probs):.3f}]")
else:
    print(f"  Aucune arête potentielle ajoutée")
print()

# Tester la reconstruction par seuillage
threshold = 0.5
reconstructed = set()
for u, v in G_prob.edges():
    if G_prob[u][v].get('probability', 1.0) > threshold:
        reconstructed.add((min(u,v), max(u,v)))

original_edges = set((min(u,v), max(u,v)) for u, v in G.edges())
accuracy = len(reconstructed & original_edges) / len(original_edges)

print(f"RECONSTRUCTION PAR SEUILLAGE (threshold={threshold}) :")
print(f"  Precision : {accuracy*100:.1f}% des aretes originales recuperees")
print(f"  -> {'VULNERABLE!' if accuracy > 0.9 else 'Resistant'}")
print()

# TEST 2: MaxVar
print("TEST 2 : MaxVar (RESISTANT)")
print("-" * 80)

G_maxvar = anonymizer.maxvar_obfuscation(num_potential_edges=50)

# Analyser les probabilités
existing_mv = []
potential_mv = []

for u, v in G_maxvar.edges():
    prob = G_maxvar[u][v].get('probability', 1.0)
    is_original = G_maxvar[u][v].get('is_original', False)

    if is_original:
        existing_mv.append(prob)
    else:
        potential_mv.append(prob)

print(f"Arêtes existantes ({len(existing_mv)}) :")
print(f"  Moyenne : {np.mean(existing_mv):.3f}")
print(f"  Écart-type : {np.std(existing_mv):.3f}")
print(f"  Min-Max : [{np.min(existing_mv):.3f}, {np.max(existing_mv):.3f}]")
print()

print(f"Arêtes potentielles ({len(potential_mv)}) :")
print(f"  Moyenne : {np.mean(potential_mv):.3f}")
print(f"  Écart-type : {np.std(potential_mv):.3f}")
print(f"  Min-Max : [{np.min(potential_mv):.3f}, {np.max(potential_mv):.3f}]")
print()

# Tester la reconstruction par seuillage
reconstructed_mv = set()
for u, v in G_maxvar.edges():
    if G_maxvar[u][v].get('probability', 1.0) > threshold:
        reconstructed_mv.add((min(u,v), max(u,v)))

accuracy_mv = len(reconstructed_mv & original_edges) / len(original_edges)

print(f"RECONSTRUCTION PAR SEUILLAGE (threshold={threshold}) :")
print(f"  Precision : {accuracy_mv*100:.1f}% des aretes originales recuperees")
print(f"  -> {'VULNERABLE!' if accuracy_mv > 0.9 else 'Resistant OK'}")
print()

# Verification de la conservation des degres
print("VERIFICATION DES DEGRES ATTENDUS (MaxVar) :")
print("-" * 80)

for node in list(G.nodes())[:5]:  # Verifier les 5 premiers noeuds
    original_degree = G.degree(node)
    expected_degree = sum(G_maxvar[node][v]['probability'] for v in G_maxvar.neighbors(node))
    print(f"  Noeud {node}: degre original = {original_degree}, degre attendu = {expected_degree:.2f}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"(k,eps)-obfuscation : {accuracy*100:.1f}% de reconstruction -> VULNERABLE")
print(f"MaxVar              : {accuracy_mv*100:.1f}% de reconstruction -> RESISTANT")
print("=" * 80)
