# Démonstration d'Anonymisation de Graphes Sociaux

Application de démonstration basée sur la thèse **"Anonymisation de Graphes Sociaux"** par **NGUYEN Huu-Hiep** (Université de Lorraine, 2016).

## 📚 Contexte de la Thèse

Cette thèse traite de la protection de la vie privée dans les réseaux sociaux lors de la publication de graphes. Elle identifie et catégorise **5 types principaux de méthodes d'anonymisation** :

### 1. Anonymisation par Randomisation 🎲

**Principe** : Perturber la structure du graphe en ajoutant, supprimant ou échangeant des arêtes de manière aléatoire.

**Méthodes implémentées** :
- **Random Add/Del** : Ajoute k fausses arêtes puis supprime k vraies arêtes
- **Random Switch** : Échange des paires d'arêtes pour préserver les degrés des nœuds

**Avantages** :
- Simple à implémenter
- Préservation possible de certaines propriétés (degrés avec Random Switch)

**Inconvénients** :
- Pas de garantie formelle de privacy
- Peut dégrader significativement l'utilité du graphe

### 2. K-Anonymisation 🔒

**Principe** : Assurer que chaque nœud est indistinguable d'au moins k-1 autres nœuds en termes de propriétés structurelles.

**Méthode implémentée** :
- **k-degree anonymity** : Garantit que chaque degré apparaît au moins k fois

**Avantages** :
- Garantie formelle contre les attaques basées sur les degrés
- Contrôle du niveau d'anonymat via le paramètre k

**Inconvénients** :
- Nécessite l'ajout/suppression déterministe d'arêtes
- Peut être coûteux en calcul (NP-difficile dans le cas général)

### 3. Anonymisation par Généralisation 🌐

**Principe** : Regrouper les nœuds en "super-nœuds" et les arêtes en "super-arêtes", créant ainsi une vue agrégée du graphe.

**Méthode implémentée** :
- **Clustering en super-nodes** : Groupe les nœuds en clusters de taille ≥ k

**Avantages** :
- Réduction significative de la taille du graphe publié
- Protection forte de l'identité des nœuds individuels

**Inconvénients** :
- Perte importante d'information structurelle
- Difficile de trouver le partitionnement optimal

### 4. Approches Probabilistes 🎯

**Principe** : Assigner des probabilités d'existence aux arêtes, créant un "graphe incertain".

**Méthode implémentée** :
- **(k,ε)-obfuscation** : Ajoute des arêtes potentielles avec des probabilités contrôlées

**Avantages** :
- Modélisation explicite de l'incertitude
- Bon compromis privacy/utilité
- Permet l'échantillonnage de graphes compatibles

**Inconvénients** :
- Complexité de l'échantillonnage
- Nécessite des algorithmes adaptés aux graphes probabilistes

### 5. Privacy Différentielle 🛡️

**Principe** : Garantir mathématiquement que la présence ou l'absence d'une arête (ou d'un nœud) n'affecte pas significativement la sortie de l'algorithme.

**Méthodes implémentées** :
- **EdgeFlip** : Applique le Randomized Response Technique (inverse chaque arête avec probabilité ε-dépendante)
- **Mécanisme de Laplace** : Ajoute du bruit Laplacien pour décider de l'inclusion des arêtes

**Avantages** :
- Garanties théoriques rigoureuses (ε-differential privacy)
- Composabilité des mécanismes
- Pas d'hypothèses sur les connaissances de l'attaquant

**Inconvénients** :
- Peut nécessiter beaucoup de bruit (faible ε = haute privacy = basse utilité)
- Complexité quadratique pour certaines méthodes

## 🎮 Utilisation

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Exécution de la démonstration

```bash
python graph_anonymization_demo.py
```

Cette commande :
1. Charge le graphe **Karate Club** de Zachary (34 nœuds, 78 arêtes)
2. Applique les 7 variantes des 5 méthodes d'anonymisation
3. Génère 3 fichiers de visualisation :
   - `graph_anonymization_comparison.png` : Comparaison visuelle des graphes
   - `degree_distributions.png` : Distributions des degrés
   - `metrics_comparison.png` : Métriques quantitatives

## 📊 Métriques Évaluées

L'application compare plusieurs métriques entre le graphe original et les graphes anonymisés :

- **Nombre d'arêtes** : Mesure les modifications structurelles
- **Degré moyen** : Indique la préservation de la connectivité
- **Coefficient de clustering** : Évalue la préservation des communautés
- **Densité** : Ratio arêtes existantes / arêtes possibles

## 🔍 Le Graphe Karate Club

Le graphe de Zachary est un réseau social classique en analyse de réseaux :
- **34 nœuds** : Membres d'un club de karaté
- **78 arêtes** : Relations sociales entre les membres
- **2 communautés** : Reflète une scission réelle du club

C'est un graphe de référence pour tester les algorithmes de détection de communautés et d'anonymisation.

## 🎓 Références

**Thèse** : "Anonymisation de Graphes Sociaux" (Social Graph Anonymization)
**Auteur** : NGUYEN Huu-Hiep
**Institution** : Université de Lorraine, LORIA
**Directeurs** : Abdessamad Imine, Michaël Rusinowitch
**Année** : 2016

### Publications clés mentionnées dans la thèse :

1. **Randomisation** : Ying & Wu (2008, 2011), Bonchi et al. (2011)
2. **K-anonymity** : Liu & Terzi (2008), Zhou & Pei (2008), Zou et al. (2009)
3. **Généralisation** : Hay et al. (2008), Campan & Truta (2008)
4. **Probabiliste** : Boldi et al. (2012), Mittal et al. (2013)
5. **Differential Privacy** : Dwork (2011), Sala et al. (2011), Xiao et al. (2014)

## 💡 Pour votre exposé

### Points clés à présenter :

1. **Motivation** : Pourquoi l'anonymisation naïve (suppression des IDs) ne suffit pas
   - Attaques par ré-identification basées sur les degrés
   - Exemple du graphe à 13 nœuds (Fig. 1.1 de la thèse)

2. **Trade-off Privacy/Utility** : Plus on protège, plus on distord
   - Visualiser ce trade-off avec vos résultats

3. **Évolution des approches** :
   - Méthodes ad-hoc (randomisation) → Garanties formelles (k-anonymity) → Privacy différentielle

4. **Applications pratiques** :
   - Publication de données pour la recherche
   - Partage entre organisations
   - Open data de réseaux sociaux

### Structure suggérée pour l'exposé :

1. **Introduction** (5 min)
   - Contexte : Big Data et réseaux sociaux
   - Problème : Privacy vs Utilité

2. **Les 5 types de méthodes** (15 min)
   - Pour chaque type : principe, exemple visuel, avantages/inconvénients

3. **Démonstration** (10 min)
   - Montrer les visualisations générées
   - Comparer les métriques

4. **Conclusion** (5 min)
   - État de l'art actuel
   - Défis restants (scalabilité, nouvelles attaques, etc.)

## 📈 Extensions possibles

- Ajouter d'autres graphes de test (Facebook, Email-Eu-core, etc.)
- Implémenter des métriques de privacy (re-identification rate, incorrectness)
- Ajouter des visualisations de communautés
- Tester sur des graphes de différentes tailles
- Implémenter des attaques pour quantifier la privacy

## 🛠️ Structure du code

```
graph_anonymization_demo.py
├── GraphAnonymizer : Classe principale contenant les 5 méthodes
│   ├── random_add_del()
│   ├── random_switch()
│   ├── k_degree_anonymity()
│   ├── generalization()
│   ├── probabilistic_obfuscation()
│   ├── differential_privacy_edgeflip()
│   └── differential_privacy_laplace()
│
└── GraphVisualizer : Classe pour les visualisations
    ├── plot_graph_comparison()
    ├── plot_degree_distribution()
    └── plot_metrics_comparison()
```

## ❓ Questions pour l'exposé

Préparez-vous à répondre à :
- Quelle méthode choisir selon le cas d'usage ?
- Comment mesurer concrètement la "privacy" ?
- Quelle est la différence entre edge-DP et node-DP ?
- Comment les graphes probabilistes sont-ils utilisés en pratique ?
- Y a-t-il des alternatives à la differential privacy ?

Bon exposé ! 🎉
