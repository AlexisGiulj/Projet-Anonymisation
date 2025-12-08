# Application Interactive d'Anonymisation de Graphes Sociaux

Application web interactive basée sur la thèse **"Anonymisation de Graphes Sociaux"** par **NGUYEN Huu-Hiep** (Université de Lorraine, 2016).

## 🌟 Caractéristiques

- **Interface Streamlit intuitive** : Visualisation et interaction en temps réel
- **Thèse PDF intégrée** : Références académiques directement dans l'application
- **7 méthodes d'anonymisation** implémentées selon la thèse
- **Métriques détaillées** : Privacy, utilité, et analyse comparative
- **Visualisations interactives** : Graphes, distributions, métriques

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

**Formule** : $|\{v \in V : \deg(v) = d\}| \geq k \quad \forall d$

**Avantages** :
- Garantie formelle contre les attaques basées sur les degrés
- Contrôle du niveau d'anonymat via le paramètre k

**Inconvénients** :
- Nécessite l'ajout/suppression déterministe d'arêtes
- NP-difficile dans le cas général

### 3. Anonymisation par Généralisation 🌐

**Principe** : Regrouper les nœuds en "super-nœuds" et les arêtes en "super-arêtes", créant ainsi une vue agrégée du graphe.

**Méthode implémentée** :
- **Clustering en super-nodes** : Utilise Label Propagation puis ajuste pour garantir $|C_i| \geq k$

**Algorithme** :
1. Label Propagation pour détecter les communautés naturelles
2. Fusion/division des clusters pour respecter la contrainte de taille minimale

**Avantages** :
- Réduction significative de la taille du graphe publié
- Protection forte de l'identité des nœuds individuels

**Inconvénients** :
- Perte importante d'information structurelle
- Difficile de trouver le partitionnement optimal

### 4. Approches Probabilistes 🎯

**Principe** : Assigner des probabilités d'existence aux arêtes, créant un "graphe incertain".

**Méthodes implémentées** :

#### (k,ε)-obfuscation (Boldi et al. 2012)
Implémentation conforme à l'algorithme original de Boldi et al. utilisant une **distribution normale tronquée** $R_\sigma$ sur $[0,1]$.

**Algorithme de Construction** :
1. Pour chaque nœud, identifier les k voisins candidats
2. Assigner des probabilités via distribution normale tronquée centrée
3. Garantir l'entropie minimale : $H(N_k(v)) \geq \log(k) - \varepsilon$

**⚠️ Limitation connue** : Vulnérable au threshold attack (voir thèse p.75)

#### MaxVar (Variance Maximizing Scheme)
Solution au threshold attack via optimisation quadratique.

**Programme** : $\min \sum_i p_i^2$ sous contrainte $\sum_{v \in N(u)} p_{uv} = \deg(u)$

**Avantages** :
- Résiste au threshold attack
- Probabilités dispersées (pas de concentration en 0/1)
- Arêtes "nearby" (distance 2) pour minimiser la distance d'édition

**Inconvénients** :
- Complexité $O(m^2)$

### 5. Privacy Différentielle 🛡️

**Principe** : Garantir mathématiquement que la présence ou l'absence d'une arête n'affecte pas significativement la sortie.

**Définition** : $P[\mathcal{A}(G) = O] \leq e^\varepsilon \cdot P[\mathcal{A}(G') = O]$

**Méthodes implémentées** :
- **EdgeFlip** : Randomized Response Technique avec $s = \frac{2}{e^\varepsilon + 1}$
- **Laplace** : Mécanisme de Laplace avec bruit $\sim \text{Lap}(\Delta f / \varepsilon)$

**Avantages** :
- Garanties théoriques rigoureuses (ε-differential privacy)
- Composabilité des mécanismes
- Pas d'hypothèses sur les connaissances de l'attaquant

**Inconvénients** :
- Trade-off privacy/utilité : faible ε = haute privacy = basse utilité
- Complexité $O(n^2)$ pour certaines méthodes

## 🎮 Utilisation

### Installation des dépendances

```bash
pip install streamlit networkx matplotlib numpy scipy pandas
```

Ou via requirements.txt :

```bash
pip install -r requirements.txt
```

### Lancement de l'application

```bash
streamlit run graph_anonymization_app.py
```

L'application s'ouvrira dans votre navigateur à `http://localhost:8501`

### Fonctionnalités de l'interface

1. **Sélection du graphe** : Karate Club, ou graphes aléatoires
2. **Choix de la méthode** : 7 méthodes d'anonymisation disponibles
3. **Configuration des paramètres** : k, ε, nombre d'arêtes potentielles, etc.
4. **Visualisation** : Graphes comparatifs avec code couleur pour les probabilités
5. **Métriques** : Analyse détaillée de la privacy et de l'utilité
6. **Références thèse** : Liens directs vers les sections pertinentes du PDF

## 📊 Métriques Évaluées

### Métriques d'Utilité

- **Distance d'édition** : Nombre d'arêtes modifiées (ajoutées + supprimées)
- **Degré moyen** : Préservation de la connectivité
- **Coefficient de clustering** : Préservation des communautés
- **Densité** : Ratio arêtes existantes / arêtes possibles
- **Diamètre** : Plus long plus court chemin
- **Corrélation des degrés** : Similarité des distributions de degrés

### Métriques de Privacy

- **k-anonymity** : Nombre minimum d'occurrences de chaque degré
- **Variance des probabilités** : Résistance au threshold attack (MaxVar)
- **Taux de reconstruction** : Efficacité du threshold attack
- **Epsilon** : Budget de differential privacy

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

### Publications clés implémentées :

1. **Randomisation** : Ying & Wu (2008, 2011), Bonchi et al. (2011)
2. **K-anonymity** : Liu & Terzi (2008), Zhou & Pei (2008)
3. **Généralisation** : Hay et al. (2008), Campan & Truta (2008)
4. **Probabiliste** : **Boldi et al. (2012)**, Mittal et al. (2013)
5. **Differential Privacy** : Dwork (2011), Sala et al. (2011)

### Références directes dans la thèse :

- **p.30-32** : k-Anonymity et k-degree anonymity
- **p.40** : Généralisation par super-nodes
- **p.50-52** : Differential Privacy (EdgeFlip)
- **p.70-75** : (k,ε)-obfuscation et threshold attack
- **p.80-85** : MaxVar (solution au threshold attack)

## 📈 Détails d'Implémentation

### Algorithme de Boldi et al. (Distribution Normale Tronquée)

Contrairement à la formule simplifiée $p = 1 - \varepsilon/k$, l'implémentation suit l'algorithme original de Boldi et al. (2012) :

**Distribution $R_\sigma$** : Normale tronquée sur $[0,1]$ avec écart-type $\sigma$ calculé pour garantir :
$$H(N_k(v)) = -\sum_i p_i \log(p_i) \geq \log(k) - \varepsilon$$

**Processus** :
1. Pour chaque nœud $v$, identifier $N_k(v)$ (k voisins candidats)
2. Tirer $k$ valeurs de $R_\sigma$ et normaliser
3. Assigner ces probabilités normalisées aux arêtes candidates
4. Vérifier la contrainte d'entropie

**Avantage** : Distribution plus réaliste que la formule uniforme
**Inconvénient** : Plus sensible au threshold attack (d'où l'importance de MaxVar)

### MaxVar : Résolution du Threshold Attack

MaxVar résout un programme quadratique pour disperser les probabilités :

```python
# Objectif : minimiser la somme des p_i^2
# Contrainte : somme des probabilités sortantes = degré du nœud
# Résolution : SLSQP (Sequential Least Squares Programming)
```

**Résultat** : Probabilités autour de 0.5 au lieu de 0/1, rendant le threshold attack inefficace.

## 🛠️ Structure du Projet

```
GraphAnonymizationDemo/
├── graph_anonymization_app.py      # Application Streamlit principale
├── method_details.py                # Documentation attaques & garanties
├── definitions_and_attacks.py       # Définitions et dictionnaires
├── thesis_references.py             # Références vers la thèse
├── assets/
│   └── thesis.pdf                   # Thèse PDF intégrée
├── requirements.txt                 # Dépendances Python
└── README.md                        # Ce fichier
```

## 💡 Utilisation Pédagogique

### Points clés à présenter :

1. **Motivation** : Pourquoi l'anonymisation naïve (suppression des IDs) ne suffit pas
   - Attaques par ré-identification basées sur les degrés
   - Attaques par connaissance du voisinage

2. **Trade-off Privacy/Utility** : Plus on protège, plus on distord
   - Visualiser ce trade-off avec les métriques de l'application
   - Comparer distance d'édition vs garanties de privacy

3. **Évolution des approches** :
   - Méthodes ad-hoc (randomisation)
   - → Garanties formelles (k-anonymity)
   - → Privacy différentielle (gold standard)

4. **Cas d'usage réel** :
   - Publication de données pour la recherche médicale
   - Partage de graphes sociaux entre organisations
   - Open data de réseaux de mobilité

### Structure suggérée pour présentation :

1. **Introduction** (5 min)
   - Contexte : Big Data et réseaux sociaux
   - Problème : Privacy vs Utilité
   - Démo rapide de l'application

2. **Les 5 types de méthodes** (15 min)
   - Pour chaque type : principe, exemple visuel, avantages/inconvénients
   - Focus sur threshold attack et MaxVar

3. **Démonstration interactive** (10 min)
   - Montrer les visualisations en direct
   - Comparer les métriques
   - Tester différents paramètres

4. **Conclusion** (5 min)
   - État de l'art actuel
   - Défis restants (scalabilité, nouvelles attaques, ML-based attacks)

## ❓ Questions Fréquentes

**Q : Quelle méthode choisir selon le cas d'usage ?**
- Privacy maximale : Differential Privacy (EdgeFlip)
- Préservation structure : MaxVar
- Simplicité : k-degree anonymity
- Réduction taille : Generalization

**Q : Pourquoi Boldi et al. au lieu de la formule simplifiée ?**
- Distribution normale tronquée plus réaliste
- Conforme à la publication originale
- Meilleure modélisation de l'incertitude

**Q : Différence entre (k,ε)-obf et MaxVar ?**
- (k,ε)-obf : Garantit entropie, mais vulnérable au seuillage
- MaxVar : Maximise variance, résiste au threshold attack

**Q : Comment mesurer concrètement la "privacy" ?**
- Métriques formelles : k-anonymity, ε-differential privacy
- Métriques empiriques : taux de reconstruction, distance d'édition
- Simulations d'attaques : degree attack, neighborhood attack

## 📝 Licence

Ce projet est développé à des fins pédagogiques basé sur la thèse publique de NGUYEN Huu-Hiep.

## 🤝 Contributions

Les contributions sont les bienvenues ! Pour toute amélioration :
1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push et créer une Pull Request
