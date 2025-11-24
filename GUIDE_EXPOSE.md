# Guide pour votre Exposé sur l'Anonymisation de Graphes

## 📋 Résumé de ce qui a été créé

Vous disposez maintenant d'une **application complète de démonstration** qui implémente et compare les **5 types de méthodes d'anonymisation de graphes** présentés dans la thèse de NGUYEN Huu-Hiep.

### Fichiers générés :

1. **graph_anonymization_demo.py** (630 lignes) - Application principale
2. **README.md** - Documentation complète du projet
3. **GUIDE_EXPOSE.md** (ce fichier) - Guide pour votre présentation
4. **requirements.txt** - Dépendances Python

### Visualisations générées :

1. **graph_anonymization_comparison.png** (3.3 MB) - Comparaison visuelle des 7 variantes
2. **degree_distributions.png** (574 KB) - Distributions des degrés
3. **metrics_comparison.png** (443 KB) - Métriques quantitatives

---

## 🎯 Les 5 Types de Méthodes Implémentées

### 1. Anonymisation par Randomisation 🎲

**Principe** : Modifier aléatoirement la structure du graphe

**Deux variantes testées** :
- **Random Add/Del** : Ajoute 5 arêtes, supprime 5 arêtes
  - Résultat sur Karate Club : 78 arêtes (identique)
- **Random Switch** : Échange 10 paires d'arêtes
  - Résultat : 78 arêtes (degrés préservés)

**Points pour l'exposé** :
- Méthode la plus simple mais pas de garantie formelle
- Random Switch préserve les degrés mais pas les chemins
- Utilisée dans : Hay et al. (2008), Ying & Wu (2008)

---

### 2. K-Anonymisation 🔒

**Principe** : Garantir que chaque nœud est indistinguable d'au moins k-1 autres

**Variante testée** :
- **k-degree anonymity** (k=3)
  - Résultat : 92 arêtes (ajout de 14 arêtes)

**Points pour l'exposé** :
- Protection formelle contre attaques par degrés
- Doit ajouter/supprimer des arêtes de manière déterministe
- NP-difficile en général
- Utilisée dans : Liu & Terzi (2008), Zhou & Pei (2008)

---

### 3. Généralisation 🌐

**Principe** : Regrouper les nœuds en "super-nœuds"

**Variante testée** :
- **Clustering** (taille minimale k=3)
  - Résultat : 3 super-nœuds (au lieu de 34 nœuds)

**Points pour l'exposé** :
- Réduction drastique de la taille : 34 → 3 nœuds
- Forte protection mais perte d'information importante
- Produit un graphe agrégé, pas le graphe original
- Utilisée dans : Hay et al. (2008), Campan & Truta (2008)

---

### 4. Approches Probabilistes 🎯

**Principe** : Créer un "graphe incertain" avec probabilités sur les arêtes

**Variante testée** :
- **(k,ε)-obfuscation** (k=3, ε=0.1)
  - Résultat : 316 arêtes (dont beaucoup avec faible probabilité)

**Points pour l'exposé** :
- Modélise explicitement l'incertitude
- Permet l'échantillonnage de graphes compatibles
- Bon compromis privacy/utilité
- Utilisée dans : Boldi et al. (2012), Mittal et al. (2013)

---

### 5. Privacy Différentielle 🛡️

**Principe** : Garantie mathématique formelle (ε-differential privacy)

**Deux variantes testées** :
- **EdgeFlip** (ε=1.0)
  - Résultat : 208 arêtes (inversion probabiliste)
- **Laplace Mechanism** (ε=0.5)
  - Résultat : 225 arêtes (ajout de bruit)

**Points pour l'exposé** :
- Garanties théoriques les plus fortes
- Composabilité des mécanismes
- Pas d'hypothèses sur l'attaquant
- Trade-off : ε faible = haute privacy mais basse utilité
- Utilisée dans : Dwork (2011), Sala et al. (2011), Xiao et al. (2014)

---

## 📊 Résultats Observés

### Nombre d'arêtes (graphe original : 78)

| Méthode | Arêtes | Variation |
|---------|--------|-----------|
| Random Add/Del | 78 | 0% |
| Random Switch | 78 | 0% |
| k-degree (k=3) | 92 | +18% |
| Généralisation | 3 super-nœuds | N/A |
| (k,ε)-obf | 316 | +305% |
| EdgeFlip (ε=1.0) | 208 | +167% |
| Laplace (ε=0.5) | 225 | +188% |

**Observations clés** :
- Random Switch préserve parfaitement le nombre d'arêtes ET les degrés
- k-anonymity ajoute peu d'arêtes (+18%)
- Méthodes probabilistes et DP ajoutent beaucoup d'arêtes (pour créer de l'incertitude)
- Généralisation compresse radicalement le graphe

---

## 🎤 Structure Suggérée pour l'Exposé (30-35 min)

### Introduction (5 min)

**Slide 1 : Titre**
- Titre : "État de l'Art de l'Anonymisation de Graphes Sociaux"
- Sous-titre : "Revue basée sur la thèse de NGUYEN Huu-Hiep (2016)"
- Votre nom

**Slide 2 : Contexte**
- Explosion des réseaux sociaux (Facebook, Twitter, LinkedIn...)
- Big Data : besoin de partager les données pour la recherche
- Problème : protéger la vie privée des utilisateurs

**Slide 3 : Le Problème**
- Montrer l'exemple de la Figure 1.1 de la thèse
- Attaque par ré-identification basée sur les degrés
- L'anonymisation naïve (suppression des IDs) ne suffit PAS

**Question rhétorique** : "Comment publier des graphes sociaux tout en protégeant la vie privée ?"

---

### Les 5 Familles de Méthodes (20 min - 4 min par méthode)

**Pour chaque méthode :**

1. **Principe** (30 sec)
   - Une phrase simple pour expliquer l'idée

2. **Exemple visuel** (1 min 30)
   - Montrer la comparaison Original vs Anonymisé
   - Pointer les différences visuelles

3. **Résultats quantitatifs** (1 min)
   - Montrer les métriques (nb arêtes, degrés, clustering...)
   - Interpréter les changements

4. **Avantages / Inconvénients** (1 min)
   - Forces et faiblesses de l'approche
   - Quand l'utiliser ?

**Ordre suggéré :**
1. Randomisation (la plus simple)
2. K-anonymisation (garantie formelle)
3. Généralisation (approche radicale)
4. Probabiliste (compromis)
5. Privacy Différentielle (gold standard actuel)

---

### Comparaison et Discussion (7 min)

**Slide : Tableau Comparatif**

| Critère | Randomisation | K-anonymity | Généralisation | Probabiliste | Diff. Privacy |
|---------|---------------|-------------|----------------|--------------|---------------|
| **Garantie formelle** | ❌ | ✅ (k-anonymity) | ⚠️ | ⚠️ | ✅ (ε-DP) |
| **Préservation utilité** | ✅ | ✅ | ❌ | ✅ | ⚠️ |
| **Simplicité** | ✅ | ⚠️ | ⚠️ | ❌ | ⚠️ |
| **Scalabilité** | ✅ | ❌ (NP-dur) | ⚠️ | ⚠️ | ❌ (O(n²)) |

**Slide : Trade-off Privacy/Utility**
- Montrer le graphique metrics_comparison.png
- Expliquer qu'il y a toujours un compromis
- Plus on protège, plus on distord

**Questions de recherche ouvertes :**
- Comment mesurer précisément la "privacy" ?
- Peut-on avoir privacy ET utility ?
- Comment adapter à des graphes dynamiques ?
- Nouvelles attaques ?

---

### Démonstration (3 min)

**Option 1 : Vidéo**
- Enregistrer un screencast du script qui tourne
- Montrer la génération des visualisations en temps réel

**Option 2 : Images**
- Montrer les 3 PNG générés
- Zoomer sur des détails intéressants

**Ce qu'il faut montrer :**
1. Le graphe Karate Club original
2. Une transformation visuelle frappante (ex: Généralisation)
3. La comparaison des distributions de degrés
4. Les métriques quantitatives

---

### Conclusion (2-3 min)

**Slide : Récapitulatif**
- 5 grandes familles de méthodes
- Évolution : méthodes ad-hoc → garanties formelles
- Privacy Différentielle : état de l'art actuel

**Slide : Perspectives**
- Graphes dynamiques et streaming
- Graphes avec attributs riches
- Privacy pour d'autres structures (hypergraphes, etc.)
- Applications pratiques (Open Data, partage inter-organisations)

**Slide : Questions**
- "Merci de votre attention"
- Vos coordonnées ou références

---

## 💡 Conseils pour la Présentation

### Avant l'exposé

✅ **Testez votre setup**
- Vérifiez que les images s'affichent correctement
- Préparez un backup PDF de vos slides

✅ **Chronométrez-vous**
- Répétez votre présentation
- Ajustez pour tenir dans le temps imparti

✅ **Anticipez les questions**
- Voir section "Questions Fréquentes" ci-dessous

### Pendant l'exposé

✅ **Interaction avec le public**
- Posez des questions rhétoriques
- Demandez : "Qui utilise Facebook ? LinkedIn ?"

✅ **Storytelling**
- Commencez par une anecdote (ex: le scandale Cambridge Analytica)
- Utilisez des exemples concrets

✅ **Gestion du temps**
- Gardez un œil sur l'horloge
- Préparez des "slides de backup" qu'on peut sauter si nécessaire

---

## ❓ Questions Fréquentes à Anticiper

### Q1 : "Quelle méthode est la meilleure ?"

**Réponse** : Ça dépend du contexte !
- **Pour la recherche** : Privacy Différentielle (garanties formelles)
- **Pour la publication rapide** : Randomisation (simple et rapide)
- **Pour la protection maximale** : Généralisation (mais perte d'utilité)
- **Pour le compromis** : Approches Probabilistes

---

### Q2 : "Comment on mesure concrètement la 'privacy' ?"

**Réponse** : Plusieurs métriques existent :
1. **Min-entropy** : Quantifie la plus grande probabilité de ré-identification
2. **Shannon entropy** : Mesure l'incertitude globale
3. **Incorrectness** : Nombre de mauvaises suppositions de l'attaquant
4. **ε dans DP** : Borne sur le ratio de probabilités

Dans notre démo, on se concentre sur l'**utilité** (préservation de la structure), mais mesurer la privacy rigoureusement nécessite de simuler des attaques.

---

### Q3 : "Quelle est la différence entre edge-DP et node-DP ?"

**Réponse** :
- **Edge-DP** : Protège la présence/absence d'une arête
  - Deux graphes voisins diffèrent par une arête
  - Plus facile à atteindre

- **Node-DP** : Protège la présence/absence d'un nœud entier
  - Deux graphes voisins diffèrent par un nœud et toutes ses arêtes
  - Beaucoup plus difficile (sensibilité = degré max)

La plupart des méthodes de la thèse se concentrent sur **edge-DP**.

---

### Q4 : "Ces méthodes marchent-elles sur des graphes de millions de nœuds ?"

**Réponse** : Ça dépend !
- **Scalables** : Randomisation, k-degree (heuristiques)
- **Moyennement scalables** : Probabilistes, certaines méthodes DP
- **Pas scalables** : Généralisation (clustering coûteux), EdgeFlip (O(n²))

Les défis de **scalabilité** sont un axe de recherche actif. Méthodes récentes : HRG-FixedTree, 1K-series (mentionnés dans la thèse).

---

### Q5 : "Peut-on appliquer plusieurs méthodes en séquence ?"

**Réponse** : Oui, mais attention !
- Pour la **Privacy Différentielle** : OUI, grâce à la composabilité
  - ε_total = ε₁ + ε₂ + ... (composition séquentielle)

- Pour les **autres méthodes** : Possible mais pas de garantie formelle
  - Peut améliorer la privacy empiriquement
  - Risque de dégrader davantage l'utilité

---

### Q6 : "Quels sont les logiciels/librairies disponibles ?"

**Réponse** :
- **NetworkX** (Python) : Manipulation de graphes, mais pas d'anonymisation intégrée
- **Google DP Library** : Pour la privacy différentielle générale
- **OpenDP** : Framework moderne pour DP
- **Implementations académiques** : Souvent prototypes dans les papiers

Il n'existe **pas encore** de librairie standard unifiée pour l'anonymisation de graphes.

---

## 🎓 Références Clés à Citer

### La Thèse

**NGUYEN Huu-Hiep** (2016). *Anonymisation de Graphes Sociaux* (Social Graph Anonymization).
Thèse de doctorat, Université de Lorraine, LORIA.
Directeurs : Abdessamad Imine, Michaël Rusinowitch.

### Papers Fondateurs (par catégorie)

**Randomisation :**
- Hay et al. (2008) - Resisting Structural Re-identification
- Ying & Wu (2008, 2011) - Randomizing Social Networks

**K-anonymity :**
- Liu & Terzi (2008) - k-degree Anonymization
- Zhou & Pei (2008) - k-neighborhood
- Zou et al. (2009) - k-automorphism

**Généralisation :**
- Hay et al. (2008) - Generalization Strategy
- Campan & Truta (2008) - Clustering Approach

**Probabiliste :**
- Boldi et al. (2012) - (k,ε)-obfuscation
- Mittal et al. (2013) - RandWalk

**Differential Privacy :**
- Dwork (2011) - Algorithmic Foundations of DP
- Sala et al. (2011) - Sharing Graphs using DP Graph Models
- Xiao et al. (2014) - HRG-MCMC

---

## 📁 Organisation des Fichiers pour l'Exposé

```
Votre_Presentation/
│
├── slides.pdf ou slides.pptx        # Vos slides
│
├── images/                           # Dossier d'images
│   ├── graph_anonymization_comparison.png
│   ├── degree_distributions.png
│   └── metrics_comparison.png
│
├── demo/                             # Code de démonstration (optionnel)
│   ├── graph_anonymization_demo.py
│   ├── requirements.txt
│   └── README.md
│
└── references/                       # Papiers importants (optionnel)
    └── Nguyen16_thesis.pdf
```

---

## 🚀 Pour Aller Plus Loin (après l'exposé)

Si vous voulez enrichir la démonstration :

1. **Ajouter d'autres graphes** :
   - Facebook ego-network
   - Email-Eu-core
   - Ca-GrQc (collaboration)

2. **Implémenter des métriques de privacy** :
   - Simuler des attaques de ré-identification
   - Calculer l'incorrectness

3. **Ajouter des visualisations** :
   - Détection de communautés avant/après
   - Heatmap des matrices d'adjacence

4. **Créer une interface web** :
   - Streamlit ou Plotly Dash
   - Permettre à l'utilisateur de choisir les paramètres

---

## ✅ Checklist Finale

**24h avant l'exposé** :
- [ ] Slides finalisés et testés
- [ ] Images exportées en haute résolution
- [ ] Chronométrage fait (avec marge de 2-3 min)
- [ ] Réponses aux questions préparées
- [ ] Backup des fichiers sur clé USB + cloud

**Le jour J** :
- [ ] Arriver 15 min en avance
- [ ] Tester vidéoprojecteur/écran
- [ ] Vérifier le son (si vidéo)
- [ ] Avoir de l'eau à disposition
- [ ] Respirer et sourire 😊

---

## 📞 Besoin d'Aide ?

Si vous avez des questions techniques sur le code :
- Consultez le README.md
- Lisez les commentaires dans graph_anonymization_demo.py
- Testez différents paramètres (k, epsilon, etc.)

**Bonne chance pour votre exposé ! 🎉**

---

*Document généré le 24 novembre 2025*
*Basé sur la thèse de NGUYEN Huu-Hiep (2016)*
