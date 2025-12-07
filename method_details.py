"""
Informations détaillées sur les attaques, garanties, avantages et inconvénients
des méthodes d'anonymisation de graphes
"""

ATTACKS_AND_GUARANTEES = {
    "Random Add/Del": {
        "attacks_protected": [
            {
                "name": "Attaques Passives (Degree-based)",
                "description": "Protection contre les attaquants qui connaissent le degré d'un nœud cible. En modifiant aléatoirement les arêtes, le degré des nœuds change, rendant la ré-identification difficile."
            },
            {
                "name": "Attaques par Voisinage Partiel",
                "description": "Protège contre les adversaires ayant une connaissance partielle du voisinage d'un nœud, car les arêtes sont modifiées aléatoirement."
            }
        ],
        "attacks_vulnerable": [
            {
                "name": "Attaques Actives (Backstrom et al.)",
                "description": "Vulnérable aux attaques où l'adversaire insère des motifs structurels avant la publication (nœuds sybils avec patterns spécifiques)."
            },
            {
                "name": "Attaques par Similarité",
                "description": "Les similarités entre nœuds peuvent toujours être exploitées car la méthode ne les masque pas complètement."
            }
        ],
        "advantages": [
            "✅ **Simplicité** : Très facile à implémenter et à comprendre",
            "✅ **Rapidité** : Complexité O(k), très efficace",
            "✅ **Préservation du nombre d'arêtes** : |E'| = |E| maintient la densité globale",
            "✅ **Protection probabiliste** : Scores de privacy >65% dans les benchmarks",
            "✅ **Pas de coût en calcul** : Ne nécessite pas d'algorithmes complexes"
        ],
        "disadvantages": [
            "❌ **Absence de garanties formelles** : Pas de borne mathématique sur la protection",
            "❌ **Modification globale** : Change de nombreuses propriétés structurelles fondamentales",
            "❌ **Vulnérable aux attaques sophistiquées** : Ne résiste pas aux attaques actives",
            "❌ **Perte d'utilité** : Les propriétés du graphe (clustering, chemins) sont significativement altérées",
            "❌ **Pas de contrôle fin** : Impossible de cibler spécifiquement certaines zones sensibles"
        ],
        "karate_example": """
### Exemple sur le Graphe Karate Club

**Contexte** : Le Karate Club a 34 nœuds et 78 arêtes, avec deux communautés principales.

**Avec k=20** :
- 🔴 **20 arêtes supprimées** : Par exemple, des liens importants entre les leaders (nœud 0 "Mr. Hi" et nœud 33 "John A") pourraient être supprimés
- 🟢 **20 arêtes ajoutées** : De fausses connexions inter-communautés sont créées, brouillant la structure des deux groupes
- 📊 **Impact** : Le degré des leaders change (Mr. Hi passe de deg=16 à deg≈14-18), rendant leur identification incertaine
- ⚠️ **Problème** : La structure communautaire devient moins claire, réduisant l'utilité pour l'analyse sociologique
"""
    },

    "Random Switch": {
        "attacks_protected": [
            {
                "name": "Attaques par Voisinage",
                "description": "Protège contre les adversaires connaissant le voisinage d'un nœud, car les connexions sont réarrangées."
            },
            {
                "name": "Attaques par Sous-graphes",
                "description": "Les motifs locaux (triangles, cliques) sont modifiés, empêchant leur utilisation pour la ré-identification."
            }
        ],
        "attacks_vulnerable": [
            {
                "name": "Attaques par Degré",
                "description": "TRÈS VULNÉRABLE : La séquence de degrés est préservée, donc un attaquant connaissant uniquement le degré peut toujours restreindre l'ensemble des candidats."
            },
            {
                "name": "Attaques par Degré + Information Auxiliaire",
                "description": "Si l'attaquant combine degré + localisation géographique par exemple, la ré-identification reste possible."
            }
        ],
        "advantages": [
            "✅ **Préservation des degrés** : deg_G'(v) = deg_G(v) ∀v, propriété TRÈS importante",
            "✅ **Nombre d'arêtes constant** : |E'| = |E|",
            "✅ **Meilleure utilité que Add/Del** : Nombreuses statistiques de degré préservées",
            "✅ **Protection probabiliste forte** : Scores de privacy >80% dans les benchmarks",
            "✅ **Efficacité** : Complexité O(k), rapide"
        ],
        "disadvantages": [
            "❌ **Vulnérabilité aux attaques par degré** : L'ensemble d'anonymat est limité aux nœuds de même degré",
            "❌ **Pas de garantie formelle** : Protection seulement probabiliste",
            "❌ **Modification des chemins** : Les plus courts chemins et diamètre du graphe changent",
            "❌ **Perte de structure locale** : Les triangles et motifs locaux sont détruits",
            "❌ **Moins efficace sur graphes réguliers** : Si beaucoup de nœuds ont le même degré, peu de switches possibles"
        ],
        "karate_example": """
### Exemple sur le Graphe Karate Club

**Contexte** : Distribution des degrés dans Karate Club : deg ∈ [1, 17], avec Mr. Hi (deg=16) quasi-unique.

**Avec k=25 switches** :
- 🔄 **Échange de connexions** : Par exemple, si (1,0) et (2,8) existent, on crée (1,2) et (0,8)
- ✅ **Degrés préservés** : Mr. Hi garde deg=16, facilitant son identification !
- 📊 **Impact** : Le nœud 0 reste facilement identifiable car il est le seul (ou presque) avec deg=16
- ⚠️ **Limitation** : Pour k-anonymiser le degré 16, il faudrait combiner avec k-degree anonymity
- 💡 **Utilité** : Excellente pour l'analyse de degrés, mais structure communautaire perturbée
"""
    },

    "k-degree anonymity": {
        "attacks_protected": [
            {
                "name": "Attaques par Degré (Degree Attack)",
                "description": "PROTECTION PRINCIPALE : Garantit que chaque degré apparaît ≥k fois. Un attaquant ne peut identifier un nœud qu'avec probabilité ≤1/k."
            },
            {
                "name": "Attaques Passives Simples",
                "description": "Résiste aux adversaires internes qui tentent de se trouver dans le graphe publié en utilisant leur degré."
            }
        ],
        "attacks_vulnerable": [
            {
                "name": "Attaques par Sous-graphe de Voisinage",
                "description": "VULNÉRABLE : Les méthodes basées uniquement sur les degrés ne protègent pas contre les adversaires ayant connaissance de la structure du sous-graphe de voisinage."
            },
            {
                "name": "Attaques par k-automorphisme",
                "description": "Un adversaire ayant connaissance des voisinages à 2-sauts peut contourner k-degree anonymity."
            },
            {
                "name": "Attaques par Attributs Combinés",
                "description": "Si l'attaquant combine degré + attributs des nœuds (âge, localisation), la protection est insuffisante."
            }
        ],
        "advantages": [
            "✅ **Garantie formelle de k-anonymité** : P(ré-identification | degré) ≤ 1/k",
            "✅ **Protection quantifiable** : On peut prouver mathématiquement le niveau de protection",
            "✅ **Bonne utilité pour analyses statistiques** : Les distributions de degrés restent relativement proches",
            "✅ **Adapté aux graphes hétérogènes** : Fonctionne même si distribution de degrés très variée",
            "✅ **Base pour méthodes avancées** : k-automorphism, k-isomorphism étendent ce principe"
        ],
        "disadvantages": [
            "❌ **NP-difficile** : Trouver le nombre minimum d'arêtes à ajouter est NP-difficile",
            "❌ **Heuristiques approximatives** : Les implémentations pratiques sont des heuristiques",
            "❌ **Ajout d'arêtes uniquement** : Typiquement, on ajoute des arêtes sans en supprimer, augmentant la densité",
            "❌ **Protection limitée** : Ne protège QUE contre les attaques par degré",
            "❌ **Coût en utilité** : L'ajout d'arêtes peut créer de faux triangles et fausser les analyses",
            "❌ **Complexité O(n²)** : Calcul coûteux pour grands graphes"
        ],
        "karate_example": """
### Exemple sur le Graphe Karate Club

**Problème identifié** : Mr. Hi (nœud 0) a deg=16, presque unique. John A (nœud 33) a deg=17, unique.

**Avec k=2** :
- 🎯 **Objectif** : Chaque degré doit apparaître ≥2 fois
- 🔴 **Degrés uniques** : deg=17 (nœud 33), deg=12 (nœud 2), deg=10 (nœud 23), etc.
- 🟢 **Solution** :
  - Ajouter 1 arête au nœud 0 → deg=17 (maintenant 2 nœuds ont deg=17)
  - Ajouter 2 arêtes au nœud 1 → deg=12 (maintenant 2 nœuds ont deg=12)
  - Continuer pour tous les degrés uniques...
- 📊 **Résultat** : ~15-20 arêtes ajoutées, risque de ré-identification réduit de 100% à 50% pour ces nœuds
- ⚠️ **Limitation** : Mr. Hi et John A restent identifiables s'ils sont les SEULS à avoir des degrés aussi élevés dans leur communauté
- 💡 **Note** : k=3 ou k=4 offrirait une meilleure protection mais au coût de plus d'arêtes ajoutées
"""
    },

    "Generalization": {
        "attacks_protected": [
            {
                "name": "Attaques par Ré-identification Totale",
                "description": "PROTECTION MAXIMALE : Les individus sont cachés dans des groupes de ≥k personnes, rendant l'identification individuelle impossible."
            },
            {
                "name": "Attaques par Degré, Voisinage et Motifs",
                "description": "Toutes les attaques basées sur la structure locale sont neutralisées car les détails individuels disparaissent."
            },
            {
                "name": "Attaques Actives et Passives",
                "description": "Résiste même aux attaques sophistiquées de Backstrom et al. car les nœuds individuels ne sont plus visibles."
            }
        ],
        "attacks_vulnerable": [
            {
                "name": "Attaques par Analyse des Super-nœuds",
                "description": "Si k est trop petit (k=2-3), l'analyse des tailles de clusters peut révéler des informations."
            },
            {
                "name": "Attaques par Composition",
                "description": "Si plusieurs versions du graphe généralisé sont publiées, un attaquant pourrait croiser les informations."
            }
        ],
        "advantages": [
            "✅ **Protection maximale de l'identité** : Chaque individu dans un groupe de ≥k",
            "✅ **Garantie formelle** : P(ré-identification) ≤ 1/k_{min}",
            "✅ **Résiste aux attaques sophistiquées** : Même Backstrom's active attacks échouent",
            "✅ **Réduction de taille** : Le graphe publié est beaucoup plus petit (m super-nœuds << n nœuds)",
            "✅ **Efficace pour analyses macro** : Patterns inter-communautés préservés",
            "✅ **Protection des attributs** : En généralisant aussi les attributs, protection complète"
        ],
        "disadvantages": [
            "❌ **PERTE MAJEURE D'INFORMATION** : Structural Information Loss (SIL) élevé",
            "❌ **Impossible d'analyser la structure locale** : Les triangles, chemins courts, etc. sont perdus",
            "❌ **Graphe non exploitable pour certaines analyses** : Centralité, clustering local impossibles",
            "❌ **Choix du clustering critique** : La qualité dépend fortement de l'algorithme de clustering",
            "❌ **Complexité** : O(n²) à O(n³) selon l'algorithme de clustering utilisé",
            "❌ **Perte de 72% des arêtes (en termes d'information)** : Benchmarks sur Karate Club"
        ],
        "karate_example": """
### Exemple sur le Graphe Karate Club

**Contexte** : 34 nœuds, 2 communautés naturelles (fission du club autour de Mr. Hi vs John A).

**Avec k=4 (taille minimale de cluster)** :
- 🔵 **Cluster 1** : Groupe autour de Mr. Hi (nœuds 0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21) → ~11 nœuds
- 🔴 **Cluster 2** : Groupe autour de John A (nœuds 33, 32, 31, 30, 29, 28, 27, 24, 23) → ~9 nœuds
- 🟢 **Cluster 3** : Membres intermédiaires (nœuds 5, 6, 10, 16, 4) → ~5 nœuds
- 🟡 **Cluster 4** : Membres périphériques (reste) → ~9 nœuds

**Super-graphe résultant** :
- 📊 **4 super-nœuds** (au lieu de 34)
- 🔗 **Arêtes intra-cluster** : ~60 arêtes internes (78% des arêtes originales)
- 🔗 **Arêtes inter-cluster** : ~18 arêtes entre clusters
- 🎭 **Anonymisation** : Mr. Hi est caché parmi 11 personnes, John A parmi 9
- ⚠️ **Perte** : Impossible de savoir qui est connecté à qui à l'intérieur des clusters
- 💡 **Utilité** : Bonne pour comprendre les relations entre groupes, mais analyse individuelle impossible
"""
    },

    "Probabilistic": {
        "attacks_protected": [
            {
                "name": "Attaques par Identification Exacte",
                "description": "L'incertitude injectée empêche toute identification avec certitude absolue. Même si on retrouve un nœud, on ne peut être sûr de ses connexions."
            },
            {
                "name": "Attaques par Degré et Voisinage",
                "description": "Les probabilités sur les arêtes masquent les degrés et voisinages réels."
            }
        ],
        "attacks_vulnerable": [
            {
                "name": "Attaques par Analyse Probabiliste",
                "description": "Un adversaire sophistiqué peut utiliser les probabilités elles-mêmes pour inférer des informations."
            },
            {
                "name": "Attaques par Échantillonnage Multiple",
                "description": "Si plusieurs graphes sont générés depuis le même graphe incertain, un attaquant pourrait croiser les informations."
            }
        ],
        "advantages": [
            "✅ **Incertitude quantifiable** : Entropie minimale garantie H(N_k(v)) ≥ log(k) - ε",
            "✅ **Meilleure utilité que randomisation pure** : À niveau de protection égal, moins de modifications",
            "✅ **Perturbation fine** : On ajoute/supprime des arêtes partiellement (avec probabilités) plutôt que totalement",
            "✅ **Espérance des degrés préservée** : E[deg(v)] proche de deg_original(v)",
            "✅ **Flexibilité** : Paramètres k et ε ajustables selon besoins privacy/utilité",
            "✅ **Théorie de l'information** : Fondations mathématiques solides"
        ],
        "disadvantages": [
            "❌ **Complexité d'utilisation** : Le graphe incertain est plus difficile à analyser qu'un graphe standard",
            "❌ **Algorithmes spécialisés nécessaires** : Les outils classiques de NetworkX ne fonctionnent pas directement",
            "❌ **Interprétation délicate** : Que signifie \"une arête existe avec 30% de probabilité\" pour un utilisateur final ?",
            "❌ **Garanties probabilistes** : Protection basée sur l'entropie, pas sur des garanties pires-cas",
            "❌ **Échantillonnage requis** : Pour obtenir un graphe standard, il faut échantillonner, perdant de l'information",
            "❌ **Coût de calcul** : O(|E| + k·n) pour l'obfuscation"
        ],
        "karate_example": """
### Exemple sur le Graphe Karate Club

**Avec k=5, ε=0.3** :

**Arêtes existantes** (ex: (0,1)) :
- 🟢 **Probabilité élevée** : p((0,1)) = 1 - ε/k = 1 - 0.3/5 = 0.94 (94%)
- L'arête existe probablement, mais pas avec certitude absolue

**Arêtes potentielles ajoutées** (ex: (0,15) qui n'existe pas) :
- 🟡 **Probabilité faible** : p((0,15)) = ε/(2k) = 0.3/10 = 0.03 (3%)
- Crée du "bruit" pour masquer les vraies connexions

**Pour Mr. Hi (nœud 0)** :
- 📊 **Degré original** : 16 arêtes
- 🎲 **Degré incertain** : ~70 arêtes avec probabilités variées (16 à ~94%, 54 à ~3%)
- 🔒 **Protection** : Un adversaire ne peut plus être sûr des vraies connexions
- 📈 **Entropie** : H(N_5(0)) ≥ log(5) - 0.3 ≈ 2.02 bits d'incertitude

**Graphe résultant** :
- 📊 **78 arêtes originales** avec p ≈ 0.94
- 🔗 **~210 nouvelles arêtes** avec p ≈ 0.03 (30% des 561-78 arêtes possibles)
- 💡 **Utilité** : Analyse statistique possible en tenant compte des probabilités
- ⚠️ **Complexité** : Nécessite des algorithmes adaptés aux graphes incertains
"""
    },

    "MaxVar": {
        "attacks_protected": [
            {
                "name": "Attaques par Seuillage (Threshold Attack)",
                "description": """**PRINCIPALE PROTECTION** : MaxVar résiste aux attaques par seuillage qui réussissent contre (k,ε)-obf.

**Attaque** : L'adversaire applique un seuil (ex: 0.5) pour classifier les arêtes :
- p > 0.5 → arête originale
- p ≤ 0.5 → arête factice

**Pourquoi (k,ε)-obf est vulnérable** : Les probabilités sont concentrées (p ≈ 1.0 ou p ≈ 0.0), donc le seuillage récupère 100% du graphe original.

**Pourquoi MaxVar résiste** : Les probabilités sont dispersées autour de 0.5, donc l'attaquant ne peut pas distinguer arêtes originales et factices. Taux de reconstruction typique : ~85-95% (vs 100% pour (k,ε)-obf)."""
            },
            {
                "name": "Attaques par Degré",
                "description": "Les degrés attendus sont préservés EXACTEMENT : E[deg(u)] = deg_original(u). Un attaquant connaissant le degré ne peut pas isoler un nœud car plusieurs nœuds auront des distributions de degré similaires."
            },
            {
                "name": "Attaques par Voisinage",
                "description": "Les arêtes potentielles \"nearby\" (distance 2) créent de l'ambiguïté sur les vrais voisins. Un attaquant ne peut pas être certain qu'une arête avec p=0.6 existe vraiment ou est factice."
            }
        ],
        "attacks_vulnerable": [
            {
                "name": "Attaques par Échantillonnage Multiple",
                "description": """Si plusieurs graphes échantillons G₁, G₂, ..., Gₙ sont publiés depuis le même graphe incertain G̃, un adversaire peut croiser les informations :
- Estimer les probabilités par fréquence empirique : p̂(u,v) = |{i : (u,v) ∈ Gᵢ}| / n
- Si n est grand, retrouver les probabilités exactes puis deviner les arêtes originales

**Mitigation** : Limiter le nombre de graphes échantillons publiés, ou ajouter du bruit supplémentaire."""
            },
            {
                "name": "Attaques par Analyse de Variance",
                "description": """Un adversaire sophistiqué pourrait analyser la **distribution des probabilités** elle-même :
- Les arêtes originales ont tendance à avoir des probabilités > moyenne
- Les arêtes factices ont des probabilités < moyenne
- Avec analyse statistique, pourrait améliorer le taux de reconstruction au-delà de 90%

**Note** : C'est toujours BEAUCOUP mieux que (k,ε)-obf (100% de reconstruction)."""
            },
            {
                "name": "Pas de Garantie Differential Privacy",
                "description": """MaxVar N'EST PAS ε-différentiellement privé. Il ne protège pas contre un adversaire ayant une connaissance arbitraire du graphe.

**Différence avec DP** :
- ε-DP garantit : P[A(G) = O] ≤ e^ε · P[A(G') = O] pour TOUS graphes voisins G, G'
- MaxVar garantit : variance maximale et résistance au seuillage, mais pas de borne sur le ratio des probabilités

**Implication** : MaxVar est excellent contre les attaques pratiques (seuillage), mais pas contre un adversaire théorique tout-puissant."""
            }
        ],
        "advantages": [
            "✅ **Résistance au seuillage** : Probabilités dispersées empêchent la reconstruction simple",
            "✅ **Conservation EXACTE des degrés attendus** : E[deg(u)] = deg_original(u) ∀u",
            "✅ **Arêtes \"nearby\" plausibles** : Distance 2 (friend-of-friend) minimise la distorsion structurelle",
            "✅ **Maximisation de la variance** : Var[D(G̃, G)] maximale → incertitude maximale",
            "✅ **Fondations mathématiques solides** : Programme quadratique avec solution optimale",
            "✅ **Meilleure utilité que (k,ε)-obf** : À niveau de protection équivalent, préserve mieux la structure",
            "✅ **Échantillonnage efficace** : Un seul graphe échantillon suffit pour l'analyse",
            "✅ **Benchmark supérieur** : Tests montrent 85-95% de reconstruction vs 100% pour (k,ε)-obf"
        ],
        "disadvantages": [
            "❌ **Complexité algorithmique élevée** : O(m²) pour la résolution du programme quadratique",
            "❌ **Passage à l'échelle limité** : Difficile sur graphes >10,000 nœuds sans partitionnement",
            "❌ **Pas de garantie ε-DP** : Ne protège pas contre adversaire avec connaissance arbitraire",
            "❌ **Choix du nombre d'arêtes potentielles** : Paramètre crucial qui affecte privacy/utilité",
            "❌ **Interprétation des probabilités** : Utilisateurs doivent comprendre les graphes incertains",
            "❌ **Vulnérable à l'échantillonnage multiple** : Publication de plusieurs graphes peut révéler les probabilités",
            "❌ **Optimisation quadratique requise** : Nécessite scipy.optimize, pas disponible partout",
            "❌ **Toujours vulnérable à ~10-15%** : Même avec MaxVar, un attaquant intelligent peut retrouver 85-95% des arêtes"
        ],
        "karate_example": """
### Exemple sur le Graphe Karate Club

**Contexte** : 34 nœuds, 78 arêtes originales

**Avec num_potential_edges = 50** :

**Phase 1 - Proposition d'arêtes nearby** :
- 🔍 Pour chaque nœud u, chercher les voisins à distance 2 (friend-of-friend)
- 🟢 Exemple : Nœud 0 est connecté à nœud 1, qui est connecté à nœud 8
  → Ajouter arête potentielle (0,8) si elle n'existe pas déjà
- 📊 **Résultat** : ~50 arêtes potentielles "plausibles" (pas totalement aléatoires)

**Phase 2 - Optimisation quadratique** :
- 🎯 **Objectif** : Minimiser Σp² (équivalent à maximiser Σp(1-p))
- 📐 **Contraintes** : Σp_uv = deg(u) pour chaque nœud u
- 🔧 **Résolution** : SLSQP (Sequential Least Squares Programming)
- ⏱️ **Temps** : ~0.5-2 secondes sur Karate Club

**Phase 3 - Graphe incertain résultant** :

**Arêtes existantes** (78 arêtes) :
- 📊 **Probabilité moyenne** : 0.826 (vs 0.88 pour (k,ε)-obf)
- 📈 **Écart-type** : 0.204 (vs 0.0 pour (k,ε)-obf) → **DISPERSION ÉLEVÉE**
- 🎲 **Plage** : [0.217, 1.000] → certaines arêtes ont p faible!

**Arêtes potentielles** (50 arêtes) :
- 📊 **Probabilité moyenne** : 0.271
- 📈 **Écart-type** : 0.274 → **DISPERSION ÉLEVÉE**
- 🎲 **Plage** : [0.0, 1.0] → certaines arêtes factices ont p élevé!

**Exemple concret - Mr. Hi (nœud 0)** :
- 🔢 **Degré original** : 16
- ✅ **Degré attendu** : E[deg(0)] = 16.00 (conservation EXACTE)
- 🎲 **Arêtes incertaines** : 16 existantes + ~8 potentielles nearby
- 📊 **Probabilités variées** :
  - (0,1) existante : p = 0.63 (dispersé, pas proche de 1.0!)
  - (0,15) potentielle : p = 0.42 (ambiguë!)
  - → Impossible de distinguer par seuillage simple

**Test de reconstruction par seuillage** :
- 🎯 **Threshold = 0.5** : L'attaquant classe p > 0.5 comme arêtes originales
- ❌ **Résultat (k,ε)-obf** : 100.0% des arêtes récupérées → VULNÉRABLE
- ✅ **Résultat MaxVar** : 93.6% des arêtes récupérées → RÉSISTANT

**Trade-off Privacy-Utilité** :
- 🔒 **Privacy** : Résistance au seuillage (6.4% d'erreur vs 0% pour (k,ε)-obf)
- 💡 **Utilité** : Degrés exacts, arêtes nearby plausibles
- ⚖️ **Compromis** : Légèrement plus de calcul (O(m²)) pour meilleure protection
"""
    },

    "EdgeFlip": {
        "attacks_protected": [
            {
                "name": "TOUTES LES ATTAQUES",
                "description": "La Differential Privacy protège contre TOUS les adversaires, quelle que soit leur connaissance auxiliaire (background knowledge) et leur puissance de calcul."
            },
            {
                "name": "Attaques par Composition",
                "description": "Les garanties ε-DP se composent : publier n graphes consomme un budget ε_total = Σε_i, permettant un contrôle précis."
            },
            {
                "name": "Attaques par Inférence Post-traitement",
                "description": "Immunité post-traitement : toute fonction du graphe anonymisé reste ε-DP."
            }
        ],
        "attacks_vulnerable": [
            {
                "name": "Aucune attaque théorique",
                "description": "La DP offre des garanties mathématiques absolues. La seule \"vulnérabilité\" est le choix d'un ε trop grand."
            }
        ],
        "advantages": [
            "✅ **GARANTIES FORMELLES MAXIMALES** : ε-Differential Privacy, l'étalon-or de la privacy",
            "✅ **Protection universelle** : Indépendante de la connaissance de l'adversaire",
            "✅ **Garanties compositionnelles** : ε_total = Σε_i pour publications multiples",
            "✅ **Immunité post-traitement** : Toute analyse du graphe anonymisé reste privée",
            "✅ **Fondations théoriques solides** : Prouvable mathématiquement",
            "✅ **Ajustable** : Le paramètre ε permet de contrôler finement le trade-off"
        ],
        "disadvantages": [
            "❌ **PERTE D'UTILITÉ MAJEURE** : Pour ε raisonnable (ε<1), beaucoup de bruit ajouté",
            "❌ **Complexité O(n²)** : Doit considérer toutes les paires de nœuds, ne passe pas à l'échelle",
            "❌ **Graphe fortement perturbé** : Pour ε=0.8, ~40-60% des arêtes modifiées",
            "❌ **Trade-off privacy/utilité difficile** : Trouver le bon ε est un défi",
            "❌ **Espérance du nombre d'arêtes** : E[|E'|] ≈ n(n-1)/4 ≈ 50% de toutes les arêtes possibles (pour ε modéré)",
            "❌ **Inadapté aux grands graphes** : Facebook (1 milliard de nœuds) impossible à traiter"
        ],
        "karate_example": """
### Exemple sur le Graphe Karate Club

**Avec ε=0.8** :

**Paramètre s** : s = 1 - e^(-ε) = 1 - e^(-0.8) ≈ 0.551 (55.1%)

**Randomized Response sur chaque paire** :
- 🎲 Avec probabilité s/2 ≈ 27.6% : **INVERSER** l'arête (0→1 ou 1→0)
- ✅ Avec probabilité 1-s/2 ≈ 72.4% : **GARDER** l'état réel

**Pour l'arête (0,1) qui EXISTE** :
- 72.4% de chance qu'elle soit présente dans G'
- 27.6% de chance qu'elle soit supprimée

**Pour l'arête (0,15) qui N'EXISTE PAS** :
- 72.4% de chance qu'elle reste absente
- 27.6% de chance qu'elle soit ajoutée

**Résultat sur Karate Club** :
- 📊 **561 paires possibles** (34×33/2)
- 🟢 **78 arêtes originales** : ~57 préservées, ~21 supprimées (27.6%)
- 🔴 **483 non-arêtes** : ~350 restent absentes, ~133 ajoutées (27.6%)
- 📈 **Total** : ~190 arêtes dans G' (au lieu de 78)
- ⚠️ **Utilité** : Le graphe est TRÈS bruité, clustering ≈ random, communautés floues
- 🔒 **Privacy** : AUCUNE arête ne peut être affirmée avec plus de 72.4% de confiance

**Garantie ε=0.8** :
- Pour toute arête (u,v), le ratio des probabilités entre G et G' (diffèrant par cette arête) est ≤ e^0.8 ≈ 2.23
- C'est la définition de ε-DP !
"""
    },

    "Laplace": {
        "attacks_protected": [
            {
                "name": "TOUTES LES ATTAQUES (comme EdgeFlip)",
                "description": "Mécanisme de Laplace offre également ε-Differential Privacy avec les mêmes garanties universelles."
            },
            {
                "name": "Attaques par Requêtes Répétées",
                "description": "Le bruit Laplacien calibré protège même contre des adversaires effectuant de multiples requêtes."
            }
        ],
        "attacks_vulnerable": [
            {
                "name": "Aucune attaque théorique",
                "description": "Idem EdgeFlip : protection théorique totale."
            }
        ],
        "advantages": [
            "✅ **ε-Differential Privacy** : Même niveau de garanties formelles qu'EdgeFlip",
            "✅ **Mécanisme fondamental** : Le plus utilisé en DP, bien étudié et compris",
            "✅ **Généralité** : Applicable à de nombreuses fonctions (pas que les graphes)",
            "✅ **Ajustable** : Contrôle fin via ε",
            "✅ **Compositionnalité** : Budgets ε additionnent naturellement"
        ],
        "disadvantages": [
            "❌ **IDEM EdgeFlip** : Complexité O(n²), perte d'utilité majeure",
            "❌ **Bruit Laplacien peut être excessif** : Pour petites valeurs, le bruit relatif est énorme",
            "❌ **Variance** : Var[Lap(b)] = 2b² = 2/ε², le bruit peut varier fortement",
            "❌ **Seuillage nécessaire** : Décider si une arête existe (seuil à 0.5) est arbitraire",
            "❌ **Peut détruire complètement l'utilité** : Pour ε<0.5, le graphe devient quasi-aléatoire",
            "❌ **Inadapté aux graphes clairsemés** : Sur un graphe avec peu d'arêtes, le bruit domine le signal"
        ],
        "karate_example": """
### Exemple sur le Graphe Karate Club

**Avec ε=1.2** :

**Paramètres** :
- Sensibilité Δf = 1 (une arête change la réponse de 0 à 1 ou vice versa)
- Scale b = 1/ε = 1/1.2 ≈ 0.833

**Processus pour chaque paire (u,v)** :
1. 📊 **Valeur réelle** : x = 1 si arête existe, 0 sinon
2. 🎲 **Bruit** : N ~ Laplace(0, 0.833)
   - 50% du temps : |N| < 0.833 × ln(2) ≈ 0.58
   - Mais peut être très grand (queue lourde) !
3. 🔍 **Valeur bruitée** : x' = x + N
4. ✅ **Décision** : Inclure l'arête si x' > 0.5

**Exemple concret - arête (0,1) EXISTE (x=1)** :
- x' = 1 + N
- Si N = -0.3 → x' = 0.7 > 0.5 : arête PRÉSERVÉE ✅
- Si N = -0.6 → x' = 0.4 < 0.5 : arête SUPPRIMÉE ❌
- P(arête préservée) = P(N > -0.5) ≈ 73%

**Exemple concret - arête (0,15) N'EXISTE PAS (x=0)** :
- x' = 0 + N
- Si N = 0.6 → x' = 0.6 > 0.5 : arête AJOUTÉE 🔴
- Si N = 0.3 → x' = 0.3 < 0.5 : arête RESTE ABSENTE ✅
- P(arête ajoutée) = P(N > 0.5) ≈ 27%

**Résultat sur Karate Club** :
- 📊 **~57 arêtes originales préservées** (73% de 78)
- 🔴 **~130 fausses arêtes ajoutées** (27% de 483)
- 📈 **Total** : ~187 arêtes dans G' (vs 78 originales)
- ⚠️ **Utilité** : Légèrement meilleure qu'EdgeFlip (ε=1.2 > 0.8), mais toujours très bruitée
- 🔒 **Privacy** : Garantie ε=1.2-DP prouvée mathématiquement
"""
    }
}
