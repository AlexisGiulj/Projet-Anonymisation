"""
Définitions des concepts d'anonymisation et dictionnaires des attaques/propriétés
"""

# ============================================================================
# DÉFINITIONS DES CONCEPTS D'ANONYMISATION
# ============================================================================

ANONYMIZATION_DEFINITIONS = {
    "k-anonymity": {
        "name": "k-Anonymité",
        "definition": """
        Un graphe G satisfait la **k-anonymité sur les degrés** si pour chaque nœud,
        il existe au moins k-1 autres nœuds ayant le même degré.
        """,
        "math_formula": "∀v ∈ V, |{u ∈ V : deg(u) = deg(v)}| ≥ k",
        "intuition": """
        **Intuition** : Imaginez que vous essayez d'identifier quelqu'un dans une foule uniquement
        par sa taille. Si au moins k personnes ont la même taille, vous ne pouvez pas distinguer
        l'individu cible parmi ces k personnes. La k-anonymité applique ce principe aux graphes :
        chaque nœud se "cache" parmi au moins k-1 autres nœuds ayant la même caractéristique (ici le degré).
        """,
        "privacy_guarantee": "P(ré-identification | degré) ≤ 1/k",
        "parameter_meaning": "k = nombre minimum de nœuds indistinguables par leur degré"
    },

    "differential_privacy": {
        "name": "Privacy Différentielle (ε-DP)",
        "definition": """
        Un algorithme A satisfait la **ε-differential privacy** si pour deux graphes G et G'
        différant d'une seule arête, et pour tout résultat possible O :
        """,
        "math_formula": "P[A(G) = O] ≤ e^ε × P[A(G') = O]",
        "intuition": """
        **Intuition** : La privacy différentielle garantit que la présence ou l'absence d'un individu
        (ou d'une relation) dans les données change très peu les résultats statistiques publiés.
        C'est comme ajouter du "bruit" calibré : un adversaire ne peut pas déterminer si vous êtes
        dans la base de données, même avec une connaissance parfaite de tous les autres individus.

        Le paramètre ε contrôle le "budget de privacy" : plus ε est petit, plus la protection est forte,
        mais plus les données sont bruitées (perte d'utilité).
        """,
        "privacy_guarantee": "ε-indistinguishability : protection contre TOUTES les attaques",
        "parameter_meaning": "ε = budget de privacy (typiquement 0.1 à 2.0). Plus ε est petit, plus la privacy est forte"
    },

    "randomization": {
        "name": "Randomisation",
        "definition": """
        La **randomisation** consiste à modifier aléatoirement le graphe en ajoutant et/ou supprimant
        des arêtes, ou en échangeant des paires d'arêtes.
        """,
        "math_formula": """
        Random Add/Del : G' = G ± k arêtes aléatoires
        Random Switch : Échanger k paires d'arêtes (u,w) et (v,x) → (u,v) et (w,x)
        """,
        "intuition": """
        **Intuition** : C'est comme mélanger les cartes d'un jeu. En ajoutant des connexions fictives
        et en supprimant des connexions réelles, on crée de l'incertitude : un adversaire ne peut plus
        être sûr qu'une arête observée existait vraiment dans le graphe original.

        **Random Switch** préserve en plus les degrés des nœuds, ce qui maintient mieux la structure globale.
        """,
        "privacy_guarantee": "Aucune garantie formelle, mais résistance empirique aux attaques par degré",
        "parameter_meaning": "k = nombre d'opérations de modification (typiquement 10-30% des arêtes)"
    },

    "generalization": {
        "name": "Généralisation (Super-nœuds)",
        "definition": """
        La **généralisation** regroupe les nœuds similaires en **super-nœuds** (clusters),
        où chaque super-nœud représente un groupe d'au moins k nœuds.
        """,
        "math_formula": "Partition V = C₁ ∪ C₂ ∪ ... ∪ Cₘ avec |Cᵢ| ≥ k",
        "intuition": """
        **Intuition** : Au lieu de montrer chaque personne individuellement, on montre des groupes.
        C'est comme remplacer "Alice, Bob et Charlie habitent à Paris" par "3 personnes habitent à Paris".
        On perd en précision, mais on gagne en privacy car on ne peut plus identifier les individus.

        Dans un graphe social, on remplace des communautés entières par un seul super-nœud,
        et on compte combien d'arêtes existent entre ces super-nœuds.
        """,
        "privacy_guarantee": "k-anonymité structurelle : impossible d'identifier un nœud spécifique dans un cluster",
        "parameter_meaning": "k = taille minimale des clusters (typiquement 3-5)"
    },

    "probabilistic": {
        "name": "Approche Probabiliste (k,ε)-obfuscation",
        "definition": """
        Une **(k,ε)-obfuscation** publie un **graphe incertain** où chaque arête a une probabilité
        d'existence, garantissant qu'il existe au moins k graphes possibles ayant une probabilité
        suffisamment élevée.
        """,
        "math_formula": "∃ au moins k graphes G₁,...,Gₖ : P(Gᵢ|G^pub) ≥ e^(-ε)",
        "intuition": """
        **Intuition** : Au lieu de publier "Alice connaît Bob : OUI ou NON", on publie
        "Alice connaît Bob : 70% de probabilité". Un adversaire ne peut pas être certain du graphe réel,
        car plusieurs graphes différents sont compatibles avec les probabilités publiées.

        C'est comme un jeu de devinettes où plusieurs réponses sont plausibles : plus il y a de
        graphes candidats plausibles (paramètre k), plus l'attaquant a du mal à deviner le bon.
        """,
        "privacy_guarantee": "Confusion probabiliste : au moins k graphes candidats plausibles",
        "parameter_meaning": "k = nombre de graphes plausibles, ε = marge d'entropie (typiquement 0.1-0.5)"
    }
}


# ============================================================================
# DICTIONNAIRE DES ATTAQUES
# ============================================================================

ATTACKS_DICTIONARY = {
    "degree_attack": {
        "name": "Attaque par Degré",
        "description": """
        L'adversaire connaît le degré (nombre de connexions) de certains nœuds cibles
        et tente de les ré-identifier dans le graphe anonymisé.
        """,
        "example": """
        **Exemple Karate Club** : Si l'adversaire sait que Mr. Hi (nœud 0) a 16 connexions,
        il peut chercher dans le graphe anonymisé le nœud ayant exactement 16 connexions.
        Si un seul nœud a ce degré, la ré-identification est réussie.
        """,
        "protection": "k-degree anonymity garantit qu'au moins k nœuds partagent le même degré",
        "severity": "Moyenne - Facile à implémenter mais limitée"
    },

    "active_attack": {
        "name": "Attaque Active (Sybil)",
        "description": """
        L'adversaire crée de faux comptes (Sybils) avec une structure unique AVANT
        la publication du graphe, puis retrouve ces Sybils dans le graphe anonymisé
        pour identifier d'autres utilisateurs.
        """,
        "example": """
        **Exemple Backstrom et al. (2007)** : En créant seulement 7 nœuds Sybil avec
        une structure en "étoile" unique, un adversaire a pu ré-identifier 2400 relations
        dans un réseau de 4.4 millions de nœuds.

        **Sur Karate Club** : Créer 3 Sybils connectés en triangle uniquement entre eux,
        puis les retrouver après anonymisation (structure unique identifiable).
        """,
        "protection": "Randomisation intensive, Differential Privacy",
        "severity": "Élevée - Très efficace, peut compromettre de nombreux utilisateurs"
    },

    "passive_attack": {
        "name": "Attaque Passive (Interne)",
        "description": """
        Un utilisateur légitime du réseau utilise sa connaissance de son voisinage local
        pour se retrouver dans le graphe anonymisé, puis découvrir d'autres connexions.
        """,
        "example": """
        **Exemple Karate Club** : Si vous êtes le nœud 33 et savez que vous êtes connecté
        aux nœuds 0, 2, 8, 14, 15, vous pouvez chercher dans le graphe anonymisé un nœud
        ayant exactement cette structure de voisinage (degré 5 + degrés des voisins).
        """,
        "protection": "Randomisation, k-degree anonymity, Generalization",
        "severity": "Moyenne - Nécessite connaissance locale mais réalisable"
    },

    "subgraph_attack": {
        "name": "Attaque par Sous-graphe",
        "description": """
        L'adversaire connaît un motif structurel local (sous-graphe) autour du nœud cible
        et le recherche dans le graphe anonymisé.
        """,
        "example": """
        **Exemple Karate Club** : Si l'adversaire sait que le nœud 0 forme un triangle
        avec les nœuds 1 et 2 (tous trois connectés entre eux), il peut chercher ce motif
        triangulaire où un nœud a degré 16, un autre degré 9, et le troisième degré 10.
        """,
        "protection": "Generalization (détruit les motifs locaux), Differential Privacy",
        "severity": "Élevée - Plus puissante que l'attaque par degré seul"
    },

    "neighborhood_attack": {
        "name": "Attaque par Voisinage",
        "description": """
        L'adversaire connaît l'ensemble des voisins (1-hop ou 2-hop) du nœud cible
        et leurs interconnexions.
        """,
        "example": """
        **Exemple Karate Club** : Mr. Hi (nœud 0) a 16 voisins directs. Si l'adversaire
        connaît l'identité de ces 16 voisins et comment ils sont connectés entre eux
        (formant plusieurs triangles), cette "signature" peut être unique.
        """,
        "protection": "Generalization, (k,ε)-obfuscation, Differential Privacy",
        "severity": "Très élevée - Très efficace si connaissance du voisinage disponible"
    },

    "walk_based_attack": {
        "name": "Attaque par Marche Aléatoire",
        "description": """
        L'adversaire utilise les propriétés des marches aléatoires (random walks)
        sur le graphe pour identifier des nœuds par leurs statistiques de visite.
        """,
        "example": """
        **Exemple** : Les nœuds ayant beaucoup de connexions (hubs) sont plus souvent
        visités par les marches aléatoires. Cette propriété peut être exploitée même
        si les degrés sont anonymisés.
        """,
        "protection": "Differential Privacy avec mécanisme sur les marches aléatoires",
        "severity": "Moyenne - Nécessite sophistication mais contourne certaines défenses"
    },

    "auxiliary_info_attack": {
        "name": "Attaque par Information Auxiliaire",
        "description": """
        L'adversaire combine le graphe anonymisé avec des informations externes
        (autres bases de données, réseaux sociaux publics) pour ré-identifier des nœuds.
        """,
        "example": """
        **Exemple réel** : Croiser un graphe de mobilité anonymisé avec des données
        publiques de trajets domicile-travail pour identifier les individus.

        **Sur Karate Club** : Si un graphe académique similaire est publié ailleurs
        avec les noms, on peut faire correspondre les structures.
        """,
        "protection": "Differential Privacy (la seule garantie indépendante de la connaissance)",
        "severity": "Très élevée - Difficile à défendre sans Differential Privacy"
    }
}


# ============================================================================
# DICTIONNAIRE DES PROPRIÉTÉS DE GRAPHES
# ============================================================================

GRAPH_PROPERTIES = {
    "degree": {
        "name": "Degré",
        "definition": "Nombre de connexions directes d'un nœud",
        "formula": "deg(v) = |{u : (v,u) ∈ E}|",
        "utility_importance": "Critique - Identifie les nœuds influents (hubs)",
        "privacy_risk": "Élevé - Peut servir de quasi-identifiant",
        "example": "Dans Karate Club, le nœud 0 (Mr. Hi) a degré 16, le plus élevé"
    },

    "clustering_coefficient": {
        "name": "Coefficient de Clustering",
        "definition": "Proportion de voisins d'un nœud qui sont aussi connectés entre eux (transitivité locale)",
        "formula": "C(v) = (2 × triangles autour de v) / (deg(v) × (deg(v)-1))",
        "utility_importance": "Élevée - Mesure la cohésion locale, importante en analyse sociale",
        "privacy_risk": "Moyen - Révèle la structure locale du voisinage",
        "example": "Un nœud dans une communauté dense aura C ≈ 0.8, dans un hub étoile C ≈ 0"
    },

    "betweenness_centrality": {
        "name": "Centralité d'Intermédiarité",
        "definition": "Fraction des plus courts chemins passant par un nœud",
        "formula": "g(v) = Σ(s,t) [σ(s,t|v) / σ(s,t)]",
        "utility_importance": "Critique - Identifie les 'ponts' et intermédiaires clés",
        "privacy_risk": "Élevé - Information globale difficile à masquer",
        "example": "Dans Karate Club, le nœud 0 et le nœud 33 ont les plus fortes centralités (leaders)"
    },

    "closeness_centrality": {
        "name": "Centralité de Proximité",
        "definition": "Inverse de la distance moyenne vers tous les autres nœuds",
        "formula": "C(v) = (n-1) / Σ(u) d(v,u)",
        "utility_importance": "Moyenne - Mesure l'accessibilité globale",
        "privacy_risk": "Moyen - Nécessite connaissance globale du graphe",
        "example": "Nœuds centraux dans le réseau ont closeness élevée"
    },

    "eigenvector_centrality": {
        "name": "Centralité de Vecteur Propre",
        "definition": "Importance d'un nœud basée sur l'importance de ses voisins (principe de PageRank)",
        "formula": "x(v) = (1/λ) × Σ(u) A(v,u) × x(u)",
        "utility_importance": "Élevée - Identifie les nœuds influents par association",
        "privacy_risk": "Élevé - Révèle la position dans la hiérarchie sociale",
        "example": "Être connecté à Mr. Hi augmente votre eigenvector centrality"
    },

    "density": {
        "name": "Densité",
        "definition": "Proportion d'arêtes existantes par rapport au maximum possible",
        "formula": "d = (2 × m) / (n × (n-1))",
        "utility_importance": "Moyenne - Indique si le réseau est sparse ou dense",
        "privacy_risk": "Faible - Information globale peu sensible",
        "example": "Karate Club : 78 arêtes sur 34×33/2=561 possibles → densité ≈ 0.14"
    },

    "diameter": {
        "name": "Diamètre",
        "definition": "Plus grande distance (plus court chemin) entre deux nœuds du graphe",
        "formula": "diam(G) = max(u,v) d(u,v)",
        "utility_importance": "Moyenne - Mesure l'étendue du réseau",
        "privacy_risk": "Faible - Information globale",
        "example": "Karate Club a un diamètre de 5 (il faut max 5 sauts pour connecter deux membres)"
    },

    "average_path_length": {
        "name": "Longueur Moyenne des Chemins",
        "definition": "Moyenne des plus courts chemins entre toutes les paires de nœuds",
        "formula": "L = (1 / (n×(n-1))) × Σ(u,v) d(u,v)",
        "utility_importance": "Élevée - Mesure l'efficacité de communication",
        "privacy_risk": "Faible - Information globale agrégée",
        "example": "Karate Club : L ≈ 2.4 ('small world' - tout le monde est proche)"
    },

    "degree_distribution": {
        "name": "Distribution des Degrés",
        "definition": "Histogramme du nombre de nœuds ayant chaque degré",
        "formula": "P(k) = |{v : deg(v) = k}| / n",
        "utility_importance": "Critique - Caractérise la topologie du réseau (scale-free, random, etc.)",
        "privacy_risk": "Moyen - Si préservée, permet attaques par degré",
        "example": "Réseaux sociaux suivent souvent une loi de puissance (peu de hubs, beaucoup de nœuds peu connectés)"
    },

    "modularity": {
        "name": "Modularité",
        "definition": "Mesure la qualité de la division en communautés (forte connexion intra-communauté, faible inter-communauté)",
        "formula": "Q = (1/2m) × Σ(i,j) [A(i,j) - k(i)k(j)/2m] × δ(c(i),c(j))",
        "utility_importance": "Élevée - Essentielle pour détection de communautés",
        "privacy_risk": "Moyen - Révèle les groupes sociaux",
        "example": "Karate Club a haute modularité (~0.4) car il y a 2 communautés bien séparées"
    },

    "triangles": {
        "name": "Nombre de Triangles",
        "definition": "Nombre de triplets de nœuds tous connectés entre eux",
        "formula": "T = |{(u,v,w) : (u,v), (v,w), (u,w) ∈ E}| / 3",
        "utility_importance": "Élevée - Mesure la transitivité globale",
        "privacy_risk": "Moyen - Les triangles révèlent les groupes d'amis",
        "example": "Karate Club contient 45 triangles (motifs de 3 amis mutuels)"
    },

    "assortativity": {
        "name": "Assortativité",
        "definition": "Tendance des nœuds similaires à se connecter (homophilie)",
        "formula": "r = correlation(deg(u), deg(v)) pour toutes les arêtes (u,v)",
        "utility_importance": "Moyenne - Révèle les patterns de connexion",
        "privacy_risk": "Faible - Information globale agrégée",
        "example": "r > 0 : les hubs se connectent entre eux ; r < 0 : les hubs se connectent aux nœuds peu connectés"
    },

    "edit_distance": {
        "name": "Distance d'Édition",
        "definition": "Nombre minimum d'opérations (ajout ou suppression d'arêtes) pour transformer le graphe original en graphe anonymisé",
        "formula": "d(G, G') = |E△E'| = |E\\E'| + |E'\\E|",
        "utility_importance": "Critique - Mesure directe de la modification du graphe",
        "privacy_risk": "N/A - Métrique d'utilité (plus la distance est faible, meilleure est l'utilité)",
        "example": "Si 10 arêtes sont ajoutées et 5 supprimées, la distance d'édition est 15"
    }
}


# ============================================================================
# EXEMPLES D'ATTAQUES CONCRÈTES SUR KARATE CLUB
# ============================================================================

CONCRETE_ATTACK_EXAMPLES = {
    "degree_attack_example": {
        "title": "Attaque par Degré sur Karate Club",
        "scenario": """
        **Scénario** : L'adversaire sait que Mr. Hi (instructeur) a 16 connexions dans le club.
        """,
        "steps": [
            "1. Observer le graphe anonymisé",
            "2. Chercher le nœud ayant degré = 16",
            "3. Si un seul nœud a ce degré → Ré-identification réussie !",
            "4. Découvrir tous les amis de Mr. Hi"
        ],
        "success_rate_no_protection": "100% (Mr. Hi a un degré unique)",
        "success_rate_k_anonymity": "≤ 50% avec k=2 (au moins 2 nœuds de degré 16)",
        "success_rate_randomization": "~40% (le degré est bruité)",
        "code_simulation": """
# Simuler l'attaque
G = nx.karate_club_graph()
target_degree = G.degree(0)  # Mr. Hi
candidates = [n for n in G.nodes() if G.degree(n) == target_degree]
print(f"Candidats pour Mr. Hi: {candidates}")
# Sans protection: [0] → 100% de succès
# Avec k-anonymity: [0, 23] → 50% de succès
        """
    },

    "active_attack_example": {
        "title": "Attaque Active (Sybil) sur Karate Club",
        "scenario": """
        **Scénario** : L'adversaire s'inscrit au club avec 2 complices (Sybils),
        crée une structure unique, puis ré-identifie cette structure après publication.
        """,
        "steps": [
            "1. Créer 3 nœuds Sybil S1, S2, S3",
            "2. Connecter: S1-S2, S2-S3, S3-S1 (triangle)",
            "3. S1 se connecte uniquement à Mr. Hi (nœud 0)",
            "4. Attendre la publication du graphe anonymisé",
            "5. Chercher un triangle où un sommet n'a qu'un seul voisin externe",
            "6. Ce voisin externe est Mr. Hi → Ré-identification !",
            "7. Découvrir tous les amis de Mr. Hi"
        ],
        "success_rate_no_protection": "~95% (structure très distinctive)",
        "success_rate_randomization": "~60% (peut casser le triangle)",
        "success_rate_differential_privacy": "<10% (structure noyée dans le bruit)",
        "defense": "Seule la Differential Privacy est vraiment efficace contre les attaques actives"
    },

    "subgraph_attack_example": {
        "title": "Attaque par Sous-graphe sur Karate Club",
        "scenario": """
        **Scénario** : L'adversaire sait que Mr. Hi, Bob et Carol forment un triangle
        (tous trois amis mutuels), et que Mr. Hi a beaucoup plus d'amis que les deux autres.
        """,
        "steps": [
            "1. Chercher tous les triangles dans le graphe anonymisé",
            "2. Pour chaque triangle, calculer les degrés des 3 sommets",
            "3. Chercher un triangle avec pattern: [degré élevé, degré moyen, degré moyen]",
            "4. Le nœud à degré élevé est probablement Mr. Hi"
        ],
        "success_rate_no_protection": "~85%",
        "success_rate_generalization": "~20% (les triangles locaux sont détruits)",
        "protection_ranking": [
            "Generalization: ★★★★★ (détruit les motifs locaux)",
            "Differential Privacy: ★★★★☆ (ajoute triangles fictifs)",
            "Randomization: ★★☆☆☆ (peut préserver les triangles)"
        ]
    }
}
