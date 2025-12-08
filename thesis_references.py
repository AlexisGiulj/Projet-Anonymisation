"""
Références vers les sections pertinentes de la thèse
"Anonymisation de Graphes Sociaux" par NGUYEN Huu-Hiep (2016)

Format: {clé: {"page": numéro, "section": titre, "description": résumé}}
"""

THESIS_REFERENCES = {
    # Introduction et concepts de base
    "introduction": {
        "page": 13,
        "section": "1. Introduction",
        "description": "Introduction générale à l'anonymisation de graphes sociaux"
    },

    "privacy_models": {
        "page": 25,
        "section": "2. Privacy Models for Graph Data",
        "description": "Modèles de privacy pour les données en graphes"
    },

    # K-anonymity
    "k_anonymity": {
        "page": 30,
        "section": "2.2 k-Anonymity",
        "description": "Définition formelle de la k-anonymity pour les graphes"
    },

    "k_degree_anonymity": {
        "page": 32,
        "section": "2.2.1 k-Degree Anonymity",
        "description": "Algorithmes pour garantir la k-degree anonymity"
    },

    # Generalization
    "generalization": {
        "page": 40,
        "section": "2.3 Generalization",
        "description": "Techniques de généralisation par super-nodes"
    },

    # Differential Privacy
    "differential_privacy": {
        "page": 50,
        "section": "2.4 Differential Privacy",
        "description": "Privacy différentielle appliquée aux graphes"
    },

    "edge_dp": {
        "page": 52,
        "section": "2.4.1 Edge-Level Differential Privacy",
        "description": "DP au niveau des arêtes (EdgeFlip, Laplace)"
    },

    # (k,ε)-obfuscation - SECTION PRINCIPALE
    "k_epsilon_obfuscation": {
        "page": 70,
        "section": "3.3 (k,ε)-obfuscation",
        "description": "Définition et formules de la (k,ε)-obfuscation"
    },

    "k_epsilon_formulas": {
        "page": 72,
        "section": "3.3.2 Probability Assignment",
        "description": "Formules d'assignation des probabilités: p_exist = 1-ε/k, p_potential = ε/(2k)"
    },

    "threshold_attack": {
        "page": 75,
        "section": "3.3.3 Threshold Attack Vulnerability",
        "description": "⚠️ CRITIQUE: Vulnérabilité de (k,ε)-obf à l'attaque par seuillage"
    },

    # MaxVar - SOLUTION au threshold attack
    "maxvar": {
        "page": 80,
        "section": "3.4 MaxVar: Variance Maximizing Scheme",
        "description": "Solution à la vulnérabilité: maximiser la variance des probabilités"
    },

    "maxvar_formulation": {
        "page": 82,
        "section": "3.4.2 Quadratic Programming Formulation",
        "description": "Formulation du programme quadratique: min Σp² sous contraintes"
    },

    "maxvar_algorithm": {
        "page": 85,
        "section": "3.4.3 Implementation Details",
        "description": "Détails d'implémentation: nearby edges, optimisation SLSQP"
    },

    # Uncertain graphs
    "uncertain_graphs": {
        "page": 68,
        "section": "3.2 Uncertain Graphs",
        "description": "Définition des graphes incertains G̃ = (V, E, p)"
    },

    "sampling": {
        "page": 70,
        "section": "3.2.2 Sampling from Uncertain Graphs",
        "description": "Échantillonnage de graphes à partir de graphes incertains"
    },

    # Utility metrics
    "utility_metrics": {
        "page": 95,
        "section": "4. Utility Metrics",
        "description": "Métriques pour mesurer la préservation d'utilité"
    },

    "structural_metrics": {
        "page": 97,
        "section": "4.2 Structural Metrics",
        "description": "Métriques structurelles: degrés, clustering, chemins"
    },

    # Experimental results
    "experiments": {
        "page": 110,
        "section": "5. Experimental Evaluation",
        "description": "Évaluation expérimentale sur différents datasets"
    },

    "karate_club": {
        "page": 115,
        "section": "5.3 Karate Club Results",
        "description": "Résultats sur le graphe Karate Club"
    },

    # Attacks
    "attacks_overview": {
        "page": 45,
        "section": "2.5 Attack Models",
        "description": "Vue d'ensemble des modèles d'attaques"
    },

    "neighborhood_attack": {
        "page": 47,
        "section": "2.5.2 Neighborhood Attack",
        "description": "Attaque basée sur le voisinage des nœuds"
    },
}

def get_thesis_link(ref_key, page=None):
    """
    Génère un lien vers une section de la thèse.

    Args:
        ref_key: clé de référence dans THESIS_REFERENCES
        page: numéro de page spécifique (optionnel, sinon utilise la page par défaut)

    Returns:
        Tuple (page_number, section_title, description)
    """
    if ref_key in THESIS_REFERENCES:
        ref = THESIS_REFERENCES[ref_key]
        page_num = page if page is not None else ref["page"]
        return (page_num, ref["section"], ref["description"])
    return (None, None, None)

def format_thesis_reference(ref_key, custom_text=None):
    """
    Formate une référence à la thèse en Markdown.

    Args:
        ref_key: clé de référence
        custom_text: texte personnalisé (optionnel)

    Returns:
        String Markdown avec icône et lien
    """
    if ref_key not in THESIS_REFERENCES:
        return ""

    ref = THESIS_REFERENCES[ref_key]
    text = custom_text if custom_text else f"Voir thèse p.{ref['page']}"

    return f"📖 **[{text}]** — *{ref['section']}*: {ref['description']}"

def get_method_references(method_name):
    """
    Retourne les références pertinentes pour une méthode donnée.

    Args:
        method_name: nom de la méthode d'anonymisation

    Returns:
        Liste de clés de références pertinentes
    """
    method_refs = {
        "KDegreeAnonymity": ["k_anonymity", "k_degree_anonymity", "attacks_overview"],
        "Generalization": ["generalization", "k_anonymity"],
        "ProbabilisticObfuscation": ["k_epsilon_obfuscation", "k_epsilon_formulas", "threshold_attack", "uncertain_graphs"],
        "MaxVar": ["maxvar", "maxvar_formulation", "maxvar_algorithm", "threshold_attack"],
        "EdgeFlip": ["differential_privacy", "edge_dp", "experiments"],
        "Laplace": ["differential_privacy", "edge_dp", "experiments"],
    }

    return method_refs.get(method_name, ["introduction", "privacy_models"])
