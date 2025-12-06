"""
Contenu des nouveaux onglets à intégrer dans l'application
Ce fichier contient le code pour les onglets 3, 4, 5, 6, 7, 8
"""

# TAB 3 - Métriques d'Utilité
tab3_content = '''
        with tab3:
            st.markdown("## 📈 Métriques d'Utilité du Graphe")

            st.markdown("""
            Ces métriques mesurent la **préservation de l'utilité** du graphe après anonymisation.
            Plus ces métriques sont proches du graphe original, mieux l'utilité est préservée.
            """)

            utility_metrics = calculate_utility_metrics(G_orig, G_anon)

            if utility_metrics.get('comparable', True):
                st.markdown("### 📊 Métriques de Base")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Nœuds", utility_metrics.get('num_nodes', 'N/A'))
                with col2:
                    st.metric("Arêtes", utility_metrics.get('num_edges', 'N/A'))
                with col3:
                    orig_density = nx.density(G_orig)
                    anon_density = utility_metrics.get('density', 0)
                    delta_density = anon_density - orig_density
                    st.metric("Densité", f"{anon_density:.3f}", delta=f"{delta_density:+.3f}")
                with col4:
                    if utility_metrics.get('avg_clustering') is not None:
                        orig_clust = nx.average_clustering(G_orig)
                        anon_clust = utility_metrics['avg_clustering']
                        delta_clust = anon_clust - orig_clust
                        st.metric("Clustering Moyen", f"{anon_clust:.3f}", delta=f"{delta_clust:+.3f}")

                st.markdown("---")
                st.markdown("### 🌐 Métriques Globales")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if utility_metrics.get('diameter') is not None:
                        try:
                            if nx.is_connected(G_orig):
                                orig_diam = nx.diameter(G_orig)
                            else:
                                largest_cc = max(nx.connected_components(G_orig), key=len)
                                orig_diam = nx.diameter(G_orig.subgraph(largest_cc))
                            delta_diam = utility_metrics['diameter'] - orig_diam
                            st.metric("Diamètre", utility_metrics['diameter'], delta=f"{delta_diam:+d}")
                        except:
                            st.metric("Diamètre", utility_metrics['diameter'])

                with col2:
                    if utility_metrics.get('avg_shortest_path') is not None:
                        try:
                            if nx.is_connected(G_orig):
                                orig_asp = nx.average_shortest_path_length(G_orig)
                            else:
                                largest_cc = max(nx.connected_components(G_orig), key=len)
                                orig_asp = nx.average_shortest_path_length(G_orig.subgraph(largest_cc))
                            delta_asp = utility_metrics['avg_shortest_path'] - orig_asp
                            st.metric("Chemin Moyen", f"{utility_metrics['avg_shortest_path']:.2f}", delta=f"{delta_asp:+.2f}")
                        except:
                            st.metric("Chemin Moyen", f"{utility_metrics['avg_shortest_path']:.2f}")

                with col3:
                    if utility_metrics.get('degree_correlation') is not None:
                        st.metric("Corrélation des Degrés", f"{utility_metrics['degree_correlation']:.3f}",
                                 help="Coefficient de Spearman : 1 = parfait, 0 = aucune corrélation")

                st.markdown("---")
                st.markdown("### 📉 Trade-off Utilité vs Modifications")

                metrics = calculate_anonymization_metrics(G_orig, G_anon)

                if metrics:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Modifications des Arêtes**")
                        added = metrics.get('edges_added', 0)
                        removed = metrics.get('edges_removed', 0)
                        preserved = metrics.get('edges_preserved', 0)

                        import pandas as pd
                        df_edges = pd.DataFrame({
                            'Type': ['Préservées', 'Ajoutées', 'Supprimées'],
                            'Nombre': [preserved, added, removed]
                        })
                        st.bar_chart(df_edges.set_index('Type'))

                    with col2:
                        st.markdown("**Taux de Modification**")
                        rate = metrics.get('modification_rate', 0)
                        st.progress(min(rate, 1.0))
                        st.metric("Taux de modification", f"{rate*100:.1f}%")

                        if rate < 0.1:
                            st.success("✅ Utilité très bien préservée")
                        elif rate < 0.3:
                            st.info("ℹ️ Utilité correctement préservée")
                        else:
                            st.warning("⚠️ Modifications importantes")

            else:
                st.info("Graphe de type super-nodes : métriques d'utilité non directement comparables")
'''

# TAB 4 - Métriques Privacy
tab4_content = '''
        with tab4:
            st.markdown("## 🔒 Métriques de Privacy")

            st.markdown("""
            Ces métriques quantifient la **protection de la vie privée** offerte par l'anonymisation.
            Plus ces valeurs sont élevées, meilleure est la protection.
            """)

            method_params = st.session_state.get('method_params', {})
            privacy_metrics = calculate_privacy_metrics_separated(G_orig, G_anon, st.session_state.method_key, method_params)

            if privacy_metrics:
                st.markdown("### 🛡️ Garanties de Privacy")

                if 'k_value' in privacy_metrics:
                    # k-anonymity
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("k requis", privacy_metrics['k_value'])
                    with col2:
                        st.metric("Ensemble d'anonymat min.", privacy_metrics['min_anonymity_set'])
                    with col3:
                        satisfies = privacy_metrics['satisfies_k_anonymity']
                        if satisfies:
                            st.success(f"✅ {privacy_metrics['k_value']}-anonymité satisfaite")
                        else:
                            st.error(f"❌ {privacy_metrics['k_value']}-anonymité NON satisfaite")

                    st.markdown("---")
                    prob = privacy_metrics['re_identification_probability']
                    st.markdown(f"**Probabilité de ré-identification** : {prob:.3f} ({prob*100:.1f}%)")

                    st.progress(1 - prob)

                    if prob < 0.2:
                        st.success("✅ Risque de ré-identification faible")
                    elif prob < 0.5:
                        st.warning("⚠️ Risque de ré-identification modéré")
                    else:
                        st.error("❌ Risque de ré-identification élevé")

                elif 'epsilon_budget' in privacy_metrics:
                    # Differential Privacy
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        eps = privacy_metrics['epsilon_budget']
                        st.metric("ε (epsilon) Budget", f"{eps:.2f}")

                    with col2:
                        loss = privacy_metrics['privacy_loss_bound']
                        st.metric("Borne de perte de privacy", f"e^{eps:.2f} = {loss:.2f}x")

                    with col3:
                        level = privacy_metrics['privacy_level']
                        if "Forte" in level:
                            st.success(f"✅ {level}")
                        elif "Moyenne" in level:
                            st.warning(f"⚠️ {level}")
                        else:
                            st.error(f"❌ {level}")

                    st.markdown("---")

                    if 'flip_probability' in privacy_metrics:
                        st.markdown("### 🎲 EdgeFlip - Paramètres de Randomisation")
                        col1, col2 = st.columns(2)

                        with col1:
                            flip_prob = privacy_metrics['flip_probability']
                            st.metric("Probabilité de flip", f"{flip_prob:.3f}")

                        with col2:
                            expected_noise = privacy_metrics['expected_noise_edges']
                            st.metric("Arêtes bruitées (attendu)", expected_noise)

                elif 'k_candidates' in privacy_metrics:
                    # Probabilistic
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("k graphes candidats", privacy_metrics['k_candidates'])

                    with col2:
                        st.metric("ε tolérance", f"{privacy_metrics['epsilon_tolerance']:.2f}")

                    with col3:
                        entropy = privacy_metrics['min_entropy']
                        st.metric("Entropie minimale", f"{entropy:.2f}")

                    st.markdown("---")
                    confusion = privacy_metrics['confusion_factor']
                    st.info(f"**Facteur de confusion** : {confusion} graphes plausibles")

                elif 'min_cluster_size' in privacy_metrics:
                    # Generalization
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Taille min. cluster", int(privacy_metrics['min_cluster_size']))

                    with col2:
                        st.metric("Taille moy. cluster", f"{privacy_metrics['avg_cluster_size']:.1f}")

                    with col3:
                        max_priv = privacy_metrics['max_privacy']
                        st.metric("Prob. max ré-identification", f"{max_priv:.3f}")

                st.markdown("---")

                # Garanties globales
                guarantees = calculate_privacy_guarantees(G_orig, G_anon, st.session_state.method_key, method_params)

                if guarantees:
                    st.markdown("### 📋 Garanties Détaillées")

                    with st.expander("Voir toutes les garanties"):
                        for key, value in guarantees.items():
                            st.text(f"{key}: {value}")

            else:
                st.info("Aucune métrique de privacy spécifique pour cette méthode")
'''

# TAB 5 - Simulations d'Attaques
tab5_content = '''
        with tab5:
            st.markdown("## 🎯 Simulations d'Attaques Réelles")

            st.markdown("""
            Cette section simule des attaques de **ré-identification** sur le graphe anonymisé.
            Ces simulations montrent concrètement si un adversaire peut retrouver des nœuds spécifiques.
            """)

            st.markdown("---")

            # Sélection du nœud cible
            st.markdown("### 🎯 Configuration de l'Attaque")

            col1, col2 = st.columns(2)

            with col1:
                target_node = st.number_input(
                    "Nœud cible à retrouver",
                    min_value=0,
                    max_value=G_orig.number_of_nodes()-1,
                    value=0,
                    help="Le nœud que l'adversaire essaie de ré-identifier"
                )

            with col2:
                attack_type = st.selectbox(
                    "Type d'attaque",
                    ["Degree Attack", "Subgraph Attack (Triangles)"]
                )

            st.markdown("---")

            if st.button("🚀 Lancer l'Attaque"):
                st.markdown("### 📊 Résultats de l'Attaque")

                with st.spinner("Simulation en cours..."):
                    if attack_type == "Degree Attack":
                        results = simulate_degree_attack(G_orig, G_anon, target_node)
                    else:
                        results = simulate_subgraph_attack(G_orig, G_anon, target_node)

                if results['success']:
                    st.error("### ⚠️ Attaque Réussie !")
                    st.markdown(results['explanation'])

                    st.markdown(f"**Nœud ré-identifié** : {results.get('re_identified_node', 'N/A')}")

                else:
                    st.success("### ✅ Attaque Échouée / Partiellement Réussie")
                    st.markdown(results['explanation'])

                st.markdown("---")
                st.markdown("### 📈 Détails Techniques")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Nœud cible** :")
                    st.info(f"Nœud {target_node}")

                    if 'target_degree' in results:
                        st.markdown("**Degré du nœud** :")
                        st.info(f"Degré = {results['target_degree']}")

                    if 'target_triangles' in results:
                        st.markdown("**Triangles** :")
                        st.info(f"{results['target_triangles']} triangles")

                with col2:
                    st.markdown("**Candidats trouvés** :")
                    if results['candidates']:
                        st.info(f"{len(results['candidates'])} nœuds : {results['candidates'][:10]}")
                    else:
                        st.info("Aucun candidat")

                    if len(results['candidates']) > 1:
                        prob_success = 1 / len(results['candidates'])
                        st.markdown("**Probabilité de succès** :")
                        st.warning(f"{prob_success*100:.1f}%")

            st.markdown("---")

            # Section éducative
            with st.expander("📚 En savoir plus sur ces attaques"):
                st.markdown("""
                ### Degree Attack (Attaque par Degré)

                L'adversaire connaît le degré (nombre de connexions) du nœud cible et cherche
                dans le graphe anonymisé tous les nœuds ayant ce degré.

                **Protection** :
                - k-degree anonymity garantit au moins k nœuds par degré
                - Randomisation modifie les degrés
                - Differential Privacy ajoute du bruit

                ### Subgraph Attack (Attaque par Sous-graphe)

                L'adversaire connaît la structure locale autour du nœud (ex: triangles, motifs).
                Cette attaque est plus puissante car elle exploite plus d'information.

                **Protection** :
                - Generalization détruit les motifs locaux
                - Differential Privacy ajoute/supprime des triangles fictifs
                - Randomisation casse certains motifs
                """)
'''

# TAB 6 - Attaques & Garanties (existant)
tab6_content = "# Rien à changer, c'est déjà le bon contenu existant (tab4 devient tab6)"

# TAB 7 - Dictionnaire des Attaques
tab7_content = '''
        with tab7:
            st.markdown("## 📚 Dictionnaire des Attaques de Ré-Identification")

            st.markdown("""
            Ce dictionnaire présente **toutes les attaques connues** contre les graphes anonymisés,
            avec des exemples concrets et des explications détaillées.
            """)

            st.markdown("---")

            # Liste des attaques
            attack_names = [ATTACKS_DICTIONARY[k]['name'] for k in ATTACKS_DICTIONARY.keys()]

            selected_attack_name = st.selectbox(
                "Choisir une attaque à explorer",
                attack_names
            )

            # Trouver l'attaque correspondante
            selected_attack_key = list(ATTACKS_DICTIONARY.keys())[attack_names.index(selected_attack_name)]
            attack = ATTACKS_DICTIONARY[selected_attack_key]

            st.markdown(f"### {attack['name']}")

            col1, col2 = st.columns([2, 1])

            with col1:
                with st.expander("📝 Description de l'Attaque", expanded=True):
                    st.markdown(attack['description'])

                with st.expander("💡 Exemple Concret"):
                    st.markdown(attack['example'])

            with col2:
                st.markdown("**⚠️ Sévérité**")
                severity = attack['severity']
                if "Très élevée" in severity or "Élevée" in severity:
                    st.error(severity)
                elif "Moyenne" in severity:
                    st.warning(severity)
                else:
                    st.info(severity)

                st.markdown("**🛡️ Protection**")
                st.success(attack['protection'])

            st.markdown("---")

            # Exemples concrets sur Karate Club
            st.markdown("### 🥋 Exemples Concrets sur Karate Club")

            example_keys = list(CONCRETE_ATTACK_EXAMPLES.keys())

            for example_key in example_keys:
                example = CONCRETE_ATTACK_EXAMPLES[example_key]

                with st.expander(f"📖 {example['title']}"):
                    st.markdown(f"**Scénario** : {example['scenario']}")

                    st.markdown("**Étapes de l'attaque** :")
                    for step in example['steps']:
                        st.markdown(f"- {step}")

                    st.markdown("---")
                    st.markdown("**Taux de Succès** :")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Sans protection", example.get('success_rate_no_protection', 'N/A'))

                    with col2:
                        if 'success_rate_k_anonymity' in example:
                            st.metric("Avec k-anonymity", example['success_rate_k_anonymity'])
                        elif 'success_rate_randomization' in example:
                            st.metric("Avec randomization", example['success_rate_randomization'])

                    with col3:
                        if 'success_rate_differential_privacy' in example:
                            st.metric("Avec Diff. Privacy", example['success_rate_differential_privacy'])
                        elif 'success_rate_generalization' in example:
                            st.metric("Avec Generalization", example['success_rate_generalization'])

                    if 'code_simulation' in example:
                        with st.expander("💻 Code de Simulation"):
                            st.code(example['code_simulation'], language='python')
'''

# TAB 8 - Dictionnaire des Propriétés
tab8_content = '''
        with tab8:
            st.markdown("## 🔍 Dictionnaire des Propriétés de Graphes")

            st.markdown("""
            Ce dictionnaire explique **toutes les propriétés de graphes** utilisées en anonymisation,
            leur importance pour l'utilité, et leur risque pour la privacy.
            """)

            st.markdown("---")

            # Liste des propriétés
            property_names = [GRAPH_PROPERTIES[k]['name'] for k in GRAPH_PROPERTIES.keys()]

            selected_property_name = st.selectbox(
                "Choisir une propriété à explorer",
                property_names
            )

            # Trouver la propriété correspondante
            selected_property_key = list(GRAPH_PROPERTIES.keys())[property_names.index(selected_property_name)]
            prop = GRAPH_PROPERTIES[selected_property_key]

            st.markdown(f"### {prop['name']}")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("📝 Définition", expanded=True):
                    st.markdown(prop['definition'])

                with st.expander("🔢 Formule"):
                    st.code(prop['formula'], language='text')

                with st.expander("💡 Exemple"):
                    st.info(prop['example'])

            with col2:
                st.markdown("**📊 Importance pour l'Utilité**")
                importance = prop['utility_importance']
                if "Critique" in importance or "Élevée" in importance:
                    st.success(importance)
                else:
                    st.info(importance)

                st.markdown("**⚠️ Risque pour la Privacy**")
                risk = prop['privacy_risk']
                if "Élevé" in risk:
                    st.error(risk)
                elif "Moyen" in risk:
                    st.warning(risk)
                else:
                    st.success(risk)

            st.markdown("---")

            # Calcul des propriétés sur le graphe actuel
            if isinstance(G_anon, nx.Graph):
                st.markdown("### 📊 Valeurs pour le Graphe Actuel")

                try:
                    if selected_property_key == 'degree':
                        degrees = dict(G_anon.degree())
                        st.metric("Degré moyen", f"{np.mean(list(degrees.values())):.2f}")
                        st.metric("Degré max", max(degrees.values()))

                    elif selected_property_key == 'clustering_coefficient':
                        clustering = nx.average_clustering(G_anon)
                        st.metric("Coefficient de clustering moyen", f"{clustering:.3f}")

                    elif selected_property_key == 'density':
                        density = nx.density(G_anon)
                        st.metric("Densité", f"{density:.3f}")

                    elif selected_property_key == 'diameter':
                        if nx.is_connected(G_anon):
                            diameter = nx.diameter(G_anon)
                            st.metric("Diamètre", diameter)
                        else:
                            st.info("Graphe non connexe, diamètre non défini")

                    elif selected_property_key == 'average_path_length':
                        if nx.is_connected(G_anon):
                            apl = nx.average_shortest_path_length(G_anon)
                            st.metric("Longueur moyenne des chemins", f"{apl:.2f}")
                        else:
                            st.info("Graphe non connexe, calculé sur la plus grande composante")

                except Exception as e:
                    st.warning(f"Calcul non disponible pour ce graphe")
'''

print("Fichier de contenu des onglets créé avec succès!")
print("\nLes contenus suivants sont prêts à être intégrés :")
print("- TAB 3: Métriques d'Utilité")
print("- TAB 4: Métriques Privacy")
print("- TAB 5: Simulations d'Attaques")
print("- TAB 7: Dictionnaire des Attaques")
print("- TAB 8: Dictionnaire des Propriétés")
