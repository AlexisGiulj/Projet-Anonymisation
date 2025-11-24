# 🔒 Application Interactive d'Anonymisation de Graphes

## 📦 Fichiers Créés

### Applications

1. **graph_anonymization_app.py** - Application Streamlit complète (RECOMMANDÉ)
   - Interface web moderne
   - Explications mathématiques détaillées
   - Métriques d'anonymisation interactives

2. **graph_anonymization_demo.py** - Version batch (génère des PNG)
   - Génère 3 visualisations PNG
   - Exécution rapide sans interaction

### Fichiers de Support

- **requirements.txt** - Dépendances Python
- **README.md** - Documentation complète du projet
- **GUIDE_EXPOSE.md** - Guide détaillé pour votre présentation
- **LANCER_APP.bat** - Script de lancement Windows

---

## 🚀 Lancement de l'Application

### Option 1 : Application Streamlit (RECOMMANDÉ)

**Si Streamlit n'est pas installé** :
```bash
pip install streamlit
```

**Si l'installation de Streamlit échoue** (problème avec pyarrow) :
```bash
# Sur Windows, installer la version pré-compilée
pip install --only-binary :all: pyarrow
pip install streamlit
```

**Lancer l'application** :
```bash
streamlit run graph_anonymization_app.py
```

L'application s'ouvrira dans votre navigateur à l'adresse `http://localhost:8501`

**Ou utiliser le script batch** :
```bash
LANCER_APP.bat
```

---

### Option 2 : Version Batch (génération d'images)

```bash
python graph_anonymization_demo.py
```

Génère 3 fichiers PNG :
- `graph_anonymization_comparison.png` (3.3 MB)
- `degree_distributions.png` (574 KB)
- `metrics_comparison.png` (443 KB)

---

## 🎯 Fonctionnalités de l'Application Interactive

### 📊 Sélection de Graphe
- Karate Club (34 nœuds)
- Graphe aléatoire petit (20 nœuds)
- Graphe aléatoire moyen (50 nœuds)

### 🔬 Méthodes d'Anonymisation

#### 1. Randomisation
- **Random Add/Del** : Ajoute/supprime k=20 arêtes aléatoirement
- **Random Switch** : Échange k=25 paires d'arêtes (préserve les degrés)

#### 2. K-Anonymisation
- **k-degree anonymity** : Garantit que chaque degré apparaît ≥ k=2 fois

#### 3. Généralisation
- **Super-nodes** : Regroupe les nœuds en clusters de taille ≥ k=4

#### 4. Probabiliste
- **(k,ε)-obfuscation** : Crée un graphe incertain (k=5, ε=0.3)

#### 5. Privacy Différentielle
- **EdgeFlip** : Randomized Response Technique (ε=0.8)
- **Laplace** : Mécanisme de Laplace (ε=1.2)

---

## 📖 Explications Détaillées

Pour chaque méthode, l'application fournit :

### 🔢 Explications Mathématiques

**Formulation complète** :
- Définitions formelles
- Algorithmes détaillés
- Propriétés mathématiques
- Complexité temporelle

**Exemple (k-degree anonymity)** :
```
∀d ∈ {deg(v) : v ∈ V}, |{v ∈ V : deg(v) = d}| ≥ k
```

### 💡 Explications en Langage Naturel

**Intuitions** :
- Analogies concrètes
- Exemples du quotidien
- Scénarios d'attaque

**Exemple (EdgeFlip)** :
```
C'est comme le "Randomized Response" en statistiques :
- Lancez une pièce en secret
- Si Face : répondez la vérité
- Si Pile : répondez au hasard
→ Votre réponse a du "déni plausible"
```

### 📊 Niveau de Privacy et d'Utilité

Pour chaque méthode :
- **Niveau de Privacy** : Faible / Moyenne / Forte / Très Forte
- **Préservation de l'Utilité** : Faible / Moyenne / Bonne / Très Bonne
- **Garanties formelles** : Aucune / k-anonymity / ε-DP

---

## 📈 Métriques d'Anonymisation

### Métriques Disponibles

#### 1. Modification des Arêtes
- **Arêtes Ajoutées** : Nouvelles arêtes créées
- **Arêtes Supprimées** : Arêtes originales retirées
- **Arêtes Préservées** : Arêtes maintenues
- **Taux de Modification** : (Ajoutées + Supprimées) / (2 × Originales)

#### 2. Modification des Degrés
- **Changement Total de Degrés** : Σ|deg_orig(v) - deg_anon(v)|
- **Nœuds avec Degré Modifié** : Nombre de nœuds dont le degré a changé
- **Taux de Préservation** : % de nœuds avec degré inchangé

#### 3. Propriétés Structurelles
- **Changement de Clustering** : Δ coefficient de clustering moyen
- **Changement de Densité** : Δ densité du graphe

### Explication du Taux d'Anonymisation

L'application fournit un bouton dédié expliquant :

- **Définition** du taux d'anonymisation
- **Interprétation** des différentes mesures
- **Trade-off** Privacy vs Utilité
- **Comparaison** entre les méthodes

**4 indicateurs clés** :
1. Taux de Modification des Arêtes (0-100%)
2. Incorrectness (0-100%)
3. Entropie de Shannon (0 à log₂(n))
4. Budget ε en Differential Privacy (0.1 à 10+)

---

## 🎨 Visualisations

### Graphes Côte-à-Côte
- **Original** (bleu clair)
- **Anonymisé** (vert clair)
  - Arêtes bleues continues = préservées
  - Arêtes rouges pointillées = ajoutées

### Distributions de Degrés
- Histogrammes comparatifs
- Permet de voir l'impact sur les degrés

---

## 💻 Structure du Code

```python
class GraphAnonymizer:
    """Classe principale d'anonymisation"""

    def __init__(self, graph):
        """Initialise avec le graphe original"""

    def random_add_del(self, k=20):
        """Randomisation - Random Add/Del"""

    def random_switch(self, k=25):
        """Randomisation - Random Switch"""

    def k_degree_anonymity(self, k=2):
        """K-anonymisation"""

    def generalization(self, k=4):
        """Généralisation en super-nodes"""

    def probabilistic_obfuscation(self, k=5, epsilon=0.3):
        """Approches Probabilistes"""

    def differential_privacy_edgeflip(self, epsilon=0.8):
        """Privacy Différentielle - EdgeFlip"""

    def differential_privacy_laplace(self, epsilon=1.2):
        """Privacy Différentielle - Laplace"""
```

### Paramètres Optimisés

Les paramètres ont été **ajustés pour équilibrer** l'effet visible de chaque méthode :

| Méthode | Paramètres | Effet Attendu |
|---------|-----------|---------------|
| Random Add/Del | k=20 | ~20-30% modification |
| Random Switch | k=25 | Visible mais degrés préservés |
| k-degree | k=2 | ~10-20% ajout d'arêtes |
| Généralisation | k=4 | ~4-6 super-nœuds |
| Probabiliste | k=5, ε=0.3 | ~50%+ ajout (prob. faibles) |
| EdgeFlip | ε=0.8 | ~40-60% modification |
| Laplace | ε=1.2 | ~30-50% modification |

**Objectif** : Chaque méthode produit un effet visible et comparable.

---

## 🔧 Dépannage

### Problème : Streamlit ne s'installe pas

**Cause** : Problème de compilation de PyArrow sur Windows

**Solution 1** : Installer la version binaire pré-compilée
```bash
pip install --only-binary :all: pyarrow
pip install streamlit
```

**Solution 2** : Utiliser la version batch
```bash
python graph_anonymization_demo.py
```

---

### Problème : Matplotlib ne s'affiche pas

**Cause** : Backend matplotlib non configuré

**Solution** :
```python
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Qt5Agg'
import matplotlib.pyplot as plt
```

---

### Problème : L'application Streamlit ne se charge pas

**Vérifications** :
1. Port 8501 disponible ?
   ```bash
   netstat -an | findstr 8501
   ```

2. Streamlit correctement installé ?
   ```bash
   streamlit --version
   ```

3. Navigateur bloque localhost ?
   - Désactiver temporairement le pare-feu/antivirus

---

## 📚 Utilisation pour l'Exposé

### Préparation

1. **Tester l'application** avant l'exposé
   ```bash
   streamlit run graph_anonymization_app.py
   ```

2. **Préparer des exemples** :
   - Tester chaque méthode
   - Capturer des screenshots
   - Noter les valeurs des métriques

3. **Plan B** : Avoir les PNG de la version batch
   ```bash
   python graph_anonymization_demo.py
   ```

### Pendant l'Exposé

**Option 1 : Démonstration Live** (IMPRESSIONNANT)
- Lancer l'app Streamlit
- Montrer la sélection interactive
- Afficher les explications mathématiques en direct
- Montrer les métriques

**Option 2 : Utiliser les Screenshots** (SAFE)
- Préparer des captures d'écran à l'avance
- Intégrer dans vos slides

**Option 3 : Vidéo Enregistrée** (HYBRIDE)
- Enregistrer une session de démonstration
- Montrer la vidéo pendant l'exposé

---

## 🎓 Points Clés à Retenir

### Avantages de l'Application

✅ **Pédagogique** :
- Explications mathématiques ET naturelles
- Visualisations comparatives
- Métriques quantitatives

✅ **Équilibrée** :
- Paramètres ajustés pour effets visibles
- Comparaison équitable entre méthodes

✅ **Complète** :
- 5 types de méthodes (7 variantes)
- 3 graphes de test disponibles
- Explications du taux d'anonymisation

### Limitations

⚠️ **Scalabilité** :
- Graphes limités à ~100 nœuds
- EdgeFlip et Laplace sont O(n²)

⚠️ **Simplicité** :
- Implémentations pédagogiques (pas production)
- Heuristiques pour k-anonymity

⚠️ **Installation** :
- Streamlit peut être difficile à installer sur certains systèmes
- PyArrow nécessite compilation

---

## 🔗 Liens Utiles

### Documentation
- **NetworkX** : https://networkx.org/
- **Streamlit** : https://streamlit.io/
- **Matplotlib** : https://matplotlib.org/

### Papiers Fondateurs
- Hay et al. (2008) - Randomization
- Liu & Terzi (2008) - k-degree anonymity
- Boldi et al. (2012) - (k,ε)-obfuscation
- Sala et al. (2011) - Differential Privacy for Graphs

### Thèse
**NGUYEN Huu-Hiep** (2016). *Anonymisation de Graphes Sociaux*.
Université de Lorraine, LORIA.

---

## 📞 Support

Pour toute question sur l'utilisation de l'application :
1. Consulter ce README
2. Consulter le GUIDE_EXPOSE.md
3. Lire les commentaires dans le code source

**Bonne présentation ! 🎉**
