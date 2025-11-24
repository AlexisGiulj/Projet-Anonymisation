# Installation de Streamlit pour Python 3.14

## Problème Résolu

Python 3.14 est très récent et Streamlit 1.51.0 nécessite PyArrow < 22, mais PyArrow 22.0.0 est la première version compatible avec Python 3.14.

## Solution Appliquée

L'installation a été adaptée pour contourner le conflit de version :

1. **Installation de PyArrow 22.0.0** (seule version compatible Python 3.14)
   ```bash
   pip install pyarrow==22.0.0
   ```

2. **Installation de Streamlit sans dépendances** (évite le conflit PyArrow)
   ```bash
   pip install streamlit --no-deps
   ```

3. **Installation manuelle des dépendances de Streamlit**
   ```bash
   pip install altair blinker cachetools click pandas protobuf pydeck requests tenacity toml tornado typing-extensions watchdog
   ```

4. **Installation de GitPython et correction d'Altair**
   ```bash
   pip install gitpython "altair<6,>=4.0"
   ```

## Lancement de l'Application

Utilisez **toujours** `python -m streamlit run` au lieu de `streamlit run` :

```bash
python -m streamlit run graph_anonymization_app.py
```

Cette commande contourne le problème de PATH avec streamlit.exe.

## Compatibilité

✅ **Testé et Fonctionnel** :
- Python 3.14.0
- PyArrow 22.0.0
- Streamlit 1.51.0
- Windows 10/11

⚠️ **Avertissement de compatibilité** :
Pip affiche un avertissement sur l'incompatibilité PyArrow, mais l'application fonctionne correctement.

## Lanceurs Mis à Jour

Les fichiers suivants ont été modifiés pour utiliser la bonne commande :
- `LANCER.bat` - Menu principal (Option 1)
- `LANCER_APP.bat` - Lanceur simple

## Réinstallation

Si vous devez réinstaller Streamlit, utilisez l'option [3] du menu `LANCER.bat` qui applique automatiquement cette méthode.
