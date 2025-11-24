@echo off
chcp 65001 > nul
title Anonymisation de Graphes Sociaux
color 0A

:: Changer vers le répertoire du script
cd /d "%~dp0"

:MENU
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                                                                ║
echo ║      🔒 ANONYMISATION DE GRAPHES SOCIAUX 🔒                   ║
echo ║                                                                ║
echo ║      Basé sur la thèse de NGUYEN Huu-Hiep (2016)             ║
echo ║                                                                ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo.
echo  Choisissez une option :
echo.
echo   [1] Application Interactive (Streamlit - RECOMMANDÉ)
echo.
echo   [2] Version Batch (Génération d'images PNG)
echo.
echo   [3] Installer les dépendances Python
echo.
echo   [4] Ouvrir le dossier du projet
echo.
echo   [5] Lire la documentation
echo.
echo   [0] Quitter
echo.
echo ════════════════════════════════════════════════════════════════
echo.
set /p choice="Votre choix : "

if "%choice%"=="1" goto STREAMLIT
if "%choice%"=="2" goto BATCH
if "%choice%"=="3" goto INSTALL
if "%choice%"=="4" goto FOLDER
if "%choice%"=="5" goto DOCS
if "%choice%"=="0" goto EXIT
goto MENU

:STREAMLIT
cls
echo.
echo ══════════════════════════════════════════════════════════════
echo  Lancement de l'application Streamlit...
echo ══════════════════════════════════════════════════════════════
echo.
echo  L'application va s'ouvrir dans votre navigateur à l'adresse :
echo  http://localhost:8501
echo.
echo  Pour arrêter l'application, fermez cette fenêtre ou appuyez
echo  sur Ctrl+C
echo.
echo ══════════════════════════════════════════════════════════════
echo.
pause

python -m streamlit run graph_anonymization_app.py 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ╔════════════════════════════════════════════════════════════════╗
    echo ║  ERREUR : Streamlit n'est pas installé ou ne fonctionne pas   ║
    echo ╚════════════════════════════════════════════════════════════════╝
    echo.
    echo  Solutions possibles :
    echo.
    echo   1. Installer Streamlit :
    echo      python -m pip install streamlit
    echo.
    echo   2. Utiliser l'option [3] du menu pour installer automatiquement
    echo.
    echo   3. Utiliser l'option [2] pour la version batch (sans Streamlit)
    echo.
    pause
)
goto MENU

:BATCH
cls
echo.
echo ══════════════════════════════════════════════════════════════
echo  Lancement de la version batch...
echo ══════════════════════════════════════════════════════════════
echo.
echo  Cette version va générer 3 fichiers PNG dans le dossier :
echo   - graph_anonymization_comparison.png
echo   - degree_distributions.png
echo   - metrics_comparison.png
echo.
echo ══════════════════════════════════════════════════════════════
echo.
pause

python graph_anonymization_demo.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ╔════════════════════════════════════════════════════════════════╗
    echo ║  ✓ Visualisations générées avec succès !                      ║
    echo ╚════════════════════════════════════════════════════════════════╝
    echo.
    echo  Les fichiers PNG ont été créés dans ce dossier.
    echo.
    echo  Voulez-vous ouvrir le dossier ? (O/N)
    set /p open="  > "
    if /i "%open%"=="O" explorer .
) else (
    echo.
    echo ╔════════════════════════════════════════════════════════════════╗
    echo ║  ERREUR : Impossible de générer les visualisations            ║
    echo ╚════════════════════════════════════════════════════════════════╝
    echo.
    echo  Vérifiez que Python et les dépendances sont installés.
    echo.
)
pause
goto MENU

:INSTALL
cls
echo.
echo ══════════════════════════════════════════════════════════════
echo  Installation des dépendances Python...
echo ══════════════════════════════════════════════════════════════
echo.
echo  Cette opération peut prendre plusieurs minutes.
echo.
pause

echo.
echo [1/5] Installation de NetworkX...
pip install networkx --quiet

echo [2/5] Installation de Matplotlib...
pip install matplotlib --quiet

echo [3/5] Installation de NumPy...
pip install numpy --quiet

echo [4/5] Installation de SciPy...
pip install scipy --quiet

echo [5/8] Installation de PyArrow...
pip install pyarrow --quiet

echo [6/8] Installation de Streamlit...
pip install streamlit --no-deps --quiet

echo [7/8] Installation des dépendances Streamlit...
pip install altair blinker cachetools click pandas protobuf pydeck requests tenacity toml tornado typing-extensions watchdog --quiet

echo [8/8] Installation de GitPython et correction d'Altair...
pip install gitpython "altair<6,>=4.0" --quiet

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ╔════════════════════════════════════════════════════════════════╗
    echo ║  ✓ Toutes les dépendances ont été installées !               ║
    echo ╚════════════════════════════════════════════════════════════════╝
) else (
    echo.
    echo ╔════════════════════════════════════════════════════════════════╗
    echo ║  ⚠ Certaines dépendances n'ont pas pu être installées        ║
    echo ║    Vous pouvez utiliser la version batch (option 2)           ║
    echo ╚════════════════════════════════════════════════════════════════╝
)
echo.
pause
goto MENU

:FOLDER
cls
echo.
echo  Ouverture du dossier du projet...
echo.
explorer .
goto MENU

:DOCS
cls
echo.
echo ══════════════════════════════════════════════════════════════
echo  Documentation disponible
echo ══════════════════════════════════════════════════════════════
echo.
echo  Fichiers de documentation dans ce dossier :
echo.
echo   [1] README.md           - Documentation complète du projet
echo   [2] README_APP.md       - Guide d'utilisation de l'application
echo   [3] GUIDE_EXPOSE.md     - Guide pour votre présentation
echo.
echo   [0] Retour au menu principal
echo.
echo ══════════════════════════════════════════════════════════════
echo.
set /p doc="Quel fichier ouvrir ? "

if "%doc%"=="1" start README.md
if "%doc%"=="2" start README_APP.md
if "%doc%"=="3" start GUIDE_EXPOSE.md
if "%doc%"=="0" goto MENU

goto DOCS

:EXIT
cls
echo.
echo  Merci d'avoir utilisé l'application !
echo.
echo  Bonne chance pour votre exposé ! 🎉
echo.
timeout /t 2 > nul
exit

