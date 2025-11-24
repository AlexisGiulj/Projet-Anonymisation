@echo off
echo ========================================
echo  Application d'Anonymisation de Graphes
echo ========================================
echo.
echo Tentative de lancement avec Streamlit...
echo.

python -m streamlit run graph_anonymization_app.py 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERREUR: Streamlit n'est pas installe correctement.
    echo.
    echo Voulez-vous lancer la version simplifiee ? (O/N)
    set /p choice="> "

    if /i "%choice%"=="O" (
        echo.
        echo Lancement de la version simplifiee...
        python graph_anonymization_demo.py
    ) else (
        echo.
        echo Installation de Streamlit recommandee :
        echo   python -m pip install streamlit --no-deps
        echo   python -m pip install altair blinker cachetools click pandas
        echo.
        pause
    )
)

pause
