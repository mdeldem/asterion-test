Test du solver [Asterion](https://github.com/asterion-solver/asterion)

## Utilisation

Traiter une ou plusieurs images FITS individuellement :

```powershell
uv run asterion-test "data/image 1.fits" "data/image 2.fits"
```

Traiter toutes les images d'une série à partir d'un nom d'exemple. Le compteur doit
être placé à la fin du nom, juste avant l'extension; les fichiers sont traités par
ordre numérique :

```powershell
uv run asterion-test --series "data/t_NSV 11142-001.fits"
```

Les sauvegardes sont désactivées par défaut. Les activer indépendamment avec
`--save-fits` et `--save-jpg`; `--output-dir` devient alors obligatoire :

```powershell
uv run asterion-test --series "data/t_NSV 11142-001.fits" `
  --save-fits --save-jpg --output-dir "data/output"
```

Les chemins relatifs sont résolus depuis le répertoire courant. Les fichiers
produits gardent le nom de base de la source avec le suffixe `_solved.fits` ou
`_solved.jpg`.
