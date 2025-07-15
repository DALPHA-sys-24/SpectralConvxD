# SpectralConvxD

Convolution Neural Networks are a paradigm for image recognition.


# Guide des commandes Git

## Configuration initiale
```
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@exemple.com"
git config --global init.defaultBranch main
```

## Créer un dépôt

**Nouveau projet :**
```
git init
git add .
git commit -m "Premier commit"
```

**Cloner un dépôt existant :**
```
git clone https://github.com/utilisateur/depot.git
cd depot
```

## Commandes de base quotidiennes

**Vérifier l'état :**
```
git status                    # État des fichiers
git log                      # Historique des commits
git log --oneline            # Historique condensé
```

**Ajouter des fichiers :**
```
git add fichier.txt          # Ajouter un fichier spécifique
git add .                    # Ajouter tous les fichiers modifiés
git add *.js                 # Ajouter tous les fichiers .js
```

**Valider les changements :**
```
git commit -m "Message descriptif"
git commit -am "Message"     # Add + commit des fichiers déjà trackés
```

## Travailler avec les dépôts distants

**Ajouter un dépôt distant :**
```
git remote add origin https://github.com/utilisateur/depot.git
git remote -v                # Voir les dépôts distants
```

**Envoyer les changements :**
```
git push origin main         # Première fois
git push                     # Ensuite
git push -u origin main      # Lier la branche locale à la distante
```

**Récupérer les changements :**
```
git pull                     # Récupérer et fusionner
git fetch                    # Récupérer sans fusionner
```

## Gestion des branches

**Créer et changer de branche :**
```
git branch                   # Lister les branches
git branch nouvelle-branche  # Créer une branche
git checkout nouvelle-branche # Changer de branche
git checkout -b nouvelle-branche # Créer et changer en une commande
git switch nouvelle-branche  # Méthode moderne pour changer
```

**Fusionner les branches :**
```
git checkout main
git merge nouvelle-branche
git branch -d nouvelle-branche # Supprimer la branche
```

## Annuler des modifications

**Annuler des modifications non commitées :**
```
git checkout -- fichier.txt # Annuler les modifications d'un fichier
git reset HEAD fichier.txt   # Retirer un fichier de l'index
git reset --hard            # Annuler toutes les modifications
```

**Annuler des commits :**
```
git reset --soft HEAD~1     # Annuler le dernier commit (garder les changements)
git reset --hard HEAD~1     # Annuler le dernier commit (perdre les changements)
git revert HEAD             # Créer un commit qui annule le précédent
```

## Commandes utiles

**Voir les différences :**
```
git diff                    # Changements non indexés
git diff --staged           # Changements indexés
git diff HEAD~1             # Comparer avec le commit précédent
```

**Ignorer des fichiers :**
```
echo "node_modules/" > .gitignore
echo "*.log" >> .gitignore
git add .gitignore
git commit -m "Ajout du .gitignore"
```

**Nettoyer le dépôt :**
```
git clean -f                # Supprimer les fichiers non trackés
git clean -fd               # Supprimer fichiers et dossiers non trackés
```

## Workflow complet typique

### 1. Démarrer un nouveau projet :
```
mkdir mon-projet
cd mon-projet
git init
echo "# Mon Projet" > README.md
git add README.md
git commit -m "Initial commit"
git remote add origin https://github.com/utilisateur/mon-projet.git
git push -u origin main
```

### 2. Travailler au quotidien :
```
git pull                    # Récupérer les dernières modifications
# ... faire des modifications ...
git add .
git commit -m "Description des changements"
git push
```

### 3. Créer une fonctionnalité :
```
git checkout -b feature/nouvelle-fonctionnalite
# ... développer la fonctionnalité ...
git add .
git commit -m "Ajout de la nouvelle fonctionnalité"
git push origin feature/nouvelle-fonctionnalite
git checkout main
git merge feature/nouvelle-fonctionnalite
git push origin main
git branch -d feature/nouvelle-fonctionnalite
```

## Résumé des commandes essentielles

Pour débuter avec Git, maîtrisez d'abord ces commandes de base :
- git add . (ajouter les fichiers)
- git commit -m "message" (valider les changements)
- git push (envoyer vers le dépôt distant)
- git pull (récupérer les changements)
- git status (vérifier l'état)

Ces commandes couvrent 95% des besoins quotidiens avec Git. Ajoutez progressivement les autres selon vos besoins.
