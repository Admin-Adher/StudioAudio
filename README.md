# Studio Audio

Application Windows pour transcrire des conversations audio, distinguer les
interlocuteurs et produire des fichiers immédiatement partageables. Le calcul
est réalisé sur le PC : l'audio n'est envoyé à aucun service de transcription.

## Installation (une seule fois)

Prérequis : Windows 10/11, Python 3.10, 3.11 ou 3.12, au moins 8 Go de RAM et
environ 4 Go d'espace libre.

1. Double-cliquer sur `INSTALLER.bat`.
2. Attendre le message « Installation terminée » (5 à 15 minutes selon le PC).
3. Double-cliquer sur `LANCER.bat`.

Le navigateur s'ouvre sur <http://127.0.0.1:7860>. Garder la petite fenêtre de
lancement ouverte pendant l'utilisation.

Au premier lancement, un écran de préparation télécharge les trois niveaux de
qualité et le modèle de séparation des voix (environ 2,3 Go). La progression et
l'état de chaque composant sont affichés clairement. Une fois cet écran terminé,
aucun téléchargement n'interrompt la première transcription et l'application
peut fonctionner hors ligne.

## Utilisation

1. Déposer un ou plusieurs fichiers WAV, MP3, M4A, FLAC ou OGG dans la file.
2. Réorganiser les fichiers par glisser-déposer ou retirer ceux qui ne doivent
   pas être traités.
3. Choisir la qualité et la langue.
4. Indiquer le nombre réel d'interlocuteurs (deux par défaut).
5. Facultatif : saisir les noms des voix et quelques mots métier difficiles.
6. Cliquer sur **Lancer la file d'attente**.
7. Suivre le texte qui apparaît progressivement et l'état de chaque audio.
8. Choisir ensuite un audio dans **Relire et corriger**, modifier directement
   le tableau, puis cliquer sur **Exporter les corrections** si nécessaire.

Les audios sont traités l'un après l'autre dans l'ordre affiché. Ce mode évite
de saturer la mémoire du PC et réutilise le modèle déjà chargé : après le
premier audio d'un lot, les suivants démarrent plus rapidement.

### Rafraîchissement et reprise

- Le traitement est exécuté par un worker local indépendant de l'onglet du
  navigateur. Rafraîchir ou fermer la page n'interrompt donc plus la file.
- L'interface relit l'état durable chaque seconde : progression, étape et texte
  partiel réapparaissent automatiquement.
- Après un arrêt complet de l'application ou du PC, le dernier checkpoint est
  repris avec deux secondes de chevauchement afin de ne pas couper une phrase.
- Les mots horodatés, la langue, les réglages, l'étape de diarisation et le
  temps cumulé sont enregistrés. Un changement de modèle invalide seulement le
  checkpoint technique, jamais le texte déjà visible dans la fiche.
- Une fiche terminée n'est pas retranscrite lors de la reprise de la file.

Les étiquettes suivent l'ordre d'apparition : la première voix entendue est
« Interlocuteur 1 », la suivante « Interlocuteur 2 ».

### Profils

- **Très rapide** (`base`) : brouillon et contrôle rapide.
- **Équilibré** (`small`, CPU INT8, batch 8, beam 2) : meilleur compromis pour
  une conversation téléphonique. Ce réglage est environ 30 % plus rapide que
  l'ancienne configuration sur l'audio de test de cette machine.
- **Précis** (`large-v3-turbo`) : davantage de précision, mais plus lent et
  nettement plus lourd sur un PC sans carte graphique.

La durée dépend fortement du processeur, de la qualité audio et de la quantité
de parole. Le téléchargement des modèles est effectué séparément sur l'écran de
préparation du premier lancement.

### Intel Arc par défaut, avec repli automatique

Pour un nouvel espace de travail, le profil Équilibré sélectionne désormais par
défaut OpenVINO `large-v3-turbo` INT8 sur Intel Arc. Faster-Whisper CPU INT8
reste disponible comme choix manuel et comme moteur de secours automatique :

1. Fermer l'application.
2. Double-cliquer sur `INSTALLER_OPENVINO.bat` (environ 1,4 Go avec le runtime
   et le modèle).
3. Relancer `LANCER.bat`.
4. Vérifier dans **Paramètres** que **Intel Arc · pilote recommandé** est actif.

Le choix est mémorisé sur le PC. Les espaces existants qui avaient sélectionné
Faster-Whisper le conservent après la mise à jour ; la nouvelle valeur par
défaut ne leur est pas imposée. Le pilote est limité au profil Équilibré ; les
autres profils utilisent automatiquement Faster-Whisper. Si
OpenVINO, le modèle ou l'Arc ne sont pas disponibles, la transcription retombe
également sur Faster-Whisper et l'état est indiqué dans l'interface. Aucun
téléchargement OpenVINO n'est déclenché silencieusement depuis le navigateur.

Le modèle Arc conserve l'étiquette « pilote » : il est sensiblement plus rapide
sur la machine de test, mais ses formulations peuvent différer. Le repli et le
choix Standard local restent donc accessibles pendant sa validation sur un
corpus français corrigé.

### Profil métier Voyance

Pour une transcription en français, les deux moteurs reçoivent automatiquement
un contexte court et un lexique spécialisé : date/heure/lieu de naissance,
thème astral, signes distinctifs, tarot, oracles, arcanes, cartomancie,
médiumnité et numérologie. Les noms ou expressions ajoutés dans le champ de
vocabulaire restent prioritaires et sont dédupliqués.

Le guidage est volontairement borné. Il ne réécrit jamais un nombre, une date
ou un homophone après coup et ne transforme pas la transcription en texte
« plausible » sans preuve audio. Sur l'enregistrement de validation, il corrige
« table/tête de naissance » en « date de naissance » ; un second passage sans
ces mots reste inchangé, sans répétition ni ajout de vocabulaire métier. Il
s'agit d'un guidage lexical du modèle, pas encore d'un fine-tuning : la prochaine
étape de qualité restera un corpus d'extraits corrigés représentatif du métier.

Le comparatif local et l'analyse des modèles Hugging Face récents sont détaillés
dans [`DIAGNOSTIC_MODELES.md`](DIAGNOSTIC_MODELES.md).

## Fichiers produits

Chaque audio traité crée un sous-dossier dans `resultats` avec :

- un ZIP contenant tous les formats ;
- un document Word (`.docx`) ;
- un texte (`.txt`) ;
- un tableau compatible Excel (`.csv`) ;
- des sous-titres (`.srt` et `.vtt`) ;
- les données structurées (`.json`).

L'audio original n'est pas copié dans `resultats`.

Lorsqu'une file contient plusieurs audios, l'application produit aussi un ZIP
global. Celui-ci regroupe les ZIP complets de tous les audios réussis, tout en
laissant disponibles les téléchargements individuels.

## Partager l'application avec des collègues

### Logiciel Windows ou macOS

La version bureau ouvre désormais le studio dans une fenêtre dédiée, sans
terminal ni onglet de navigateur. Les builds natifs et leur signature sont
documentés dans [`packaging/README_DESKTOP.md`](packaging/README_DESKTOP.md).
Les collègues reçoivent ensuite un `Setup.exe` sous Windows ou un `.dmg` sous
macOS ; les modèles restent téléchargés une seule fois sur leur ordinateur.

Les données du logiciel sont conservées en dehors du dossier d'installation,
dans `%LOCALAPPDATA%\TranscriptionLocale` sous Windows ou
`~/Library/Application Support/TranscriptionLocale` sous macOS. Une mise à jour
du logiciel ne supprime donc pas les audios, transcriptions ou discussions.

### Package source historique

1. Double-cliquer sur `CREER_PACKAGE_PARTAGE.bat`.
2. Récupérer le ZIP créé dans le dossier `packages`.
3. Envoyer ce ZIP au collègue.
4. Le collègue décompresse tout, puis lance `INSTALLER.bat` et `LANCER.bat`.

Sur un PC Intel Arc, le collègue installe ensuite l'accélération recommandée
avec `INSTALLER_OPENVINO.bat`. Si elle n'est pas encore prête, l'application
reste utilisable immédiatement grâce au repli automatique sur Faster-Whisper.

Le package n'inclut ni vos audios, ni vos résultats, ni votre environnement
Python local. Chaque collègue télécharge ses modèles une fois sur son propre PC.

## Confidentialité

- L'interface écoute uniquement sur `127.0.0.1` : elle n'est pas exposée au
  réseau local ou à Internet.
- Les fichiers audio sont décodés dans un dossier temporaire supprimé après le
  traitement.
- La télémétrie Gradio et Hugging Face est désactivée par l'application.
- Une connexion Internet est utilisée uniquement pour installer le logiciel et
  télécharger les modèles publics au premier usage.

## Limites à connaître

- La séparation reconnaît des voix anonymes ; elle ne connaît pas l'identité
  réelle des personnes. Vérifier ou modifier les noms avant l'export.
- Lorsque deux personnes parlent exactement en même temps, une seule voix peut
  être attribuée au passage.
- Les noms propres, numéros, adresses et enregistrements téléphoniques très
  compressés doivent toujours être relus.
- Pour de meilleurs résultats, enregistrer chaque personne sur un canal séparé
  lorsque le matériel le permet.

## Dépannage

### Le navigateur ne s'ouvre pas

Ouvrir manuellement <http://127.0.0.1:7860>. Si la page ne répond pas, fermer
les anciennes fenêtres de lancement puis relancer `LANCER.bat`.

### L'installation échoue

Vérifier que Python 3.10 à 3.12 est installé depuis
<https://www.python.org/downloads/windows/> et qu'au moins 4 Go sont libres.
Un proxy ou un antivirus d'entreprise peut ralentir ou bloquer les
téléchargements ; dans ce cas, transmettre le message d'erreur au support
informatique.

### La séparation inverse les personnes

Modifier les noms dans la colonne « Interlocuteur », puis utiliser
**Exporter mes corrections**. Les modèles ne peuvent pas deviner les identités.

## Composants principaux

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) pour la
  reconnaissance vocale quantifiée sur CPU ;
- [diarize](https://github.com/FoxNoseTech/diarize) pour la séparation locale
  des voix (Silero VAD + WeSpeaker) ;
- [Gradio](https://www.gradio.app/) pour l'interface locale.
