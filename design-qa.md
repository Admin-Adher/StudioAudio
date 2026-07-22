# Design QA — contrôle de lecture et timeline

## Source visual truth

- Capture utilisateur : référence visuelle interne non versionnée.
- Dimensions : 318 × 110 px.
- État : contrôle arrêté, icône Lecture visible, zoom de navigateur élevé.
- Intention explicitée : réduire et recentrer l'icône Lecture/Pause, puis ouvrir la timeline en « Vue complète » par défaut.

## Implementation evidence

- Avant : `output/design-qa/audio-controls-before-full.png` — 718 × 717 px.
- Après le réglage du contrôle : `output/design-qa/audio-controls-after-full.png` — 718 × 717 px.
- Comparaison focalisée : `output/design-qa/audio-controls-source-after-comparison.png` — 656 × 110 px.
- Viewport CSS observé : 733 × 731 px, DPR navigateur 2. La capture d'implémentation normalisée par le navigateur a été agrandie ×2 uniquement dans la comparaison focalisée afin de rapprocher la densité de la capture source.
- État testé : fiche `test_audio.wav`, contrôle Lecture puis état Pause.

## Comparison history

### Pass 1

- P1 — l'icône Lecture utilisait 32 × 32 px, soit toute la surface du cercle de 32 px.
- P2 — le triangle paraissait optiquement décalé vers la gauche.
- P2 — les icônes précédent/suivant de 24 px dominaient le bouton principal.
- P2 — la timeline démarrait en « Zoom maximal » alors que l'utilisateur souhaite voir l'enregistrement complet.

### Fixes

- Cercle maintenu à 32 × 32 px pour préserver la cible tactile et l'emprise existante.
- Icônes Lecture et Pause ramenées à 15 × 15 px.
- Décalage optique de +0,75 px appliqué uniquement à l'icône triangulaire Lecture ; l'icône Pause reste géométriquement centrée.
- Icônes précédent/suivant ramenées à 19 × 19 px.
- Ombre allégée, état focus visible et retour d'appui ajouté.
- Valeur initiale de la timeline, état JavaScript et libellés accessibles passés de 100 à 0 / « Vue complète ».

### Pass 2

- Le rendu Lecture capturé montre une hiérarchie nette et un triangle correctement contenu dans le cercle.
- Mesure DOM Lecture : cercle 32 × 32 px, icône 15 × 15 px, centre X +0,75 px, centre Y 0 px.
- Mesure DOM Pause : cercle 32 × 32 px, icône 15 × 15 px, centre X 0 px, centre Y 0 px.
- Le passage Lecture → Pause a été observé dans l'interface.
- Le post-redémarrage final de la « Vue complète » n'a pas pu être recapturé : le navigateur intégré a bloqué le rechargement de l'adresse locale par sa politique de sécurité.

## Fidelity surfaces

- Typographie : aucun changement ; les libellés et la hiérarchie de la fiche sont conservés.
- Espacement : emprise du bouton inchangée, réduction limitée aux glyphes internes.
- Couleurs : palette beige/brun existante conservée, contraste blanc/brun inchangé.
- Icônes et qualité : les icônes vectorielles natives du lecteur Gradio sont conservées ; aucun substitut ou dessin CSS n'a été introduit.
- Copie : « Vue complète » et « Enregistrement complet » sont désormais les valeurs initiales cohérentes.
- Accessibilité : bouton toujours nommé Lecture/Pause, focus visible ajouté, cible de 32 px conservée.

## Verification

- 104 tests automatisés passent.
- Le contrôle Lecture/Pause a été vérifié visuellement et géométriquement.
- La validation visuelle post-redémarrage du nouvel état initial de la timeline reste à faire après un rafraîchissement utilisateur.

final result: blocked
