# Diagnostic des moteurs de transcription

Diagnostic réalisé le 21 juillet 2026 sur la machine cible : Intel Core Ultra 7
258V, 32 Go de RAM et Intel Arc 140V intégrée, sans GPU NVIDIA.

## Mesures locales

Les trois modèles déjà installés ont été mesurés sur le même extrait français
de 20 secondes, hors diarisation :

| Profil | Modèle | Chargement | Transcription | Observation |
|---|---|---:|---:|---|
| Très rapide | Whisper `base` INT8 | 2,13 s | 5,51 s | Rapide, mais plusieurs mots et noms fortement dégradés |
| Équilibré | Whisper `small` INT8 | 2,96 s | 12,29 s | Nettement plus lisible et cohérent |
| Précis | Whisper `large-v3-turbo` INT8 | 9,40 s | 22,13 s | Meilleurs noms sur l'extrait, mais plus lent que le temps réel sur CPU |

Le profil `small` a ensuite été mesuré sur 60 secondes réelles :

| Beam | Batch | Durée | Écart |
|---:|---:|---:|---|
| 3 | 4 | 24,85 s | ancienne configuration |
| 2 | 8 | 17,50 s | **29,6 % plus rapide**, qualité visuellement comparable |
| 1 | 8 | 15,71 s | encore plus rapide, mais davantage de phrases dégradées |

Un nouveau passage isolé, effectué pendant le benchmark OpenVINO, a donné
8,513 s, 10,649 s et 11,686 s avec `beam_size=2`, `batch_size=8`. Cette
dispersion confirme l'influence de l'état thermique et énergétique du portable.
Le pic mémoire du processus Faster-Whisper était de 945,6 à 955,4 Mio.

Décision actuelle : conserver `small` en mode Équilibré avec `beam_size=2` et
`batch_size=8` tant que le pilote OpenVINO n'est pas intégré et validé sur un
corpus corrigé. Les projections à partir d'un seul passage ne sont pas des
promesses : la diarisation, le décodage, les checkpoints et les exports
s'ajoutent au temps d'ASR.

## Candidats Hugging Face examinés

### OpenVINO Whisper large-v3-turbo INT8

[`OpenVINO/whisper-large-v3-turbo-int8-ov`](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov)
est désormais **testé réellement** sur cette machine. OpenVINO a exposé le CPU,
l'Intel Arc 140V et l'Intel AI Boost NPU. Le modèle INT8 occupe environ 0,77 Gio
et l'environnement de test isolé environ 0,64 Gio.

| Cible | Résultat local |
|---|---|
| CPU | 32,47 s à chaud pour 20 s : rejeté, plus lent que le temps réel |
| Arc 140V | 2,54 à 6,47 s pour 20 s avec mots horodatés |
| Arc 140V, 60 s | 5,82 s, 137 mots horodatés, RTF 0,097 |
| NPU | construction toujours inachevée après 360 s : rejeté pour l'UX actuelle |

Sur la minute testée au même moment, l'Arc réduit le temps d'inférence de 31,6 à
50,2 % par rapport aux passes Faster-Whisper contemporaines. Le texte reconnaît
mieux l'identifiant fictif « ClienteX » et reste exploitable, mais contient encore une erreur
probable sur « date de naissance ». Une vérité terrain est nécessaire avant de
calculer le WER.

Les timestamps mot-à-mot fonctionnent réellement, donc le moteur reste
compatible avec l'alignement des locuteurs et les commentaires sur passages. Le
cache disque compilé de la nightly testée doit cependant être désactivé : deux
redémarrages ont renvoyé uniquement `...`, alors que la même pipeline sans cache
a produit le texte complet. Verdict : **meilleur candidat à intégrer en pilote
sur l'Arc, avec pipeline persistante et repli automatique vers Faster-Whisper ;
pas encore remplacement automatique du mode Équilibré.**

Mesures et protocole complets :
[`output/benchmarks/openvino-result.md`](output/benchmarks/openvino-result.md).

### Cohere Transcribe 03-2026

[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
est un modèle Apache-2.0 de 2 milliards de paramètres qui prend en charge le
français et obtient d'excellents résultats de reconnaissance. Sa propre fiche
précise toutefois qu'il ne fournit ni timestamps ni diarisation. Il ne convient
donc pas comme moteur principal pour la navigation temporelle, les commentaires
sur passages et la séparation des interlocuteurs de cette application.

### MOSS Transcribe Diarize

[`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
est fonctionnellement très intéressant : texte, timestamps et locuteurs sont
produits en un passage. Sa fiche mise à jour annonce maintenant plus de 50
langues, et le test local confirme que le français fonctionne. Sur les 20 s :

- 88,609 s d'inférence CPU float32, soit un RTF de 4,430 ;
- premier token après 52,017 s ;
- neuf segments valides et deux locuteurs plausibles `S01` / `S02` ;
- pic mémoire de 5 288,9 Mio ;
- meilleure reconnaissance de « ClienteX », mais erreur probable « table de
  l'essence » pour « date de naissance ».

La voie officielle Windows retombe sur le CPU et ne sait pas exploiter l'Arc ;
les backends optimisés recommandés ciblent CUDA. L'installation a en plus
rencontré `WinError 206` à cause de `MAX_PATH`, contourné seulement avec un
lecteur temporaire court. MOSS prouve donc l'intérêt d'une diarisation unifiée,
mais il est environ 14,2 fois plus lent que la passe Équilibré de référence sur
ce test et ne convient pas au produit local actuel.

Mesures, sortie brute et segments :
[`output/benchmarks/moss-result.md`](output/benchmarks/moss-result.md).

### Autres candidats

- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  prend en charge le français et les timestamps, mais son environnement de
  référence est Linux/NVIDIA.
- [`eustlb/distil-large-v3-fr`](https://huggingface.co/eustlb/distil-large-v3-fr)
  est dédié au français et promet un bon rapport vitesse/qualité, mais il s'agit
  d'un modèle communautaire à valider et à convertir pour le backend actuel.
- Le `distil-large-v3` officiel général est anglais uniquement et ne doit pas
  être utilisé pour ces conversations françaises.

## Applications et pipelines open source examinés

Ces projets ne doivent pas être confondus avec de nouveaux modèles. Ils sont
intéressants soit comme moteur d'exécution, soit comme références produit et
UX. Les chiffres ci-dessous proviennent de mesures locales effectuées sur les
mêmes extraits de 20 et 60 secondes, hors diarisation.

### OpenWhispr et son backend `whisper.cpp` Vulkan

[`OpenWhispr`](https://github.com/OpenWhispr/openwhispr) est une application
Electron locale sous licence MIT. Elle combine notamment `whisper.cpp`,
`sherpa-onnx`, l'import par lots, des notes et une diarisation locale. Son
binaire Vulkan Windows officiel a réellement reconnu l'Intel Arc 140V et y a
chargé `large-v3-turbo`, sans repli CPU.

| Mesure | Résultat local |
|---|---:|
| Chargement du serveur | 8,89 s puis 16,94 s |
| 20 s, première passe | 19,01 s |
| 20 s, passe chaude | 12,33 s |
| 60 s, passe chaude | 34,84 s |
| Pic mémoire | environ 2,13 Gio |

Les timestamps de segments et de mots sont présents et le texte reconnaît
« ClienteX », mais la passe de 60 s est environ six fois plus lente qu'OpenVINO
sur la même Arc. Ce backend reste donc un éventuel secours multiplateforme, pas
le moteur Équilibré. Les idées pertinentes à reprendre sont le serveur
préchauffé, les téléchargements vérifiés par SHA-256, le repli CPU visible et
la diarisation ONNX Pyannote + CAM++ à évaluer séparément.

Mesures et audit complets :
[`output/benchmarks/openwhispr-vulkan/REPORT.md`](output/benchmarks/openwhispr-vulkan/REPORT.md).

### Vibe, Sona, Parakeet et Nemotron

[`Vibe`](https://github.com/thewh1teagle/vibe) est une application Tauri/Rust
MIT avec traitement par lots, aperçu progressif, nombreux exports, API HTTP,
CLI, diarisation et prise en charge annoncée des GPU Intel via Vulkan. Son
sidecar officiel Sona 0.3.5 voit bien `Vulkan0 — Intel Arc 140V`.

Deux modèles GGUF quantifiés d'environ 463 à 473 Mio ont été testés :

| Moteur Sona | Résultat local sur 20 s | Qualité fonctionnelle |
|---|---|---|
| Parakeet TDT 0.6B v3 Q4 | meilleur passage 8,36 s ; répétitions de 17,97 à 62,23 s | texte cohérent, 8/8 segments horodatés, mais erreur « table de naissance » |
| Nemotron 3.5 ASR 0.6B Q4 | 16,79 s puis 27,80 s | « date de naissance » correcte, mais environ 5,6 s du début omises et un seul segment |

Parakeet charge bien ses 698 tenseurs sur le GPU, mais son VAD obligatoire est
instable dans ce chemin Windows/Vulkan : selon la passe, cette étape seule a
pris de 0,44 à 58,29 s. Les textes identiques montrent que cette dispersion ne
vient pas de l'audio. Aucun test de 60 s n'a donc été retenu et Vibe/Sona ne
peut pas encore remplacer OpenVINO ou Faster-Whisper dans un produit où le
temps annoncé doit être fiable. Vibe reste néanmoins une excellente référence
pour la couche multi-moteurs, l'aperçu en direct, les exports et l'API locale.

La redistribution des poids demande plus de soin que la licence MIT de Vibe :
Parakeet est sous CC BY 4.0, Nemotron sous OpenMDW 1.1 et Sortformer sous NVIDIA
Open Model License. Un installateur doit embarquer les licences et notices,
documenter la conversion/quantification et épingler les artefacts par version
et SHA-256. Les dépôts GGUF/ONNX tiers actuellement utilisés ne fournissent pas
tous ces éléments de manière suffisamment robuste pour être copiés tels quels
dans notre distribution.

Mesures et audit complets :
[`output/benchmarks/vibe-sona-result.md`](output/benchmarks/vibe-sona-result.md).

### Diarisation ONNX légère Pyannote + CAM++

La chaîne officielle `sherpa-onnx` Pyannote Segmentation 3.0 + CAM++ suggérée
par OpenWhispr a été testée séparément sous Windows. Elle réduit fortement la
mémoire (environ 160 à 193 Mio contre 390 à 397 Mio pour `diarize`) et démarre
plus vite sur 20 s. En revanche, sur la minute réelle, elle attribue la même
voix à plusieurs questions/réponses successives puis change de locuteur au
milieu du récit continu de ClienteX. Le moteur actuel suit nettement mieux
l'alternance des deux personnes.

La passe chaude de 60 s prend en outre 21,03 s en FP32, contre 13,06 s pour le
moteur actuel. L'INT8 descend à 20,38 s à froid sans corriger l'erreur de
locuteurs. Verdict : **ne pas remplacer la diarisation de production** ; garder
ce pipeline uniquement comme piste R&D pour la faible mémoire, les
chevauchements ou de futures empreintes vocales.

Mesures, segments et licences :
[`output/benchmarks/onnx-diarization-result.md`](output/benchmarks/onnx-diarization-result.md).

### noScribe

[`kaixxx/noScribe`](https://github.com/kaixxx/noScribe) est une excellente
référence pour la relecture d'entretiens : traitement séquentiel de plusieurs
fichiers, Faster-Whisper, Pyannote, éditeur synchronisé avec l'audio, vitesse de
lecture, zoom et correction des noms. Sur Windows sans NVIDIA, sa version
générale reste toutefois orientée CPU ; elle n'apporte donc pas de voie Intel
Arc plus rapide que notre moteur actuel. Sa documentation prévient qu'une
heure d'entretien peut demander jusqu'à trois heures selon la machine.

Le projet est sous GPL-3.0 : ses comportements UX peuvent être réimplémentés
indépendamment, mais son code ne doit pas être copié dans une distribution que
nous souhaitons garder sous une licence différente. La file est documentée
pour la session courante ; aucune reprise exacte après redémarrage ni discussion
ancrée à un passage n'est démontrée par les sources consultées.

### Scribe CEMEA

[`Scribe CEMEA`](https://scribe.cemea.org/) est un service Flask auto-hébergeable
AGPLv3 basé sur Vosk/Kaldi, et non un nouveau modèle de transcription. Il prend
en charge le français, l'anglais et l'espagnol ainsi que les exports texte/SRT,
mais ne propose ni diarisation, ni progression locale, ni accélération Intel
Arc. Son modèle français `vosk-model-fr-0.22` et son déploiement Linux/Docker
ne sont pas compétitifs avec les moteurs mesurés ici. À retenir uniquement :
l'import par URL, les exports simples et la communication sur l'effacement des
données.

## Ordre recommandé pour la suite

1. Continuer à utiliser le mode Équilibré optimisé, sa file persistante et ses
   checkpoints comme valeur sûre.
2. Intégrer OpenVINO Arc derrière un mode pilote : pipeline GPU persistante,
   cache disque désactivé, contrôle de sortie et repli automatique CPU
   Faster-Whisper.
3. Constituer plusieurs extraits français corrigés, puis comparer WER/CER,
   timestamps, noms propres et temps total avec diarisation avant promotion.
4. Conserver la diarisation actuelle : la variante ONNX légère consomme moins
   de mémoire, mais sa séparation des deux voix est moins fiable sur l'audio
   réel testé.
5. Surveiller les prochaines versions de Sona/Vibe et répéter Parakeet lorsque
   l'instabilité VAD sur Vulkan Intel sera corrigée.
6. Ne pas intégrer MOSS CPU, Nemotron Sona, noScribe ou Scribe comme moteur
   principal dans l'application actuelle.
7. Garder Faster-Whisper CPU INT8 comme secours local fiable et hors ligne.
