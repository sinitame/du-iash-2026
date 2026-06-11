# Évaluation de chatbots ETP MICI

Le pipeline sépare strictement trois opérations:

1. générer les réponses des chatbots;
2. faire évaluer ces réponses par un ou plusieurs juges;
3. calculer les métriques à partir des scores persistés.

Cette séparation évite de rappeler les APIs lorsqu'on modifie les métriques.

## Commande complète

Le script principal prend exactement deux entrées:

```bash
python3 evaluation/run_evaluation.py <dataset.csv> <mode>
```

Modes:

- `baseline`: évalue les cinq modèles avec le prompt minimal;
- `prompt`: évalue les prompts baseline et step-by-step pour chaque modèle;
- `rag`: compare la baseline et la variante RAG pour les cinq modèles.

Exemples:

```bash
python3 evaluation/run_evaluation.py \
  evaluation/data/dataset_test.csv \
  baseline

python3 evaluation/run_evaluation.py \
  evaluation/data/dataset_test.csv \
  prompt \
  --concurrency 4

python3 evaluation/run_evaluation.py \
  evaluation/data/dataset_test.csv \
  rag \
  --concurrency 4
```

Cette commande enchaîne automatiquement génération, scoring avec le juge fixe et
calcul des métriques.

Le mode `prompt` exécute les variantes `baseline` et `step_by_step` afin de
permettre leur comparaison dans un seul run. Il n'est donc pas nécessaire de
lancer ensuite le mode `baseline`: ses réponses et scores auront déjà été
produits. Le récapitulatif final indique le nombre d'appels API et de résultats
réutilisés depuis le cache.

Le mode `rag` réutilise les mêmes baselines et génère des systèmes séparés
nommés `<systeme>__rag`. Le scoring compare donc le même modèle avec et sans
contexte documentaire. Ses comparaisons sont écrites dans
`comparison_rag.json`, `comparison_rag.csv` et
`comparison_rag_detailed.csv`, sans remplacer les comparaisons du mode prompt.
Par défaut, les baselines existantes sont uniquement lues depuis le cache:
`rag_generate_baseline` vaut `false`. Passez cette option à `true` dans la
configuration uniquement pour générer aussi les baselines manquantes pendant le
run RAG.

Pour exécuter les cinq modèles, les variables suivantes doivent être définies:

```bash
export OPENAI_API_KEY="..."
export MISTRAL_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

La progression est affichée dans le terminal et persistée dans:

```text
evaluation/outputs/progress/<dataset>_<mode>.json
```

En cas d'interruption, relancer la même commande reprend les réponses et scores
déjà sauvegardés grâce au cache.

`--concurrency` définit le nombre maximal d'appels API simultanés pour un système.
Une limite plus basse peut être définie par fournisseur avec
`provider_transport.<provider>.max_concurrency`. La configuration fournie limite
Mistral à 4 appels simultanés et augmente ses retries pour mieux absorber les
rate limits et les erreurs réseau temporaires.
La valeur par défaut est `1`. Commencez avec `3` ou `4`; une valeur trop élevée
peut provoquer des erreurs de rate limit chez les fournisseurs.

Si certains appels parallèles échouent, les appels réussis sont tout de même
persistés. Une relance reprend uniquement les éléments manquants.

Pour scorer les réponses déjà disponibles après une interruption, sans relancer
la génération:

```bash
python3 evaluation/run_evaluation.py \
  evaluation/data/dataset.csv \
  prompt \
  --partial-summary \
  --concurrency 4
```

Cette commande peut effectuer des appels au juge, mais aucun appel aux modèles
évalués. Elle écrit `comparison_partial.json` et `comparison_partial.csv`, avec
le nombre de questions scorées et le taux de couverture de chaque système. Ces
résultats ne doivent pas être comparés entre systèmes ayant des couvertures
différentes.

`type_attendu`, `points_cles` et `signaux_securite` peuvent être absents du CSV;
le juge reçoit alors la valeur « Non renseigné ». En revanche, chaque ligne doit
posséder un `id` unique et non vide pour garantir le cache et la reprise.

Les erreurs transitoires (`429`, `5xx`, timeout, connexion et réponse vide) sont
réessayées automatiquement jusqu'à trois fois avec un backoff exponentiel. Le
champ `api_attempts` persiste le nombre de tentatives. Après une réponse vide, la
limite de sortie est doublée jusqu'à quatre fois la valeur initiale; la valeur
réellement utilisée est persistée dans `api_max_tokens`. Au-delà de 10 appels
simultanés, un avertissement recommande une concurrence comprise entre 4 et 8.

Les modèles de raisonnement peuvent consommer une partie de la limite de sortie
avant de produire le texte visible. Une limite `max_tokens` peut être définie par
système; une réponse vide est considérée comme un échec de génération et n'est
plus persistée.

## Structure des résultats

```text
evaluation/outputs/
├── responses/
│   └── <systeme>.jsonl
├── <version_juge>/
│   ├── scores/
│   │   └── <systeme>/<juge>.jsonl
│   └── metrics/
│       ├── <systeme>.csv
│       ├── <systeme>.summary.json
│       ├── comparison.json
│       └── comparison.csv
├── progress/
│   └── <dataset>_<mode>.json
└── run_configs/
    └── <configuration utilisée>.json
```

`<version_juge>` combine le nom du template et le début de son hash, par exemple
`judge_v2_f9de70a0`. Les scores et métriques d'une calibration sont regroupés
dans ce répertoire. Supprimer une version revient donc à supprimer un seul
répertoire.

Les fichiers JSONL sont append-only. Chaque requête possède un hash calculé à
partir du modèle, du prompt, des paramètres et de l'entrée.

- Une relance identique réutilise le résultat persisté.
- Une modification du prompt, du modèle ou des paramètres crée une nouvelle entrée.
- L'écriture est synchronisée après chaque réponse réussie. Une interruption ne
  fait donc pas perdre les appels précédents.
- La réponse API brute, le texte extrait, la latence et l'usage sont conservés.

## 1. Générer les réponses

Chaque système associe un modèle à un fichier `prompt_file`. Deux variantes sont
disponibles:

- `evaluation/prompts/chatbot_baseline.txt`: consignes minimales;
- `evaluation/prompts/chatbot_step_by_step.txt`: méthode de construction interne
  calibrée sur le style ETP du CSV V1: réponse directe, 2 à 4 phrases, nuance
  clinique et orientation uniquement lorsqu'elle est pertinente.

Le prompt fait partie du hash de cache. Modifier son contenu crée donc une nouvelle
requête, tandis qu'une relance sans modification réutilise les réponses persistées.

## Pouvoir discriminant du scoring

Le juge `judge_v2.txt` réserve le score maximal aux réponses qui couvrent tous les
points clés et signaux de sécurité applicables. Il ne pénalise pas une
reformulation et n'exige pas de mentionner une précaution hors contexte.

Le score global historique est conservé. Les résumés ajoutent aussi:

- le taux de couverture complète en exactitude et sécurité;
- le taux de réponses excellentes sur les quatre critères;
- le taux de réponses vides et d'échecs techniques;
- le score minimum, le percentile 10 et la dispersion des scores.

Une réponse vide reçoit automatiquement un score global nul sans appel au juge.
Elle est classée comme échec technique plutôt que comme erreur médicale critique,
ce qui évite qu'un juge reconstruise involontairement la réponse depuis la
référence.

Si la baseline globale contient un échec technique, les gains absolus contre cette
baseline ne sont pas calculés. Cela évite de présenter comme amélioration de
contenu la simple correction d'un problème de génération.

Ces indicateurs évitent qu'une moyenne élevée masque quelques réponses faibles.
Pour comparer sérieusement les systèmes, utiliser un jeu de test couvrant
plusieurs thèmes et suffisamment de questions. Le fichier `dataset_test.csv`
contient seulement 9 questions d'un seul thème et sert surtout de smoke test.

Le calcul produit trois fichiers de comparaison:

- `comparison.json`: résultats complets structurés;
- `comparison.csv`: tableau synthétique `model`, `Baseline`,
  `Baseline + prompt` ou `Baseline + RAG`, avec séparateur `;` et décimales
  françaises;
- `comparison_detailed.csv`: toutes les métriques disponibles par système.

Pour les systèmes RAG, le résumé contient aussi le taux de contextes non vides,
le nombre moyen de chunks et la latence moyenne du retrieval.

Test sans API:

```bash
python3 evaluation/generate_responses.py \
  --system test_technique \
  --limit 10
```

Pour un vrai système:

```bash
export OPENAI_API_KEY="votre-nouvelle-cle"
python3 evaluation/generate_responses.py \
  --system openai_gpt_5_mini
```

Le panel est limité à cinq modèles, répartis en deux groupes indicatifs:

Modèles généralistes avancés:

- OpenAI: `gpt-5-mini`;
- Mistral: `mistral-medium-2508`;
- Anthropic: `claude-sonnet-4-6`.

Modèles compacts:

- Mistral: `ministral-8b-2512`;
- Anthropic: `claude-haiku-4-5-20251001`.

Les groupes servent à éviter les comparaisons manifestement déséquilibrées. Ils
restent indicatifs, car les fournisseurs ne publient pas tous des caractéristiques
directement comparables.

Chaque modèle possède une baseline et une variante step-by-step:

```bash
python3 evaluation/generate_responses.py \
  --system openai_gpt_5_mini_step_by_step
```

Pour essayer une nouvelle variante:

1. Créer un fichier, par exemple `evaluation/prompts/chatbot_v2.txt`.
2. Ajouter une entrée dans `systems` avec un nom unique et:

```json
{
  "name": "openai_gpt_5_mini_v2",
  "model": "gpt-5-mini",
  "prompt_file": "prompts/chatbot_v2.txt"
}
```

Les autres paramètres fournisseur doivent être repris depuis l'entrée du même
modèle. Le nom unique sépare les réponses, scores et métriques de chaque variante.

Pour générer toutes les variantes configurées:

```bash
python3 evaluation/generate_responses.py
```

Relancer la même commande n'effectue pas de nouveaux appels pour les requêtes déjà
présentes.

## 2. Scorer avec les juges

Le prompt actif du juge est `evaluation/prompts/judge_v2.txt`. La version V1 reste
disponible pour reproduire les anciens résultats. Le prompt est un template utilisant
des variables comme:

```text
{{question_patient}}
{{age}}
{{reponse_attendue}}
{{signaux_securite}}
{{reponse_chatbot}}
```

Le juge produit uniquement les observations élémentaires:

- exactitude métier: 0 à 3;
- sécurité médicale: 0 à 3;
- adaptation au profil: 0 à 2;
- qualité conversationnelle: 0 à 2;
- erreur de sécurité critique;
- justifications et listes des problèmes.

Il ne calcule pas le score global.

Test sans API:

```bash
python3 evaluation/score_responses.py \
  --system test_technique \
  --judge juge_test \
  --limit 10
```

Pour un vrai juge:

```bash
export OPENAI_API_KEY="votre-nouvelle-cle"
python3 evaluation/score_responses.py \
  --system openai_gpt_5_mini \
  --judge openai_gpt_4_1_mini_judge
```

Le juge utilise `gpt-4.1-mini` et le mode de sortie JSON.
Le même juge doit être utilisé pour tous les modèles répondants afin de préserver
la comparabilité.

Exemple avec une variante de prompt:

```bash
python3 evaluation/score_responses.py \
  --system mistral_small_2603_step_by_step \
  --judge openai_gpt_4_1_mini_judge
```

Pour scorer tous les systèmes configurés avec ce juge:

```bash
python3 evaluation/score_responses.py \
  --judge openai_gpt_4_1_mini_judge
```

### Erreurs d'accès OpenAI

Une erreur `401` ou `403` indique généralement une clé invalide/révoquée ou un
modèle non autorisé pour le projet API. Créez une nouvelle clé dans le projet
OpenAI concerné, vérifiez que la facturation et l'accès au modèle sont actifs, puis:

```bash
unset OPENAI_API_KEY
export OPENAI_API_KEY="nouvelle-cle"
```

Ne placez jamais la clé dans `config.example.json` ou dans un fichier versionné.

Pour une clé OpenAI avec permissions `Restricted`, le pipeline actuel nécessite:

- `Chat completions (/v1/chat/completions)`: `Write`;
- `List models`: `Read` est recommandé pour le diagnostic.

Pour un environnement de développement, sélectionner temporairement `All` est
l'option la plus simple.

Les sorties brutes et structurées des juges sont persistées dans `scores/`.
Une relance identique utilise le cache.

## 3. Calculer les métriques

Cette étape ne fait aucun appel API:

```bash
python3 evaluation/compute_metrics.py
```

Cette commande sans `--system` régénère `comparison.json` avec tous les systèmes
qui possèdent des réponses et des scores persistés. Une exécution ciblée, par
exemple `--system openai_gpt_5_mini`, ne modifie plus la comparaison globale.

Le score global est normalisé sur 100:

```text
100 × (
  0,40 × exactitude / 3
  + 0,40 × sécurité / 3
  + 0,10 × adaptation / 2
  + 0,10 × qualité / 2
)
```

Si au moins un juge détecte une erreur de sécurité critique, le score global de la
réponse est fixé à zéro.

Les rapports contiennent:

- les moyennes globales et par critère;
- le taux d'erreurs critiques;
- les résultats par âge, thème et niveau de risque;
- le gain absolu par rapport à la baseline.
- le gain de chaque variante par rapport au prompt baseline du même modèle.

Le CSV détaillé reste volontairement simple:

```text
question_id
age
theme
niveau_risque
exactitude_metier
securite_medicale
adaptation_profil
qualite_conversationnelle
erreur_securite_critique
score_global
```

Si plusieurs juges sont configurés, leurs scores sont moyennés. Le détail de chaque
jugement reste disponible dans `scores/`, sans alourdir le rapport de métriques.

Le code de `compute_metrics.py` peut être modifié et relancé autant de fois que
nécessaire sans régénérer les réponses ni rappeler les juges.

## 4. Évaluer le RAG

Installer les dépendances dans l'environnement Python utilisé pour l'évaluation:

```bash
python3 -m pip install -r evaluation/requirements-rag.txt
```

Préparer les chunks à partir des PDF, images ou vidéos:

```bash
python3 -m pip install -r pre-processing/requirements.txt

python3 pre-processing/prepare.py \
  --pdfs-dir chemin/vers/pdfs \
  --output-dir rag/corpus
```

L'extraction OCR nécessite également les exécutables système `tesseract` et
`ffmpeg`.

La traduction est optionnelle. Elle est utile si le dataset contient des
questions en anglais ou en créole:

```bash
python3 pre-processing/traduction.py \
  --input-dir rag/corpus \
  --output-dir rag/corpus_translated \
  --model gpt-5-mini
```

Construire ensuite l'index FAISS:

```bash
python3 rag/indexation.py \
  --chunks-files \
    rag/corpus/chunks.jsonl \
    rag/corpus_translated/chunks_en.jsonl \
    rag/corpus_translated/chunks_gcf.jsonl \
  --output-dir rag/vectorstore_mici \
  --overwrite \
  --test-query "Que faire en cas de fatigue avec une MICI ?"
```

Si seules les sources françaises existent, fournir uniquement
`rag/corpus/chunks.jsonl`. Le chemin de l'index et le modèle d'embeddings sont
configurés dans la section `rag` de `evaluation/config.example.json`.

Lancer enfin l'évaluation:

```bash
python3 evaluation/run_evaluation.py \
  evaluation/data/dataset_questions_mici_270_V1_checked.csv \
  rag \
  --concurrency 4
```

Le manifest de l'index fait partie du hash du cache. Reconstruire le corpus crée
donc de nouvelles réponses RAG sans invalider les réponses baseline.
