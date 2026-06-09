# Chatbot ETP MICI

## Évaluation

Le répertoire [`evaluation`](evaluation/) contient le pipeline permettant de
générer les réponses de différents LLMs, de les faire noter par des juges et de
calculer les métriques de comparaison.

Les instructions d'utilisation sont disponibles dans
[`evaluation/README.md`](evaluation/README.md).

Commande principale:

```bash
python3 evaluation/run_evaluation.py <dataset.csv> baseline
```

## Chatbot

Cette partie contiendra le code du chatbot ETP spécialisé dans les MICI.

**TODO :** concevoir et implémenter l'application chatbot.
