TEST WHISPER LOCAL
1. Décompresse le ZIP.
2. Lance le dossier via un petit serveur HTTP/HTTPS (évite file://).
3. Ouvre index.html.
4. Clique sur « Charger Whisper » et attends « Whisper est prêt ».
5. Clique sur « Parler » et prononce la phrase.
6. Le prototype arrête après 2 s de silence, convertit en 16 kHz, transcrit localement et calcule Levenshtein.
WebGPU n'est PAS activé dans ce premier prototype.
Le premier chargement nécessite Internet pour récupérer Transformers.js et le modèle.
