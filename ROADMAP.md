Here is the list of features in the backlog, with corresponding "impact score" (1-5) — the importance of the feature in my judgment. Generally the more the score, the sooner I'll get to it
## Gameplay
- Replays — 2 // lowered priority, blocked by personalization
- Refiew top-out behaviour. Make it more lenient like in tetrio/jstris — 5
- Wish Upon a Bag — a less intrusive ML piece generator that shuffles the next 7 bag to try and help — 4
- For Wish Upon a Block generator display the probabilities with which the piece was generated, basically the network outputs — 5
- For Wish Upon a Block generator maybe it is possible to display the most likely next piece. Will require completely new ML pipeline — 1
- Bots — 2
- RL bots and piece generators — 1
- Change default softdrop from instant to increased gravity. Add softdrop slider — 5
- Curse Upon a Block, a generator that gives you the worst piece — 5

## UI/UX
- Patch Notes — 3
- Background Music — 3
- Favicon — 3
- In game stats — PPS, Attack, Finesse — 5
- More SFX - 2
- UI rework + actual design — 1
- Add ghost piece opacity slider — 5

## Labeling and ML
- Snapshot Explorer — 5
    - Heuristics for clustering snapshots based on the board itsel (?)
- Unsupervised learning (press "Hold" = don't want that = negative signal?) — 2
- Different generators for pro and noob — 2 // lowered priority, too much data required
- Replace softmax with threshold + flatter normalization - 3
- Labeling gameification — 5
    - Progress bar towards next checkpoint
- Data quality (versioning, source vs labeling intent discrepancy, label balancing) — 5
- Show active piece in labeling (and log it) — 5

## Personalization
- User authentication + profile + stats — 3
- Personalized ML generators (this is a big one, but also a big one, you know...) — 3