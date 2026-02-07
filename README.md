# Wish Upon a Block

Wish Upon a Block is a lightweight, low-latency guideline tetromino game. The core idea is a piece generator trained on labeled board states so the “next piece” feels just right.

## Play
- Live: [wishuponablock.com](https://wishuponablock.com/)

## What’s Special
- ML-assisted piece generator ("Wish Upon a Block")
- Snapshot + labeling pipeline for supervised training
- Fast, responsive gameplay tuned for low latency

## Local Development
```bash
npm install
npm run dev
```

## Data Collection (Snapshots + Labels)
The game can record anonymized board snapshots and collect label feedback. Uploads can be routed to a Cloudflare Worker backed by D1.

## Deploy
See `/Users/max/Desktop/wishuponablock/wishuponablock/docs/DEPLOYMENT.md` for Cloudflare deployment steps and D1 migrations.
