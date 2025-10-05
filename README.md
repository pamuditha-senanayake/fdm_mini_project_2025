# RetailIQ – Sales Prediction & Customer Insights Platform

## Overview

RetailIQ is a retail analytics platform that predicts customer behavior, forecasts sales, and generates actionable insights.

## Features

* Sales Forecasting (SARIMAX)
* Customer Segmentation (RandomForest)
* Promotion Recommendations (LightGBM)
* Descriptive Insights (demographics, purchases, temporal patterns)
* Segmentation Insights (KMeans clusters)

## Tech Stack

* Backend: FastAPI, Python, LightGBM, RandomForest, SARIMAX, pandas, numpy
* Frontend: React (Netlify)
* Database: CSV
* Deployment: Render (backend), Netlify (frontend)

## Folder Structure

```
RetailIQ/
├── backend/
│   ├── main.py
│   ├── components/
│   │   ├── train_models.py
│   │   ├── stockpredict.py
│   │   ├── promotionpredict.py
│   │   └── segmentpredict.py
│   └── data.csv
├── frontend/
│   ├── src/components/
│   │   ├── SalesForecast.jsx
│   │   ├── PromotionPredict.jsx
│   │   ├── CustomerInsights.jsx
│   │   └── SegmentPredictor.jsx
│   └── package.json
|   |__ MainPage.jsx
├── README.md
├── Final_Report.pdf
└── Presentation.mp4
```

## API Endpoints

| Endpoint               | Method | Description                  |
| ---------------------- | ------ | ---------------------------- |
| `/`                    | GET    | Health check                 |
| `/insights`            | GET    | Returns metrics and insights |
| `/api/predict-segment` | POST   | Predicts customer segment    |
| `/forecast`            | POST   | Forecasts sales              |
| `/promotion`           | POST   | Promotion recommendation     |

## Setup & Run

1. Clone repo

```bash
git clone <repo-url>
cd RetailIQ/backend
```

2. Create virtualenv

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5. Frontend available via Netlify URL

## Authors / Team

* S.M.P.B.Senanayake, W.M.I.U. Weerakoon, A.G.I.A.Dissanayake , K.M A.U.Kulathunga


## Notes

* Models are loaded once at startup for memory efficiency.
* CSV dataset contains historical transactions.
* Can scale to database-backed solution for production.
