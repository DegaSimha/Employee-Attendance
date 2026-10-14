# Employee Attendance & Analytics API

Implementation of the HROne assignment contract using FastAPI, Pydantic v2, PyMongo, and MongoDB 6.0+.

## Setup and run

Use Python 3.11 or newer and a MongoDB 6.0+ database. Copy `.env.example` to `.env` for local defaults, or set `MONGO_URI` and `MONGO_DB` in your environment; environment variables take precedence. `.env` is ignored by Git and must not be submitted.

```bash
python -m venv .venv
.venv\Scripts\activate
Copy-Item .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Create employee records through `POST /employees` or load the candidate kit samples locally if you have the supplied kit files. Do not run the kit's seed script against a database containing data you need: it replaces both collections. Indexes are created idempotently when the app starts. Interactive API documentation is at `/docs`.

## Tests

Run `python -m pytest` (or `python -m unittest discover -s tests -v`). The `TestClient` tests use `mongomock` for isolated CRUD and request checks and inspect the MongoDB pipeline builders structurally. Aggregation stages introduced in MongoDB 5/6, actual query plans, and `IXSCAN`/`COLLSCAN` behavior must also be verified against a real MongoDB 6.0+ instance; the in-memory mock does not implement those stages.

## Implemented endpoints

- `GET /health`
- `POST /employees`, `GET /employees`
- `POST /attendance/punch-in`, `POST /attendance/punch-out`, `GET /attendance`
- `PATCH /attendance/{emp_code}/{date}`
- `GET /analytics/employees/{emp_code}/monthly`
- `GET /analytics/departments/summary`
- `GET /analytics/leaderboard/late`
- `GET /analytics/departments/{department}/trend`
- `GET /admin/explain/{endpoint}`

The application keeps all runtime logic in `app/main.py`. No Dockerfile or credentials are included.
