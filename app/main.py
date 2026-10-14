import os
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from bson.decimal128 import Decimal128
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import (
    DuplicateKeyError,
    PyMongoError,
    ServerSelectionTimeoutError,
)

load_dotenv()

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
PRESENT_STATUSES = ("PRESENT", "WFH", "ON_DUTY")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MILLISECOND_MIN = 100_000_000_000
MILLISECOND_MAX = 4_102_444_800_000
logger = logging.getLogger(__name__)
indexes_ready = False


def ensure_indexes() -> None:
    global indexes_ready
    db.employees.create_index("emp_code", unique=True, name="employee_code_unique")
    db.employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING)],
        name="department_joined_on",
    )
    db.employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)],
        name="department_employee_code",
    )
    db.employees.create_index("joined_on", name="employee_joined_on")

    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="employee_attendance_date_unique",
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)],
        name="attendance_date_employee_sort",
    )
    db.attendance_logs.create_index(
        [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)],
        name="attendance_status_date_employee",
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", DESCENDING), ("status", ASCENDING)],
        name="attendance_employee_month",
    )
    db.attendance_logs.create_index(
        [("date", ASCENDING), ("emp_code", ASCENDING)],
        name="attendance_date_employee",
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING), ("punch_out", ASCENDING)],
        name="attendance_open_punch",
    )
    indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ensure_indexes()
    except ServerSelectionTimeoutError:
        logger.exception("MongoDB unavailable during index initialization")
    try:
        yield
    finally:
        client.close()


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Validation, time conversion, and calculation helpers
# --------------------------------------------------------------------------- #
def valid_date_string(value: str, field: str = "date") -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise HTTPException(status_code=422, detail=f"{field} must be YYYY-MM-DD")
    return parsed


def valid_month_string(value: str) -> tuple[date, date]:
    try:
        parsed = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="month must be YYYY-MM") from exc
    if parsed.strftime("%Y-%m") != value:
        raise HTTPException(status_code=422, detail="month must be YYYY-MM")
    start = date(parsed.year, parsed.month, 1)
    next_month = date(parsed.year + (parsed.month == 12), parsed.month % 12 + 1, 1)
    return start, next_month


def dt_to_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    delta = value.astimezone(UTC) - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1000) + delta.microseconds // 1000


def ms_to_dt(value: int | None) -> datetime | None:
    if value is None:
        return None
    return (EPOCH + timedelta(milliseconds=value)).replace(microsecond=0)


def normalize_bson_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def truncate_to_second(value: datetime) -> datetime:
    return value.replace(microsecond=0)


def punch_datetime_to_ms(value: datetime | None) -> int | None:
    normalized = normalize_bson_datetime(value)
    if normalized is None:
        return None
    return dt_to_ms(truncate_to_second(normalized))


def parse_shift_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def round_half_up(value: Decimal, places: int) -> Decimal:
    quantum = Decimal(1).scaleb(-places)
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def numeric_to_float(value: Any) -> float:
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    return float(value)


def half_up_expression(expression: Any, places: int) -> dict[str, Any]:
    """Mongo expression for nonnegative half-up rounding (Mongo $round uses ties-to-even)."""
    scale = 10**places
    return {
        "$divide": [
            {
                "$trunc": {
                    "$add": [
                        {
                            "$multiply": [
                                {"$toDecimal": expression},
                                Decimal128(str(scale)),
                            ]
                        },
                        Decimal128("0.5"),
                    ]
                }
            },
            Decimal128(str(scale)),
        ]
    }


def weekday_count(start: date, end_exclusive: date) -> int:
    if end_exclusive <= start:
        return 0
    days = (end_exclusive - start).days
    full_weeks, remainder = divmod(days, 7)
    weekdays = full_weeks * 5
    weekdays += sum(1 for offset in range(remainder) if (start.weekday() + offset) % 7 < 5)
    return weekdays


def compute_record_date(employee: dict[str, Any], punch_in_ist: datetime) -> date:
    start_time = parse_shift_time(employee["shift_start"])
    end_time = parse_shift_time(employee["shift_end"])
    if end_time <= start_time and punch_in_ist.time() < end_time:
        return punch_in_ist.date() - timedelta(days=1)
    return punch_in_ist.date()


def shift_end_datetime(record_date: date, employee: dict[str, Any]) -> datetime:
    end_time = parse_shift_time(employee["shift_end"])
    start_time = parse_shift_time(employee["shift_start"])
    end_date = record_date + timedelta(days=1) if end_time <= start_time else record_date
    return datetime.combine(end_date, end_time, tzinfo=IST)


def calculate_late_minutes(record_date: date, punch_in_ist: datetime, shift_start: str) -> int:
    start = datetime.combine(record_date, parse_shift_time(shift_start), tzinfo=IST)
    elapsed_seconds = (punch_in_ist - start).total_seconds()
    if elapsed_seconds <= 600:
        return 0
    return int(elapsed_seconds // 60)


def calculate_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    seconds = Decimal(str((punch_out - punch_in).total_seconds()))
    return float(round_half_up(seconds / Decimal(3600), 2))


def calculate_overtime(punch_out: datetime, record_date: date, employee: dict[str, Any]) -> int:
    shift_end = shift_end_datetime(record_date, employee).astimezone(UTC)
    minutes = int((punch_out.astimezone(UTC) - shift_end).total_seconds() // 60)
    return minutes if minutes >= 30 else 0


def is_weekday_expression(field: str) -> dict[str, Any]:
    return {"$in": [{"$dayOfWeek": field}, [2, 3, 4, 5, 6]]}


def build_monthly_attendance_pipeline(emp_code: str, start: date, end: date) -> list[dict]:
    return [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": start.isoformat(), "$lt": end.isoformat()},
            }
        },
        {
            "$addFields": {
                "_calendar_day": {
                    "$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}
                },
                "_late": {"$ifNull": ["$late_minutes", 0]},
                "_overtime": {"$ifNull": ["$overtime_minutes", 0]},
                "_half_day": {"$ifNull": ["$half_day", False]},
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENT_STATUSES)]},
                                    is_weekday_expression("$_calendar_day"),
                                ]
                            },
                            {"$cond": ["$_half_day", 0.5, 1]},
                            0,
                        ]
                    }
                },
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {"$sum": {"$cond": [{"$gt": ["$_late", 0]}, 1, 0]}},
                "total_late_minutes": {"$sum": "$_late"},
                "total_overtime_minutes": {"$sum": "$_overtime"},
            }
        },
    ]


def build_department_summary_pipeline(
    start: date, end: date, department: str | None
) -> list[dict]:
    employee_match: dict[str, Any] = {"joined_on": {"$lt": end.isoformat()}}
    if department is not None:
        employee_match["department"] = department
    return [
        {"$match": employee_match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"employee_code": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$employee_code"]},
                                    {"$gte": ["$date", start.isoformat()]},
                                    {"$lt": ["$date", end.isoformat()]},
                                ]
                            }
                        }
                    },
                    {
                        "$addFields": {
                            "_calendar_day": {
                                "$dateFromString": {
                                    "dateString": "$date",
                                    "format": "%Y-%m-%d",
                                }
                            },
                            "_late": {"$ifNull": ["$late_minutes", 0]},
                            "_work_hours": {"$toDecimal": "$work_hours"},
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "present_days": {
                                "$sum": {
                                    "$cond": [
                                        {
                                            "$and": [
                                                {"$in": ["$status", list(PRESENT_STATUSES)]},
                                                is_weekday_expression("$_calendar_day"),
                                            ]
                                        },
                                        {
                                            "$cond": [
                                                {"$ifNull": ["$half_day", False]},
                                                0.5,
                                                1,
                                            ]
                                        },
                                        0,
                                    ]
                                }
                            },
                            "late_count": {
                                "$sum": {"$cond": [{"$gt": ["$_late", 0]}, 1, 0]}
                            },
                            "total_late_minutes": {"$sum": "$_late"},
                            "leave_count": {
                                "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                            },
                            "on_duty_count": {
                                "$sum": {
                                    "$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]
                                }
                            },
                            "work_hours_total": {
                                "$sum": {
                                    "$cond": [
                                        {
                                            "$and": [
                                                {"$in": ["$status", list(PRESENT_STATUSES)]},
                                                {"$ne": ["$_work_hours", None]},
                                            ]
                                        },
                                        "$_work_hours",
                                        0,
                                    ]
                                }
                            },
                            "work_hours_count": {
                                "$sum": {
                                    "$cond": [
                                        {
                                            "$and": [
                                                {"$in": ["$status", list(PRESENT_STATUSES)]},
                                                {"$ne": ["$_work_hours", None]},
                                            ]
                                        },
                                        1,
                                        0,
                                    ]
                                }
                            },
                        }
                    },
                ],
                "as": "attendance",
            }
        },
        {"$unwind": {"path": "$attendance", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": "$department",
                "employee_codes": {"$addToSet": "$emp_code"},
                "present_days": {"$sum": {"$ifNull": ["$attendance.present_days", 0]}},
                "late_count": {"$sum": {"$ifNull": ["$attendance.late_count", 0]}},
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$attendance.total_late_minutes", 0]}
                },
                "leave_count": {"$sum": {"$ifNull": ["$attendance.leave_count", 0]}},
                "on_duty_count": {"$sum": {"$ifNull": ["$attendance.on_duty_count", 0]}},
                "work_hours_total": {
                    "$sum": {"$ifNull": ["$attendance.work_hours_total", 0]}
                },
                "work_hours_count": {
                    "$sum": {"$ifNull": ["$attendance.work_hours_count", 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$employee_codes"},
                "present_days": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$work_hours_count", 0]},
                        half_up_expression(
                            {"$divide": ["$work_hours_total", "$work_hours_count"]}, 2
                        ),
                        None,
                    ]
                },
            }
        },
        {"$match": {"headcount": {"$gt": 0}}},
        {"$sort": {"department": ASCENDING}},
    ]


def build_leaderboard_pipeline(
    start: date, end: date, department: str | None, limit: int
) -> list[dict]:
    pipeline: list[dict] = [
        {
            "$match": {
                "date": {"$gte": start.isoformat(), "$lt": end.isoformat()},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "employee",
            }
        },
        {"$unwind": "$employee"},
    ]
    if department is not None:
        pipeline.append({"$match": {"employee.department": department}})
    pipeline.extend(
        [
            {
                "$setWindowFields": {
                    "sortBy": {"total_late_minutes": DESCENDING},
                    "output": {"rank": {"$rank": {}}},
                }
            },
            {"$match": {"rank": {"$lte": limit}}},
            {"$sort": {"total_late_minutes": DESCENDING, "_id": ASCENDING}},
            {
                "$project": {
                    "_id": 0,
                    "rank": 1,
                    "emp_code": "$_id",
                    "name": "$employee.name",
                    "department": "$employee.department",
                    "total_late_minutes": 1,
                    "late_count": 1,
                }
            },
        ]
    )
    return pipeline


def build_trend_pipeline(
    department: str, from_date: date, to_date: date
) -> list[dict]:
    start_dt = datetime.combine(from_date, time.min, tzinfo=UTC)
    end_exclusive = datetime.combine(to_date + timedelta(days=1), time.min, tzinfo=UTC)
    return [
        {"$match": {"department": department}},
        {
            "$addFields": {
                "_joined_date": {
                    "$dateFromString": {"dateString": "$joined_on", "format": "%Y-%m-%d"}
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "employees": {
                    "$push": {
                        "emp_code": "$emp_code",
                        "joined_date": "$_joined_date",
                    }
                },
            }
        },
        {
            "$project": {
                "employees": 1,
                "day": {
                    "$map": {
                        "input": {
                            "$range": [
                                0,
                                {
                                    "$dateDiff": {
                                        "startDate": start_dt,
                                        "endDate": end_exclusive,
                                        "unit": "day",
                                    }
                                },
                            ]
                        },
                        "as": "offset",
                        "in": {
                            "$dateAdd": {
                                "startDate": start_dt,
                                "unit": "day",
                                "amount": "$$offset",
                            }
                        },
                    }
                },
            }
        },
        {"$unwind": "$day"},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {
                    "day_string": {"$dateToString": {"format": "%Y-%m-%d", "date": "$day"}},
                    "employee_codes": {
                        "$map": {
                            "input": "$employees",
                            "as": "employee",
                            "in": "$$employee.emp_code",
                        }
                    },
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$date", "$$day_string"]},
                                    {"$in": ["$emp_code", "$$employee_codes"]},
                                ]
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "present_count": {
                                "$sum": {
                                    "$cond": [
                                        {"$in": ["$status", list(PRESENT_STATUSES)]},
                                        {
                                            "$cond": [
                                                {"$ifNull": ["$half_day", False]},
                                                0.5,
                                                1,
                                            ]
                                        },
                                        0,
                                    ]
                                }
                            },
                            "late_count": {
                                "$sum": {
                                    "$cond": [
                                        {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                                        1,
                                        0,
                                    ]
                                }
                            },
                        }
                    },
                ],
                "as": "day_attendance",
            }
        },
        {"$unwind": {"path": "$day_attendance", "preserveNullAndEmptyArrays": True}},
        {
            "$addFields": {
                "headcount": {
                    "$size": {
                        "$filter": {
                            "input": "$employees",
                            "as": "employee",
                            "cond": {"$lte": ["$$employee.joined_date", "$day"]},
                        }
                    }
                },
                "present_count": {"$ifNull": ["$day_attendance.present_count", 0]},
                "late_count": {"$ifNull": ["$day_attendance.late_count", 0]},
                "is_working_day": is_weekday_expression("$day"),
            }
        },
        {
            "$addFields": {
                "attendance_rate": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        half_up_expression(
                            {
                                "$divide": [
                                    {"$toDecimal": "$present_count"},
                                    {"$toDecimal": "$headcount"},
                                ]
                            },
                            4,
                        ),
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"day": ASCENDING},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$day"}},
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": ["$moving_avg_7d", None]},
                        None,
                        half_up_expression("$moving_avg_7d", 4),
                    ]
                },
            }
        },
        {"$sort": {"date": ASCENDING}},
    ]


def build_attendance_filter(
    emp_code: str | None,
    date_from: str | None,
    date_to: str | None,
    status: str | None,
) -> dict[str, Any]:
    query: dict[str, Any] = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    if date_from is not None or date_to is not None:
        query["date"] = {}
        if date_from is not None:
            query["date"]["$gte"] = date_from
        if date_to is not None:
            query["date"]["$lte"] = date_to
    if status is not None:
        query["status"] = status
    return query


def mongo_explain(collection_name: str, pipeline: list[dict]) -> dict[str, Any]:
    return db.command(
        "explain",
        {"aggregate": collection_name, "pipeline": pipeline, "cursor": {}},
        verbosity="executionStats",
    )


def serialize_employee(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "emp_code": document["emp_code"],
        "name": document["name"],
        "email": document["email"],
        "department": document["department"],
        "shift_start": document["shift_start"],
        "shift_end": document["shift_end"],
        "joined_on": document["joined_on"],
        "created_at": dt_to_ms(normalize_bson_datetime(document.get("created_at"))),
    }


def serialize_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    result = []
    for entry in history or []:
        changes: dict[str, Any] = {}
        for field, difference in (entry.get("changes") or {}).items():
            changes[field] = {
                key: (
                    punch_datetime_to_ms(value)
                    if isinstance(value, datetime)
                    else value
                )
                for key, value in difference.items()
            }
        result.append(
            {
                "at": dt_to_ms(normalize_bson_datetime(entry["at"])),
                "by": entry["by"],
                "reason": entry["reason"],
                "changes": changes,
            }
        )
    return result


def serialize_attendance(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "emp_code": document["emp_code"],
        "date": document["date"],
        "status": document["status"],
        "punch_in": punch_datetime_to_ms(document.get("punch_in")),
        "punch_out": punch_datetime_to_ms(document.get("punch_out")),
        "work_hours": document.get("work_hours"),
        "late_minutes": int(document.get("late_minutes") or 0),
        "overtime_minutes": int(document.get("overtime_minutes") or 0),
        "half_day": bool(document.get("half_day", False)),
        "history": serialize_history(document.get("history")),
    }


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class EmployeeIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, value: str) -> str:
        if len(value) != 10 or date.fromisoformat(value).isoformat() != value:
            raise ValueError("joined_on must be YYYY-MM-DD")
        return value

    @model_validator(mode="after")
    def shifts_must_differ(self) -> "EmployeeIn":
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


EpochMillis = Annotated[StrictInt, Field(ge=MILLISECOND_MIN, le=MILLISECOND_MAX)]


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: EpochMillis | None = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"

    @model_validator(mode="after")
    def reject_explicit_null(self) -> "PunchInRequest":
        if "punched_at" in self.model_fields_set and self.punched_at is None:
            raise ValueError("punched_at must be epoch milliseconds")
        return self


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: EpochMillis | None = None

    @model_validator(mode="after")
    def reject_explicit_null(self) -> "PunchOutRequest":
        if "punched_at" in self.model_fields_set and self.punched_at is None:
            raise ValueError("punched_at must be epoch milliseconds")
        return self


class RegularizeRequest(BaseModel):
    status: Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"] | None = None
    punch_in: EpochMillis | None = None
    punch_out: EpochMillis | None = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def reject_null_timestamps(self) -> "RegularizeRequest":
        if "status" in self.model_fields_set and self.status is None:
            raise ValueError("status must be a supported attendance status")
        for field_name in ("punch_in", "punch_out"):
            if field_name in self.model_fields_set and getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must be epoch milliseconds")
        return self


MonthParameter = Annotated[str, Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
DateParameter = Annotated[str, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")]


# --------------------------------------------------------------------------- #
# System and employee endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, str]:
    try:
        db.command("ping")
        if not indexes_ready:
            ensure_indexes()
    except PyMongoError as exc:
        logger.warning("MongoDB readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn) -> dict[str, Any]:
    document = body.model_dump()
    document["created_at"] = datetime.now(UTC)
    try:
        db.employees.insert_one(document)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="emp_code already exists") from exc
    return serialize_employee(document)


@app.get("/employees")
def list_employees(
    department: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    query = {"department": department} if department is not None else {}
    total = db.employees.count_documents(query)
    documents = db.employees.find(query, {"_id": 0}).sort("emp_code", ASCENDING)
    documents = documents.skip((page - 1) * page_size).limit(page_size)
    return {
        "items": [serialize_employee(document) for document in documents],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# --------------------------------------------------------------------------- #
# Attendance endpoints
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest) -> dict[str, Any]:
    employee = db.employees.find_one({"emp_code": body.emp_code})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")

    now = datetime.now(UTC) if body.punched_at is None else ms_to_dt(body.punched_at)
    punch_in_utc = now.replace(microsecond=0)
    punch_in_ist = punch_in_utc.astimezone(IST)
    attendance_date = compute_record_date(employee, punch_in_ist)
    document = {
        "emp_code": body.emp_code,
        "date": attendance_date.isoformat(),
        "status": body.status,
        "punch_in": punch_in_utc,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": calculate_late_minutes(
            attendance_date, punch_in_ist, employee["shift_start"]
        ),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(document)
    except DuplicateKeyError as exc:
        raise HTTPException(
            status_code=409, detail="already punched in for this date"
        ) from exc
    return serialize_attendance(document)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest) -> dict[str, Any]:
    employee = db.employees.find_one({"emp_code": body.emp_code})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")

    punch_out_utc = (
        datetime.now(UTC) if body.punched_at is None else ms_to_dt(body.punched_at)
    ).replace(microsecond=0)
    latest_possible_punch_in = (
        punch_out_utc + timedelta(seconds=1) - timedelta(milliseconds=1)
    )
    candidates = db.attendance_logs.find(
        {
            "emp_code": body.emp_code,
            "punch_in": {"$ne": None, "$lte": latest_possible_punch_in},
        }
    ).sort("punch_in", DESCENDING)
    record = None
    for candidate in candidates:
        candidate_punch_in = truncate_to_second(
            normalize_bson_datetime(candidate["punch_in"])
        )
        if candidate_punch_in <= punch_out_utc:
            record = candidate
            break
    if record is None:
        later_record = db.attendance_logs.find_one(
            {"emp_code": body.emp_code, "punch_in": {"$ne": None}},
            sort=[("punch_in", DESCENDING)],
        )
        if later_record is not None:
            later_punch_in = truncate_to_second(
                normalize_bson_datetime(later_record["punch_in"])
            )
            if punch_out_utc <= later_punch_in:
                raise HTTPException(
                    status_code=422, detail="punch_out must be after punch_in"
                )
        raise HTTPException(status_code=404, detail="no eligible punch-in found")
    if record.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="record already punched out")

    punch_in_utc = truncate_to_second(normalize_bson_datetime(record["punch_in"]))
    if punch_out_utc <= punch_in_utc:
        raise HTTPException(status_code=422, detail="punch_out must be after punch_in")
    if (punch_out_utc - punch_in_utc).total_seconds() > 24 * 3600:
        raise HTTPException(
            status_code=422, detail="punch_out must be within 24 hours of punch_in"
        )

    attendance_date = valid_date_string(record["date"])
    work_hours = calculate_work_hours(punch_in_utc, punch_out_utc)
    update = {
        "punch_in": punch_in_utc,
        "punch_out": punch_out_utc,
        "work_hours": work_hours,
        "overtime_minutes": calculate_overtime(
            punch_out_utc, attendance_date, employee
        ),
        "half_day": work_hours < 4.50,
    }
    result = db.attendance_logs.update_one(
        {
            "_id": record["_id"],
            "punch_in": record["punch_in"],
            "punch_out": None,
        },
        {"$set": update},
    )
    if result.modified_count != 1:
        raise HTTPException(status_code=409, detail="record already punched out")
    record.update(update)
    return serialize_attendance(record)


@app.get("/attendance")
def list_attendance(
    emp_code: str | None = None,
    date_from: DateParameter | None = None,
    date_to: DateParameter | None = None,
    status: Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"] | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    if date_from is not None:
        valid_date_string(date_from, "date_from")
    if date_to is not None:
        valid_date_string(date_to, "date_to")
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must be <= date_to")
    query = build_attendance_filter(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(query)
    documents = db.attendance_logs.find(query).sort(
        [("date", DESCENDING), ("emp_code", ASCENDING)]
    )
    documents = documents.skip((page - 1) * page_size).limit(page_size)
    return {
        "items": [serialize_attendance(document) for document in documents],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    emp_code: str,
    body: RegularizeRequest,
    date: str = Path(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> dict[str, Any]:
    record_date = valid_date_string(date)
    employee = db.employees.find_one({"emp_code": emp_code})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")
    record = db.attendance_logs.find_one(
        {"emp_code": emp_code, "date": record_date.isoformat()}
    )
    if record is None:
        raise HTTPException(status_code=404, detail="attendance record not found")

    supplied = body.model_fields_set
    requested_fields = supplied.intersection({"status", "punch_in", "punch_out"})
    if not requested_fields:
        raise HTTPException(status_code=422, detail="request changes nothing")
    final = dict(record)
    final_status = body.status if "status" in supplied else record["status"]
    final["status"] = final_status

    if final_status in {"ABSENT", "LEAVE"}:
        if supplied.intersection({"punch_in", "punch_out"}):
            raise HTTPException(
                status_code=422,
                detail="punch times cannot be supplied with ABSENT or LEAVE",
            )
        final.update(
            {
                "punch_in": None,
                "punch_out": None,
                "work_hours": None,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
            }
        )
    else:
        if "punch_in" in supplied:
            final["punch_in"] = ms_to_dt(body.punch_in)
        if "punch_out" in supplied:
            final["punch_out"] = ms_to_dt(body.punch_out)
        punch_in_utc = normalize_bson_datetime(final.get("punch_in"))
        if punch_in_utc is None:
            raise HTTPException(
                status_code=422, detail="presence status requires punch_in"
            )
        punch_in_utc = punch_in_utc.replace(microsecond=0)
        punch_in_ist = punch_in_utc.astimezone(IST)
        if compute_record_date(employee, punch_in_ist) != record_date:
            raise HTTPException(
                status_code=422,
                detail="punch_in must remain on the record's attendance date",
            )
        final["punch_in"] = punch_in_utc
        final["late_minutes"] = calculate_late_minutes(
            record_date, punch_in_ist, employee["shift_start"]
        )
        punch_out_value = normalize_bson_datetime(final.get("punch_out"))
        if punch_out_value is None:
            final.update(
                {"punch_out": None, "work_hours": None, "overtime_minutes": 0, "half_day": False}
            )
        else:
            punch_out_utc = punch_out_value.replace(microsecond=0)
            if punch_out_utc <= punch_in_utc:
                raise HTTPException(
                    status_code=422, detail="punch_out must be after punch_in"
                )
            if (punch_out_utc - punch_in_utc).total_seconds() > 24 * 3600:
                raise HTTPException(
                    status_code=422,
                    detail="punch_out must be within 24 hours of punch_in",
                )
            work_hours = calculate_work_hours(punch_in_utc, punch_out_utc)
            final.update(
                {
                    "punch_out": punch_out_utc,
                    "work_hours": work_hours,
                    "overtime_minutes": calculate_overtime(
                        punch_out_utc, record_date, employee
                    ),
                    "half_day": work_hours < 4.50,
                }
            )

    changed_fields = (
        "status",
        "punch_in",
        "punch_out",
        "work_hours",
        "late_minutes",
        "overtime_minutes",
        "half_day",
    )
    data_model_defaults = {
        "punch_in": None,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": 0,
        "overtime_minutes": 0,
        "half_day": False,
    }
    changes = {
        field: {
            "from": record.get(field, data_model_defaults.get(field)),
            "to": final.get(field, data_model_defaults.get(field)),
        }
        for field in changed_fields
        if record.get(field, data_model_defaults.get(field))
        != final.get(field, data_model_defaults.get(field))
    }
    if not changes:
        raise HTTPException(status_code=422, detail="request changes nothing")

    history_entry = {
        "at": datetime.now(UTC),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }
    prior_history = record.get("history", [])
    final_history = list(prior_history) + [history_entry]

    snapshot: dict[str, Any] = {"_id": record["_id"]}
    for field in changed_fields:
        if field in record:
            snapshot[field] = record[field]
        else:
            snapshot[field] = {"$exists": False}
    if "history" in record:
        snapshot["history"] = prior_history
    else:
        snapshot["history"] = {"$exists": False}
    update_fields = {field: final.get(field) for field in changed_fields}
    update_fields["history"] = final_history
    result = db.attendance_logs.update_one(snapshot, {"$set": update_fields})
    if result.modified_count != 1:
        raise HTTPException(status_code=409, detail="record changed concurrently")
    record.update(update_fields)
    return serialize_attendance(record)


# --------------------------------------------------------------------------- #
# Analytics endpoints
# --------------------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: MonthParameter) -> dict[str, Any]:
    start, end = valid_month_string(month)
    employee = db.employees.find_one({"emp_code": emp_code})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")
    joined = valid_date_string(employee["joined_on"], "joined_on")
    working_start = max(start, joined)
    working_days = weekday_count(working_start, end)
    rows = list(
        db.attendance_logs.aggregate(
            build_monthly_attendance_pipeline(emp_code, start, end)
        )
    )
    metrics = rows[0] if rows else {}
    present_days = float(metrics.get("present_days", 0))
    attendance_pct = None
    if working_days:
        attendance_pct = float(
            round_half_up(
                Decimal(str(present_days)) / Decimal(working_days) * Decimal(100), 4
            )
        )
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": int(metrics.get("leave_days", 0)),
        "late_count": int(metrics.get("late_count", 0)),
        "total_late_minutes": int(metrics.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(metrics.get("total_overtime_minutes", 0)),
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: MonthParameter, department: str | None = None
) -> dict[str, Any]:
    start, end = valid_month_string(month)
    items = list(
        db.employees.aggregate(
            build_department_summary_pipeline(start, end, department)
        )
    )
    for item in items:
        if item.get("avg_work_hours") is not None:
            item["avg_work_hours"] = numeric_to_float(item["avg_work_hours"])
        item["present_days"] = float(item["present_days"])
        for count_field in (
            "headcount",
            "late_count",
            "total_late_minutes",
            "leave_count",
            "on_duty_count",
        ):
            item[count_field] = int(item[count_field])
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: MonthParameter,
    limit: int = Query(10, ge=1, le=50),
    department: str | None = None,
) -> dict[str, Any]:
    start, end = valid_month_string(month)
    items = list(
        db.attendance_logs.aggregate(
            build_leaderboard_pipeline(start, end, department, limit)
        )
    )
    for item in items:
        item["rank"] = int(item["rank"])
        item["total_late_minutes"] = int(item["total_late_minutes"])
        item["late_count"] = int(item["late_count"])
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_: str = Query(..., alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> dict[str, Any]:
    start = valid_date_string(from_, "from")
    end = valid_date_string(to, "to")
    if end < start:
        raise HTTPException(status_code=422, detail="to must be >= from")
    if (end - start).days + 1 > 92:
        raise HTTPException(status_code=422, detail="range cannot exceed 92 days")
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(status_code=404, detail="department not found")
    items = list(
        db.employees.aggregate(build_trend_pipeline(department, start, end))
    )
    for item in items:
        item["present_count"] = float(item["present_count"])
        item["headcount"] = int(item["headcount"])
        item["late_count"] = int(item["late_count"])
        if item.get("attendance_rate") is not None:
            item["attendance_rate"] = numeric_to_float(item["attendance_rate"])
        if item.get("moving_avg_7d") is not None:
            item["moving_avg_7d"] = numeric_to_float(item["moving_avg_7d"])
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Explain endpoint
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal[
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    ],
    emp_code: str | None = None,
    month: str | None = None,
    department: str | None = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: str | None = None,
    date_to: str | None = None,
    status: Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"] | None = None,
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    if endpoint == "attendance_list":
        if date_from is not None:
            valid_date_string(date_from, "date_from")
        if date_to is not None:
            valid_date_string(date_to, "date_to")
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from must be <= date_to")
        query = build_attendance_filter(emp_code, date_from, date_to, status)
        plan = db.command(
            "explain",
            {
                "find": "attendance_logs",
                "filter": query,
                "sort": {"date": DESCENDING, "emp_code": ASCENDING},
                "skip": (page - 1) * page_size,
                "limit": page_size,
            },
            verbosity="executionStats",
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": plan}

    if endpoint in {"employee_monthly", "department_summary", "late_leaderboard"}:
        if month is None:
            raise HTTPException(status_code=422, detail="month is required")
        start, end = valid_month_string(month)

    if endpoint == "employee_monthly":
        if emp_code is None:
            raise HTTPException(status_code=422, detail="emp_code is required")
        pipeline = build_monthly_attendance_pipeline(emp_code, start, end)
        collection_name = "attendance_logs"
    elif endpoint == "department_summary":
        pipeline = build_department_summary_pipeline(start, end, department)
        collection_name = "employees"
    elif endpoint == "late_leaderboard":
        pipeline = build_leaderboard_pipeline(start, end, department, limit)
        collection_name = "attendance_logs"
    else:
        if department is None or from_ is None or to is None:
            raise HTTPException(
                status_code=422, detail="department, from and to are required"
            )
        start_date = valid_date_string(from_, "from")
        end_date = valid_date_string(to, "to")
        if end_date < start_date or (end_date - start_date).days + 1 > 92:
            raise HTTPException(status_code=422, detail="invalid trend date range")
        pipeline = build_trend_pipeline(department, start_date, end_date)
        collection_name = "employees"

    return {
        "endpoint": endpoint,
        "collection": collection_name,
        "explain": mongo_explain(collection_name, pipeline),
    }
