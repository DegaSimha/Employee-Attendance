import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import mongomock
from bson.decimal128 import Decimal128
from fastapi.testclient import TestClient
from pymongo.errors import ServerSelectionTimeoutError

import app.main as main

IST = timezone(timedelta(hours=5, minutes=30))
main.client.close()


def employee(
    emp_code: str = "EMP0001",
    *,
    department: str = "Engineering",
    shift_start: str = "09:30",
    shift_end: str = "18:30",
    joined_on: str = "2026-01-05",
) -> dict:
    return {
        "emp_code": emp_code,
        "name": f"Employee {emp_code}",
        "email": f"{emp_code.lower()}@example.com",
        "department": department,
        "shift_start": shift_start,
        "shift_end": shift_end,
        "joined_on": joined_on,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }


def epoch_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


class AttendanceApiTests(unittest.TestCase):
    def setUp(self):
        self.mongo = mongomock.MongoClient(tz_aware=True)
        self.db = self.mongo["attendance_test"]
        main.db = self.db
        main.client = self.mongo
        self.client = TestClient(main.app)
        self.client.__enter__()
        main.ensure_indexes()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_ist_attendance_day_for_overnight_shift(self):
        emp = employee(shift_start="22:00", shift_end="06:00")
        early_morning = datetime(2026, 7, 7, 5, 59, tzinfo=IST)
        after_shift_end = datetime(2026, 7, 7, 6, 0, tzinfo=IST)

        self.assertEqual(main.compute_record_date(emp, early_morning), date(2026, 7, 6))
        self.assertEqual(main.compute_record_date(emp, after_shift_end), date(2026, 7, 7))

    def test_late_grace_boundary_and_flooring(self):
        self.assertEqual(
            main.calculate_late_minutes(
                date(2026, 7, 6), datetime(2026, 7, 6, 9, 40, tzinfo=IST), "09:30"
            ),
            0,
        )
        self.assertEqual(
            main.calculate_late_minutes(
                date(2026, 7, 6),
                datetime(2026, 7, 6, 9, 40, 1, tzinfo=IST),
                "09:30",
            ),
            10,
        )
        self.assertEqual(
            main.calculate_late_minutes(
                date(2026, 7, 6),
                datetime(2026, 7, 6, 10, 15, 59, tzinfo=IST),
                "09:30",
            ),
            45,
        )

    def test_overtime_threshold_and_overnight_end(self):
        emp = employee(shift_start="22:00", shift_end="06:00")
        record_date = date(2026, 7, 6)
        self.assertEqual(
            main.calculate_overtime(
                datetime(2026, 7, 7, 6, 29, 59, tzinfo=IST), record_date, emp
            ),
            0,
        )
        self.assertEqual(
            main.calculate_overtime(
                datetime(2026, 7, 7, 6, 30, tzinfo=IST), record_date, emp
            ),
            30,
        )

    def test_work_hours_use_decimal_half_up(self):
        start = datetime(2026, 7, 6, tzinfo=timezone.utc)
        self.assertEqual(
            main.calculate_work_hours(start, start + timedelta(seconds=16_218)), 4.51
        )
        self.assertEqual(
            main.calculate_work_hours(start, start + timedelta(seconds=16_182)), 4.5
        )
        self.assertEqual(
            main.round_half_up(Decimal("2.34505"), 4), Decimal("2.3451")
        )
        self.assertEqual(main.numeric_to_float(Decimal128("2.345")), 2.345)

    def test_employee_validation_and_filtered_pagination(self):
        self.db.employees.insert_many(
            [
                employee("EMP0001", department="Engineering"),
                employee("EMP0002", department="Sales"),
                employee("EMP0003", department="Engineering"),
            ]
        )
        response = self.client.get(
            "/employees?department=Engineering&page=2&page_size=1"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 2)
        self.assertEqual(
            [row["emp_code"] for row in response.json()["items"]], ["EMP0003"]
        )
        self.assertNotIn("_id", response.json()["items"][0])

        payload = employee("EMP1000")
        payload.pop("created_at")
        self.assertEqual(self.client.post("/employees", json=payload).status_code, 201)
        self.assertEqual(self.client.post("/employees", json=payload).status_code, 409)
        payload["emp_code"] = "bad"
        self.assertEqual(self.client.post("/employees", json=payload).status_code, 422)

    def test_punch_in_strict_timestamp_and_duplicate_record(self):
        self.db.employees.insert_one(employee())
        valid = {
            "emp_code": "EMP0001",
            "punched_at": epoch_ms("2026-07-06T09:40:01+05:30"),
        }
        response = self.client.post("/attendance/punch-in", json=valid)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["date"], "2026-07-06")
        self.assertEqual(response.json()["late_minutes"], 10)
        self.assertEqual(response.json()["history"], [])
        self.assertEqual(self.client.post("/attendance/punch-in", json=valid).status_code, 409)
        for invalid_value in (1783321200, 1783321200000.0, "1783321200000"):
            with self.subTest(timestamp=invalid_value):
                invalid = {**valid, "punched_at": invalid_value}
                self.assertEqual(
                    self.client.post("/attendance/punch-in", json=invalid).status_code,
                    422,
                )

    def test_concurrent_punch_ins_are_guarded_by_unique_index(self):
        self.db.employees.insert_one(employee())
        payload = {
            "emp_code": "EMP0001",
            "punched_at": epoch_ms("2026-07-06T09:40:01+05:30"),
        }
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(
                pool.map(
                    lambda _: self.client.post("/attendance/punch-in", json=payload),
                    range(6),
                )
            )
        statuses = [response.status_code for response in responses]
        self.assertEqual(statuses.count(201), 1, statuses)
        self.assertEqual(statuses.count(409), 5, statuses)

    def test_punch_out_updates_atomically_and_second_request_conflicts(self):
        self.db.employees.insert_one(employee())
        self.db.attendance_logs.insert_one(
            {
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": datetime(
                    2026, 7, 6, 4, 0, 0, 900000, tzinfo=timezone.utc
                ),
                "punch_out": None,
                "work_hours": None,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [],
            }
        )
        payload = {
            "emp_code": "EMP0001",
            "punched_at": epoch_ms("2026-07-06T13:00:00+00:00"),
        }
        response = self.client.post("/attendance/punch-out", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["work_hours"], 9.0)
        self.assertEqual(
            response.json()["punch_in"],
            epoch_ms("2026-07-06T04:00:00+00:00"),
        )
        self.assertEqual(response.json()["history"], [])
        self.assertEqual(self.client.post("/attendance/punch-out", json=payload).status_code, 409)

    def test_punch_out_before_punch_in_is_validation_error(self):
        self.db.employees.insert_one(employee())
        self.db.attendance_logs.insert_one(
            {
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": datetime(2026, 7, 6, 4, 0, tzinfo=timezone.utc),
                "punch_out": None,
                "work_hours": None,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [],
            }
        )
        response = self.client.post(
            "/attendance/punch-out",
            json={
                "emp_code": "EMP0001",
                "punched_at": epoch_ms("2026-07-06T09:29:59+05:30"),
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_regularization_clears_absent_times_and_audits_real_changes(self):
        self.db.employees.insert_one(employee())
        self.db.attendance_logs.insert_one(
            {
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": datetime(2026, 7, 6, 4, 0, tzinfo=timezone.utc),
                "punch_out": datetime(2026, 7, 6, 13, 0, tzinfo=timezone.utc),
                "work_hours": 9.0,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
                "history": [],
            }
        )
        response = self.client.patch(
            "/attendance/EMP0001/2026-07-06",
            json={
                "status": "ABSENT",
                "reason": "approved correction",
                "regularized_by": "hr",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        corrected = response.json()
        self.assertIsNone(corrected["punch_in"])
        self.assertIsNone(corrected["punch_out"])
        self.assertIsNone(corrected["work_hours"])
        self.assertEqual(len(corrected["history"]), 1)
        self.assertEqual(
            corrected["history"][0]["changes"]["status"],
            {"from": "PRESENT", "to": "ABSENT"},
        )
        stored = self.db.attendance_logs.find_one({"emp_code": "EMP0001"})
        self.assertIsNotNone(stored["history"][0]["at"].tzinfo)
        self.assertIsInstance(stored["history"][0]["changes"]["punch_in"]["from"], datetime)

        unchanged = self.client.patch(
            "/attendance/EMP0001/2026-07-06",
            json={
                "status": "ABSENT",
                "reason": "same status",
                "regularized_by": "hr",
            },
        )
        self.assertEqual(unchanged.status_code, 422)
        supplied_times = self.client.patch(
            "/attendance/EMP0001/2026-07-06",
            json={
                "status": "LEAVE",
                "punch_in": epoch_ms("2026-07-06T09:00:00+05:30"),
                "reason": "invalid request",
                "regularized_by": "hr",
            },
        )
        self.assertEqual(supplied_times.status_code, 422)

    def test_missing_seed_defaults_do_not_create_false_audit_diffs(self):
        self.db.employees.insert_one(employee())
        self.db.attendance_logs.insert_one(
            {
                "emp_code": "EMP0001",
                "date": "2026-07-06",
                "status": "LEAVE",
                "punch_in": None,
                "punch_out": None,
                "work_hours": None,
            }
        )
        response = self.client.patch(
            "/attendance/EMP0001/2026-07-06",
            json={
                "status": "ABSENT",
                "reason": "correct record",
                "regularized_by": "hr",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["history"][0]["changes"],
            {"status": {"from": "LEAVE", "to": "ABSENT"}},
        )

    def test_attendance_pagination_and_invalid_date_range(self):
        self.db.employees.insert_one(employee())
        for day in ("2026-07-06", "2026-07-07", "2026-07-08"):
            self.db.attendance_logs.insert_one(
                {
                    "emp_code": "EMP0001",
                    "date": day,
                    "status": "ABSENT",
                    "punch_in": None,
                    "punch_out": None,
                    "work_hours": None,
                    "late_minutes": 0,
                    "overtime_minutes": 0,
                    "half_day": False,
                    "history": [],
                }
            )
        response = self.client.get(
            "/attendance?page=2&page_size=1&date_from=2026-07-06"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 3)
        self.assertEqual(response.json()["items"][0]["date"], "2026-07-07")
        self.assertEqual(
            self.client.get(
                "/attendance?date_from=2026-07-08&date_to=2026-07-06"
            ).status_code,
            422,
        )

    def test_analytics_use_server_side_gap_and_window_pipeline_builders(self):
        start, end = main.valid_month_string("2026-07")
        monthly = main.build_monthly_attendance_pipeline("EMP0001", start, end)
        summary = main.build_department_summary_pipeline(start, end, "Engineering")
        leaderboard = main.build_leaderboard_pipeline(start, end, None, 2)
        trend = main.build_trend_pipeline(
            "Engineering", date(2026, 7, 6), date(2026, 7, 8)
        )

        self.assertEqual(monthly[0]["$match"]["emp_code"], "EMP0001")
        self.assertEqual(summary[0]["$match"]["department"], "Engineering")
        self.assertTrue(any("$lookup" in stage for stage in summary))
        self.assertTrue(any("$rank" in str(stage) for stage in leaderboard))
        self.assertTrue(any("$range" in str(stage) for stage in trend))
        self.assertTrue(any("$setWindowFields" in stage for stage in trend))
        map_stage = next(stage for stage in trend if "$project" in stage and "day" in stage["$project"])
        self.assertIn("input", map_stage["$project"]["day"]["$map"])
        self.assertIn("$range", map_stage["$project"]["day"]["$map"]["input"])

    def test_working_days_count_mid_month_joiner(self):
        self.assertEqual(
            main.weekday_count(date(2026, 7, 1), date(2026, 8, 1)),
            23,
        )
        self.assertEqual(
            main.weekday_count(date(2026, 7, 15), date(2026, 8, 1)),
            13,
        )
        self.assertEqual(
            main.weekday_count(date(2026, 8, 1), date(2026, 8, 1)),
            0,
        )

    def test_trend_date_range_validation_and_contract_query_names(self):
        response = self.client.get(
            "/analytics/departments/Engineering/trend?from=2026-07-01&to=2026-10-01"
        )
        self.assertEqual(response.status_code, 422)
        response = self.client.get(
            "/analytics/departments/Engineering/trend?from=2026-07-08&to=2026-07-06"
        )
        self.assertEqual(response.status_code, 422)
        parameters = main.app.openapi()["paths"][
            "/analytics/departments/{department}/trend"
        ]["get"]["parameters"]
        self.assertTrue({"department", "from", "to"}.issubset(
            {parameter["name"] for parameter in parameters}
        ))

    def test_health_reports_unavailable_database(self):
        def fail_ping(command):
            raise ServerSelectionTimeoutError("unavailable")

        self.db.command = fail_ping
        self.assertEqual(self.client.get("/health").status_code, 503)

    def test_health_and_docs_when_database_is_available(self):
        self.db.command = lambda command: {"ok": 1}
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/docs").status_code, 200)

    def test_explain_endpoint_validates_required_parameters(self):
        self.assertEqual(
            self.client.get("/admin/explain/employee_monthly").status_code, 422
        )
        self.assertEqual(
            self.client.get("/admin/explain/department_trend").status_code, 422
        )


if __name__ == "__main__":
    unittest.main()
