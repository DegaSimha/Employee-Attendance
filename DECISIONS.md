# Design decisions

1. **Indexes.** Unique indexes on `employees.emp_code` and `attendance_logs(emp_code, date)` enforce business keys and support lookups. Department/join-date indexes serve headcount queries; attendance compounds support date sorting/filtering, monthly metrics, and punch-out. I rejected separate single-field indexes where an existing compound prefix serves the same access pattern.

2. **Punch-in race.** Concurrent requests may both reach insert, but the unique employee/date index permits one. MongoDB raises `DuplicateKeyError` for the loser; the API returns 409. The preliminary read is not the guarantee.

3. **Ties.** `$rank` uses late-minute totals, so ties share ranks and subsequent rank numbers are skipped. Output is then ordered by total descending and employee code ascending. Filtering `rank <= limit` retains everyone tied at the cutoff.

4. **Headcount.** The summary starts from employees joined by month end and looks up their attendance. Employees without logs remain in the pipeline and count once toward headcount.

5. **100x data.** I would benchmark explain plans and workload, then consider materialized monthly summaries with a correction-aware rebuild process. Raw logs and audit history would remain authoritative.
