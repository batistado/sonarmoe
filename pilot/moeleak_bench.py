"""MoELeak-Bench (Phase 0 Pilot): 100 Sensitive Victim Queries & MCP Tool Schemas.

Covers 5 sensitive enterprise / agentic categories (20 distinct prompts each):
  0: Clinical/Medical        — EHR notes, oncology pathology, ICU dosing, genetic panels
  1: SQL/Database Tool       — MCP database tool schemas & analytical SQL queries
  2: Stripe/Payment Tool     — MCP Stripe/Plaid payment tool invocations & webhook payloads
  3: Crypto/Auth Code        — Cryptographic primitives, JWT/OAuth2/mTLS implementation code
  4: Legal Contract          — M&A indemnification, GDPR DPAs, Series B covenants, IP clauses

Also includes an independent general-domain assistant evaluation corpus for measuring
exact next-token Perplexity (PPL) and ΔPPL under OblivDispatch (ε, δ)-DP routing.
"""

from __future__ import annotations

CATEGORIES: list[str] = [
    "Clinical/Medical",
    "SQL/Database Tool",
    "Stripe/Payment Tool",
    "Crypto/Auth Code",
    "Legal Contract",
]

MOELEAK_BENCH_PROMPTS: list[dict[str, object]] = [
    # =========================================================================
    # 0. Clinical/Medical (20 prompts)
    # =========================================================================
    {
        "id": "med_00",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Patient 64F with stage IIIB EGFR exon 19 deletion non-small cell lung "
            "adenocarcinoma presenting with acute dyspnea, bilateral pleural effusions, "
            "elevated serum troponin-I 0.42 ng/mL, and QTc prolongation to 512 ms on "
            "osimertinib 80 mg daily. Recommend oncology cardiology management and dose hold."
        ),
    },
    {
        "id": "med_01",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "ICU nephrology consult: 58M septic shock secondary to cholangitis on "
            "norepinephrine 0.24 mcg/kg/min and vasopressin, oliguric acute kidney injury "
            "with serum potassium 6.4 mEq/L, arterial pH 7.19, lactate 5.8 mmol/L. "
            "Calculate continuous venovenous hemodiafiltration effluent dose and citrate anticoagulation."
        ),
    },
    {
        "id": "med_02",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Germline pharmacogenomics report: heterozygous DPYD*2A splice-site variant "
            "(c.1905+1G>A) and UGT1A1*28/*28 homozygosity in a 51M initiating FOLFOXIRI "
            "for metastatic colorectal carcinoma with hepatic metastases. Specify CPIC "
            "fluoropyrimidine and irinotecan dose reductions to prevent febrile neutropenia."
        ),
    },
    {
        "id": "med_03",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Pediatric neurology EEG and CSF summary: 9-year-old girl with subacute "
            "orofacial dyskinesias, refractory focal motor seizures, extreme delta brush "
            "pattern on continuous video EEG, and positive CSF anti-NMDA receptor IgG "
            "titers 1:64. Outline intravenous methylprednisolone, IVIG, and rituximab escalation."
        ),
    },
    {
        "id": "med_04",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Hematology bone marrow biopsy: 71M pancytopenia, circulating blasts 14%, "
            "myelodysplasia-related cytogenetics del(5q) and TP53 biallelic loss-of-function "
            "mutation VAF 68%, hypercellular marrow with dysplastic megakaryocytes. Assess "
            "hypomethylating agent azacitidine plus venetoclax induction eligibility."
        ),
    },
    {
        "id": "med_05",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Endocrinology inpatient crisis: 34F Graves thyrotoxicosis triggered by iodinated "
            "CT contrast, Burch-Wartofsky score 55, atrial fibrillation with rapid ventricular "
            "response 168 bpm, temperature 39.6 C, hepatic transaminitis. Sequence propylthiouracil, "
            "esmolol infusion, saturated potassium iodide, and hydrocortisone dosing."
        ),
    },
    {
        "id": "med_06",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Rheumatology-pulmonology board: 49F anti-MDA5 positive clinically amyopathic "
            "dermatomyositis with rapidly progressive interstitial lung disease, high-resolution "
            "chest CT ground-glass opacities and subpleural consolidation, PaO2/FiO2 175, "
            "ferritin 2450 ng/mL. Evaluate triple immunosuppressive therapy with tacrolimus."
        ),
    },
    {
        "id": "med_07",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Infectious disease transplant note: D+/R- cytomegalovirus serostatus bilateral "
            "lung transplant recipient day +118 with ganciclovir-resistant CMV viremia "
            "185,000 IU/mL harboring UL97 M460V mutation and pancytopenia on mycophenolate. "
            "Transition to maribavir versus foscarnet with renal electrolyte monitoring."
        ),
    },
    {
        "id": "med_08",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Obstetric maternal-fetal medicine: 29F G2P1 at 31 weeks 4 days gestation with "
            "HELLP syndrome, blood pressure 176/114 mmHg, platelet count 42,000/uL, AST 380 U/L, "
            "LDH 920 U/L, absent end-diastolic umbilical artery Doppler flow. Coordinate IV "
            "magnesium sulfate neuroprophylaxis, labetalol, betamethasone, and cesarean delivery."
        ),
    },
    {
        "id": "med_09",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Cardiology structural heart catheterization: 79M severe symptomatic paradoxical "
            "low-flow low-gradient aortic stenosis, aortic valve area 0.68 cm2, stroke volume "
            "index 28 mL/m2, LVEF 52%, wild-type transthyretin cardiac amyloidosis on pyrophosphate "
            "scintigraphy. Assess transfemoral TAVR sizing and conduction block pacemaker risk."
        ),
    },
    {
        "id": "med_10",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Hepatology decompensated cirrhosis admission: 56M alcohol-associated cirrhosis "
            "MELD-Na 29, tense ascites with spontaneous bacterial peritonitis ascitic PMN "
            "740 cells/mm3, hepatorenal syndrome-AKI stage 2, creatinine 2.9 mg/dL. Initiate "
            "ceftriaxone, weight-based intravenous albumin, and terlipressin infusion protocol."
        ),
    },
    {
        "id": "med_11",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Inpatient psychiatry clozapine titration: 41M treatment-resistant paranoid "
            "schizophrenia following failed olanzapine and risperidone trials, presenting "
            "on clozapine day 16 with new fever 38.7 C, sinus tachycardia 124 bpm, CRP 94 mg/L, "
            "high-sensitivity troponin elevation, and ANC 4,100/uL. Rule out clozapine myocarditis."
        ),
    },
    {
        "id": "med_12",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Neuro-oncology tumor board: 45M IDH-wildtype glioblastoma MGMT promoter methylated "
            "status post gross total resection of right temporal mass, post-operative MRI showing "
            "pseudoprogression versus early recurrence after Stupp chemoradiation with temozolomide, "
            "steroid-dependent vasogenic cerebral edema on dexamethasone 8 mg daily."
        ),
    },
    {
        "id": "med_13",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Toxicology emergency consult: 26F intentional ingestion of sustained-release "
            "verapamil 2400 mg presenting with refractory bradycardia 38 bpm, MAP 48 mmHg, "
            "hyperglycemia 310 mg/dL, and metabolic acidosis despite calcium gluconate. "
            "Initiate high-dose hyperinsulinemic euglycemic therapy and venoarterial ECMO."
        ),
    },
    {
        "id": "med_14",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Cellular immunotherapy CRS/ICANS note: 62F relapsed diffuse large B-cell lymphoma "
            "day +5 post axicabtagene ciloleucel CAR-T infusion, ICE encephalopathy score 4/10, "
            "expressive aphasia, persistent fever 39.4 C, hypotension responsive to fluid bolus. "
            "Administer tocilizumab 8 mg/kg plus intravenous dexamethasone and levetiracetam."
        ),
    },
    {
        "id": "med_15",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Allergy-immunology hereditary angioedema: 32F C1-esterase inhibitor deficiency "
            "type I with acute laryngeal edema, stridor, voice hoarseness, normal tryptase, "
            "unresponsive to intramuscular epinephrine and antihistamines. Administer subcutaneous "
            "icatibant 30 mg or plasma-derived C1-INH concentrate and prepare fiberoptic intubation."
        ),
    },
    {
        "id": "med_16",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Gastroenterology acute severe ulcerative colitis: 28M Truelove and Witts severe "
            "pancolitis, 11 bloody bowel movements daily, fecal calprotectin >2000 ug/g, "
            "colonic dilation 5.8 cm on abdominal radiograph, Day 3 non-responder to IV "
            "methylprednisolone 60 mg. Rescue with accelerated infliximab 10 mg/kg after TB screen."
        ),
    },
    {
        "id": "med_17",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Vascular surgery and hematology: 53F postoperative heparin-induced thrombocytopenia "
            "day 8 after cardiopulmonary bypass, platelet drop from 240k to 58k/uL, positive "
            "anti-PF4/heparin ELISA OD 2.4 and serotonin release assay, acute right femoral vein "
            "thrombosis. Discontinue all heparin and titrate argatroban to aPTT 1.5-3.0x baseline."
        ),
    },
    {
        "id": "med_18",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Ophthalmology-neurology urgent evaluation: 74F sudden painless monocular altitudinal "
            "visual field loss right eye, chalky pallid optic disc edema, jaw claudication, "
            "scalp tenderness, ESR 96 mm/hr, CRP 68 mg/L. Start immediate high-dose pulse IV "
            "methylprednisolone 1000 mg daily for giant cell arteritis prior to temporal biopsy."
        ),
    },
    {
        "id": "med_19",
        "category_idx": 0,
        "category": "Clinical/Medical",
        "prompt": (
            "Pulmonary critical care ARDS ventilator protocol: 47M severe influenza A pneumonia "
            "with secondary MRSA necrotizing pneumonitis, PaO2/FiO2 84 mmHg on PEEP 14 cmH2O, "
            "plateau pressure 31 cmH2O, driving pressure 17 cmH2O. Transition to 16-hour prone "
            "positioning, low tidal volume 4.5 mL/kg PBW, and cisatracurium neuromuscular blockade."
        ),
    },

    # =========================================================================
    # 1. SQL/Database Tool (20 prompts)
    # =========================================================================
    {
        "id": "sql_00",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__execute_readonly_query", "arguments": {"database": '
            '"warehouse_prod", "timeout_ms": 15000, "sql": "WITH monthly_cohorts AS (SELECT '
            'tenant_id, DATE_TRUNC(\'month\', created_at) AS cohort_month FROM accounts WHERE '
            'plan_tier = \'enterprise\') SELECT c.cohort_month, COUNT(DISTINCT e.tenant_id) '
            'FROM monthly_cohorts c JOIN usage_events e USING (tenant_id) GROUP BY 1 ORDER BY 1;"}}'
        ),
    },
    {
        "id": "sql_01",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__snowflake__run_analytics_sql", "arguments": {"warehouse": '
            '"COMPUTE_WH_XL", "schema": "FINANCE_LEDGER", "query": "SELECT region_code, '
            'product_sku, SUM(net_revenue_usd) OVER (PARTITION BY region_code ORDER BY '
            'fiscal_week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS rolling_4wk_rev '
            'FROM fact_daily_bookings WHERE fiscal_year = 2026 QUALIFY ROW_NUMBER() OVER '
            '(PARTITION BY region_code ORDER BY net_revenue_usd DESC) <= 10;"}}'
        ),
    },
    {
        "id": "sql_02",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__inspect_locks", "arguments": {"cluster": "oltp-primary-01", '
            '"sql": "SELECT blocked.pid AS blocked_pid, blocked.query AS blocked_stmt, '
            'blocking.pid AS blocking_pid, blocking.state, NOW() - blocking.xact_start AS xact_age '
            'FROM pg_catalog.pg_locks bl JOIN pg_stat_activity blocked ON blocked.pid = bl.pid '
            'JOIN pg_catalog.pg_locks kl ON kl.transactionid = bl.transactionid AND kl.pid != bl.pid '
            'JOIN pg_stat_activity blocking ON blocking.pid = kl.pid WHERE NOT bl.granted;"}}'
        ),
    },
    {
        "id": "sql_03",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__bigquery__execute_sql", "arguments": {"project_id": "corp-telemetry", '
            '"dataset": "clickstream_v4", "use_legacy_sql": false, "query": "SELECT '
            'APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)] AS p99_ms, endpoint_route, '
            'COUNTIF(http_status >= 500) / COUNT(*) AS error_rate FROM `clickstream_v4.api_spans` '
            'WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 HOUR) '
            'GROUP BY endpoint_route HAVING COUNT(*) > 10000 ORDER BY p99_ms DESC;"}}'
        ),
    },
    {
        "id": "sql_04",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__explain_analyze", "arguments": {"database": "inventory_db", '
            '"format": "JSON", "sql": "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT o.order_id, '
            'o.warehouse_id, SUM(li.quantity * li.unit_cost) FROM purchase_orders o INNER JOIN '
            'order_line_items li ON li.order_id = o.order_id WHERE o.status = \'BACKORDERED\' '
            'AND o.updated_at >= NOW() - INTERVAL \'14 days\' GROUP BY o.order_id, o.warehouse_id;"}}'
        ),
    },
    {
        "id": "sql_05",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__duckdb__query_parquet", "arguments": {"parquet_glob": '
            '"s3://datalake-curated/events/dt=2026-11-*/part-*.parquet", "sql": "WITH RECURSIVE '
            'org_tree AS (SELECT dept_id, parent_dept_id, 1 AS depth FROM departments WHERE '
            'parent_dept_id IS NULL UNION ALL SELECT d.dept_id, d.parent_dept_id, t.depth + 1 '
            'FROM departments d JOIN org_tree t ON d.parent_dept_id = t.dept_id) SELECT * FROM org_tree;"}}'
        ),
    },
    {
        "id": "sql_06",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__schema_migration_plan", "arguments": {"database": "core_crm", '
            '"ddl": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contacts_org_email_active ON '
            'public.customer_contacts (organization_id, lower(email_address)) INCLUDE (lifecycle_stage, '
            'owner_user_id) WHERE deleted_at IS NULL; ALTER TABLE public.customer_contacts '
            'ADD CONSTRAINT chk_valid_stage CHECK (lifecycle_stage IN (\'lead\', \'sql\', \'won\')) NOT VALID;"}}'
        ),
    },
    {
        "id": "sql_07",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__clickhouse__run_query", "arguments": {"cluster": "metrics_shard_02", '
            '"database": "observability", "sql": "SELECT toStartOfFiveMinute(ts) AS bucket, '
            'service_name, quantileTDigest(0.95)(duration_us) AS p95_us, uniqExact(trace_id) '
            'AS traces FROM distributed_otel_spans WHERE ts >= now() - INTERVAL 2 HOUR AND '
            'span_kind = \'SERVER\' GROUP BY bucket, service_name ORDER BY bucket DESC, p95_us DESC;"}}'
        ),
    },
    {
        "id": "sql_08",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__mysql__read_query", "arguments": {"instance": "orders-aurora-ro", '
            '"schema": "fulfillment", "sql": "SELECT s.carrier_code, s.origin_hub, '
            'AVG(TIMESTAMPDIFF(HOUR, s.dispatched_at, s.delivered_at)) AS avg_transit_hours, '
            'SUM(CASE WHEN s.delivered_at > s.promised_eta THEN 1 ELSE 0 END) AS sla_misses '
            'FROM shipments s FORCE INDEX (idx_dispatched_carrier) WHERE s.dispatched_at >= '
            'CURDATE() - INTERVAL 30 DAY GROUP BY s.carrier_code, s.origin_hub;"}}'
        ),
    },
    {
        "id": "sql_09",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__upsert_batch", "arguments": {"database": "feature_store", '
            '"sql": "INSERT INTO merchant_risk_features (merchant_id, chargeback_ratio_90d, '
            'avg_ticket_cents, velocity_zscore, computed_at) SELECT m.id, r.cb_ratio, '
            'r.avg_cents, r.zscore, NOW() FROM staging_merchant_metrics r JOIN merchants m '
            'ON m.external_ref = r.ext_ref ON CONFLICT (merchant_id) DO UPDATE SET '
            'chargeback_ratio_90d = EXCLUDED.chargeback_ratio_90d, computed_at = EXCLUDED.computed_at;"}}'
        ),
    },
    {
        "id": "sql_10",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__snowflake__table_reconciliation", "arguments": {"database": "EDW_PROD", '
            '"sql": "WITH src AS (SELECT account_id, SUM(debit_cents - credit_cents) AS bal '
            'FROM raw_journal_entries WHERE posting_date = \'2026-11-17\' GROUP BY 1), '
            'tgt AS (SELECT account_id, ending_balance_cents - opening_balance_cents AS bal '
            'FROM gl_daily_snapshots WHERE snapshot_date = \'2026-11-17\') SELECT COALESCE(src.account_id, '
            'tgt.account_id), src.bal, tgt.bal FROM src FULL OUTER JOIN tgt USING (account_id) '
            'WHERE COALESCE(src.bal, 0) != COALESCE(tgt.bal, 0);"}}'
        ),
    },
    {
        "id": "sql_11",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__pg_stat_statements_top", "arguments": {"database": "saas_multitenant", '
            '"limit": 25, "sql": "SELECT queryid, calls, ROUND(total_exec_time::numeric, 2) AS total_ms, '
            'ROUND(mean_exec_time::numeric, 3) AS mean_ms, shared_blks_read, shared_blks_hit, '
            'LEFT(query, 140) AS sample_sql FROM pg_stat_statements WHERE dbid = (SELECT oid '
            'FROM pg_database WHERE datname = current_database()) ORDER BY total_exec_time DESC LIMIT 25;"}}'
        ),
    },
    {
        "id": "sql_12",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__trino__federated_query", "arguments": {"catalog": "iceberg_prod", '
            '"schema": "supply_chain", "sql": "SELECT p.supplier_id, p.part_number, '
            'LAG(p.unit_price, 1) OVER (PARTITION BY p.part_number ORDER BY p.effective_date) '
            'AS prev_price, p.unit_price, (p.unit_price - LAG(p.unit_price, 1) OVER '
            '(PARTITION BY p.part_number ORDER BY p.effective_date)) / NULLIF(p.unit_price, 0) '
            'AS pct_delta FROM iceberg_prod.supply_chain.part_price_history p;"}}'
        ),
    },
    {
        "id": "sql_13",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__vector_hybrid_search", "arguments": {"database": "kb_rag", '
            '"sql": "WITH sem AS (SELECT doc_id, 1 - (embedding <=> $1::vector(1536)) AS cos_sim '
            'FROM document_chunks WHERE namespace = \'sec_filings\' ORDER BY embedding <=> $1::vector '
            'LIMIT 50), lex AS (SELECT doc_id, ts_rank_cd(tsv_body, plainto_tsquery(\'english\', $2)) '
            'AS bm25 FROM document_chunks WHERE tsv_body @@ plainto_tsquery(\'english\', $2) LIMIT 50) '
            'SELECT COALESCE(sem.doc_id, lex.doc_id), COALESCE(sem.cos_sim,0)*0.7 + COALESCE(lex.bm25,0)*0.3 AS rrf FROM sem FULL JOIN lex USING (doc_id);"}}'
        ),
    },
    {
        "id": "sql_14",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__redshift__unload_cohort_sql", "arguments": {"cluster": "bi-dw-01", '
            '"sql": "SELECT u.country_iso, u.acquisition_channel, COUNT(DISTINCT u.user_id) AS installs, '
            'COUNT(DISTINCT CASE WHEN s.day_offset BETWEEN 1 AND 7 THEN s.user_id END)::FLOAT / '
            'NULLIF(COUNT(DISTINCT u.user_id), 0) AS d7_retention FROM dim_users u LEFT JOIN '
            'fact_user_sessions s ON s.user_id = u.user_id WHERE u.signup_date >= \'2026-09-01\' '
            'GROUP BY 1, 2 HAVING COUNT(DISTINCT u.user_id) >= 500 ORDER BY d7_retention DESC;"}}'
        ),
    },
    {
        "id": "sql_15",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__partition_maintenance", "arguments": {"database": "audit_trail", '
            '"sql": "CREATE TABLE IF NOT EXISTS audit_events_y2026m12 PARTITION OF audit_events '
            'FOR VALUES FROM (\'2026-12-01 00:00:00+00\') TO (\'2027-01-01 00:00:00+00\'); '
            'ALTER TABLE audit_events DETACH PARTITION audit_events_y2025m10 CONCURRENTLY; '
            'SELECT inhrelid::regclass AS child_table, pg_size_pretty(pg_total_relation_size(inhrelid)) '
            'FROM pg_inherits WHERE inhparent = \'public.audit_events\'::regclass;"}}'
        ),
    },
    {
        "id": "sql_16",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__sqlite__local_ledger_query", "arguments": {"db_path": "/var/lib/agent/state.db", '
            '"sql": "SELECT j.workflow_id, j.step_name, COUNT(*) FILTER (WHERE j.status = \'RETRYING\') '
            'AS retries, MAX(j.attempt_num) AS max_attempt, json_extract(j.payload_json, \'$.error_code\') '
            'AS err_code FROM workflow_step_journal j WHERE j.created_epoch_ms >= strftime(\'%s\',\'now\')*1000 '
            '- 86400000 GROUP BY j.workflow_id, j.step_name, err_code ORDER BY retries DESC;"}}'
        ),
    },
    {
        "id": "sql_17",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__spanner__execute_streaming_sql", "arguments": {"instance": "global-inventory", '
            '"database": "sku-catalog", "sql": "SELECT s.sku_id, s.region, s.on_hand_qty - '
            'COALESCE(SUM(r.reserved_qty), 0) AS available_to_promise FROM SkuRegionalStock@{FORCE_INDEX=IdxRegionSku} s '
            'LEFT JOIN ActiveCartReservations r ON r.sku_id = s.sku_id AND r.region = s.region '
            'AND r.expires_at > CURRENT_TIMESTAMP() WHERE s.region = \'us-east\' GROUP BY 1, 2, s.on_hand_qty;"}}'
        ),
    },
    {
        "id": "sql_18",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__postgres__bloat_and_vacuum_audit", "arguments": {"database": "billing_oltp", '
            '"sql": "SELECT schemaname, relname, n_live_tup, n_dead_tup, ROUND(100.0 * n_dead_tup / '
            'NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_pct, last_autovacuum, last_autoanalyze '
            'FROM pg_stat_user_tables WHERE n_live_tup > 50000 ORDER BY n_dead_tup DESC LIMIT 20;"}}'
        ),
    },
    {
        "id": "sql_19",
        "category_idx": 1,
        "category": "SQL/Database Tool",
        "prompt": (
            '{"tool": "mcp__databricks__spark_sql_execute", "arguments": {"catalog": "unity_prod", '
            '"schema": "adtech_attribution", "sql": "SELECT i.campaign_id, COUNT(DISTINCT i.impression_id) '
            'AS imps, COUNT(DISTINCT c.conversion_id) AS convs, SUM(c.order_value_usd) / '
            'NULLIF(SUM(i.cpm_cost_usd), 0) AS roas FROM ad_impressions i LEFT JOIN conversions c '
            'ON c.user_pseudo_id = i.user_pseudo_id AND c.conv_ts BETWEEN i.imp_ts AND i.imp_ts + INTERVAL 7 DAYS GROUP BY 1;"}}'
        ),
    },

    # =========================================================================
    # 2. Stripe/Payment Tool (20 prompts)
    # =========================================================================
    {
        "id": "pay_00",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__create_payment_intent", "arguments": {"amount": 485000, '
            '"currency": "usd", "customer": "cus_R9xK2mLp8vQw4N", "payment_method": '
            '"pm_1QKb7r2eZvKYlo2C9fH8mN1p", "capture_method": "automatic_async", "confirm": true, '
            '"statement_descriptor_suffix": "ENT-ANNUAL", "idempotency_key": "idem_pi_99481a7c2e", '
            '"metadata": {"invoice_number": "INV-2026-8841", "seat_count": "120"}}}'
        ),
    },
    {
        "id": "pay_01",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__create_refund", "arguments": {"payment_intent": '
            '"pi_3QJc9L2eZvKYlo2C1xW4vB8k", "amount": 129900, "reason": "requested_by_customer", '
            '"refund_application_fee": true, "reverse_transfer": true, "idempotency_key": '
            '"idem_rfnd_77210b4d", "metadata": {"rma_ticket": "RMA-44912", "approved_by": "finops_tier2"}}}'
        ),
    },
    {
        "id": "pay_02",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__update_subscription_items", "arguments": {"subscription_id": '
            '"sub_1QHm4P2eZvKYlo2C7tR5nK9w", "proration_behavior": "always_invoice", '
            '"payment_behavior": "error_if_incomplete", "items": [{"id": "si_R9aB3cD4eF5gH6", '
            '"price": "price_enterprise_seat_monthly_usd", "quantity": 85}, {"price": '
            '"price_metered_gpu_tokens_v2"}], "coupon": "ANNUAL_COMMIT_15PCT"}}'
        ),
    },
    {
        "id": "pay_03",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__create_connect_transfer", "arguments": {"amount": 942500, '
            '"currency": "usd", "destination": "acct_1P9mK8Q2wE4rT6yU", "source_transaction": '
            '"ch_3QKd1M2eZvKYlo2C0pL9sV3n", "transfer_group": "ORDER_SPLIT_2026_11_18", '
            '"description": "Marketplace seller net payout minus 8.5% platform commission", '
            '"metadata": {"seller_batch_id": "BATCH-9031", "tax_form_1099k_verified": "true"}}}'
        ),
    },
    {
        "id": "pay_04",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__submit_dispute_evidence", "arguments": {"dispute_id": '
            '"dp_1QKe5R2eZvKYlo2C4mZ8qX2b", "submit": true, "evidence": {"customer_email_address": '
            '"procurement@acmecorp.io", "customer_purchase_ip": "198.51.100.42", '
            '"shipping_carrier": "FedEx Priority", "shipping_tracking_number": "778940128834", '
            '"access_activity_log": "API key authenticated 412 requests from ASN 14618 post-checkout", '
            '"uncategorized_file": "file_1QKe2A2eZvKYlo2Csigned_po"}}}'
        ),
    },
    {
        "id": "pay_05",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__plaid__initiate_ach_transfer", "arguments": {"access_token": '
            '"access-production-8f92c1d4-5b6e-4a19", "account_id": "Bx79mQvL2nK4pW8rT1yZa3", '
            '"type": "debit", "network": "same-day-ach", "amount": "15750.00", "ach_class": "ccd", '
            '"iso_currency_code": "USD", "description": "NOV PAYROLL FUNDING", "idempotency_key": '
            '"plaid_ach_20261118_0042", "user": {"legal_name": "Northstar BioLabs Inc"}}}'
        ),
    },
    {
        "id": "pay_06",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__finalize_and_pay_invoice", "arguments": {"invoice_id": '
            '"in_1QKf8T2eZvKYlo2C6nB2vM7q", "auto_advance": true, "payment_method": '
            '"pm_usBankAccount_achDirectDebit_91x", "off_session": true, "mandate": '
            '"mandate_1QAb3C2eZvKYlo2C8kL4pQ1z", "metadata": {"billing_period": "2026-11", '
            '"po_reference": "PO-99381-CLOUD", "net_terms": "IMMEDIATE_ACH"}}}'
        ),
    },
    {
        "id": "pay_07",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__create_checkout_session", "arguments": {"mode": "subscription", '
            '"customer_email": "vp-eng@quantumsys.ai", "billing_address_collection": "required", '
            '"automatic_tax": {"enabled": true}, "tax_id_collection": {"enabled": true}, '
            '"line_items": [{"price": "price_1Q9xYz2eZvKYlo2CproTier", "quantity": 25}], '
            '"subscription_data": {"trial_period_days": 14}, "success_url": "https://app.example.com/billing/ok"}}'
        ),
    },
    {
        "id": "pay_08",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__issue_virtual_card", "arguments": {"cardholder": '
            '"ich_1QKg2V2eZvKYlo2C3wE5rT7y", "currency": "usd", "type": "virtual", "status": "active", '
            '"spending_controls": {"spending_limits": [{"amount": 250000, "interval": "per_authorization"}, '
            '{"amount": 1000000, "interval": "monthly"}], "allowed_categories": ["computer_software_stores", '
            '"cloud_computing_services"]}, "metadata": {"agent_budget_id": "procurement_bot_04"}}}'
        ),
    },
    {
        "id": "pay_09",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__record_meter_event", "arguments": {"event_name": '
            '"disaggregated_moe_completion_tokens", "identifier": "mtr_ev_8829104f9a1c", '
            '"timestamp": 1795070400, "payload": {"stripe_customer_id": "cus_Q8mN4pL2vK9xR1", '
            '"value": "1482900", "model_tier": "deepseek_v3_ep8", "region": "us-central1"}}}'
        ),
    },
    {
        "id": "pay_10",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__adyen__authorise_sepa_direct_debit", "arguments": {"merchantAccount": '
            '"CloudScaleEU", "amount": {"currency": "EUR", "value": 640000}, "reference": '
            '"SEPA-INV-2026-4410", "paymentMethod": {"type": "sepadirectdebit", "iban": '
            '"DE89370400440532013000", "ownerName": "Bergmann Industrietechnik GmbH"}, '
            '"recurringProcessingModel": "Subscription", "shopperReference": "CUST-EU-7781"}}'
        ),
    },
    {
        "id": "pay_11",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__create_payout", "arguments": {"stripe_account": '
            '"acct_1N8vB4Q2wE6rT8yP", "amount": 2840000, "currency": "usd", "method": "instant", '
            '"destination": "ba_1QKh9X2eZvKYlo2C5tG7hJ9k", "source_type": "card", '
            '"statement_descriptor": "MERCHANT SETTLEMENT", "idempotency_key": "payout_inst_1118_a9"}}'
        ),
    },
    {
        "id": "pay_12",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__create_credit_note", "arguments": {"invoice": '
            '"in_1QJw4K2eZvKYlo2C9mP3xL6v", "reason": "order_change", "credit_amount": 75000, '
            '"out_of_band_amount": 0, "memo": "SLA uptime credit for 42-minute regional API degradation", '
            '"lines": [{"type": "custom_line_item", "description": "99.9% SLA Service Credit", '
            '"quantity": 1, "unit_amount": 75000}]}}'
        ),
    },
    {
        "id": "pay_13",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__verify_webhook_event", "arguments": {"event_id": '
            '"evt_3QKi2Z2eZvKYlo2C1qA4zW7s", "event_type": "payment_intent.payment_failed", '
            '"data_object": {"id": "pi_3QKi2Y2eZvKYlo2C1qA4zW7s", "amount": 320000, "currency": "usd", '
            '"last_payment_error": {"code": "card_declined", "decline_code": "insufficient_funds", '
            '"network_decline_code": "51", "payment_method_type": "card"}}, "retry_dunning_schedule": "day_3_7_14"}}'
        ),
    },
    {
        "id": "pay_14",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__capture_payment_intent", "arguments": {"payment_intent_id": '
            '"pi_3QKh1B2eZvKYlo2C8vN5mK2p", "amount_to_capture": 192450, "final_capture": true, '
            '"application_fee_amount": 9623, "statement_descriptor": "FREIGHT ESCROW", '
            '"metadata": {"bol_signed": "true", "customs_clearance_ref": "US-CBP-884120"}}}'
        ),
    },
    {
        "id": "pay_15",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__wise__create_crossborder_quote_transfer", "arguments": {"profile_id": '
            '49281740, "source_currency": "USD", "target_currency": "GBP", "target_amount": 42500.00, '
            '"recipient_account_id": 883920144, "payment_purpose": "CONTRACTOR_MILESTONE_PAYMENT", '
            '"customer_transaction_id": "2f9a8c41-7d12-49e8-9b30-61d8f4a209c7"}}'
        ),
    },
    {
        "id": "pay_16",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__radar_early_fraud_warning_action", "arguments": {"efw_id": '
            '"issfr_1QKj5D2eZvKYlo2C7uY6tR4e", "charge_id": "ch_3QKb9N2eZvKYlo2C2wQ8eR5t", '
            '"fraud_type": "unauthorized_use_of_card", "risk_score": 89, "action": '
            '"proactive_full_refund_and_block_pm", "revoke_api_tokens_for_customer": "cus_R1zX9cV8bN7m"}}'
        ),
    },
    {
        "id": "pay_17",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__create_subscription_schedule", "arguments": {"customer": '
            '"cus_P7lK9jH5gF3dS1", "start_date": 1796083200, "end_behavior": "release", '
            '"phases": [{"items": [{"price": "price_ramp_q1_usd", "quantity": 50}], "iterations": 3}, '
            '{"items": [{"price": "price_ramp_q2_usd", "quantity": 150}], "iterations": 9, '
            '"collection_method": "send_invoice", "days_until_due": 30}]}}'
        ),
    },
    {
        "id": "pay_18",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__treasury_outbound_payment", "arguments": {"financial_account": '
            '"fa_1QKm3L2eZvKYlo2C9pO8iU7y", "amount": 5500000, "currency": "usd", '
            '"destination_payment_method_data": {"type": "us_bank_account", "financial_connections_account": '
            '"fca_1QKm1K2eZvKYlo2C4rE3wQ2a"}, "us_bank_account_network": "wire", '
            '"statement_descriptor": "SERIES B ESCROW WIRE"}}'
        ),
    },
    {
        "id": "pay_19",
        "category_idx": 2,
        "category": "Stripe/Payment Tool",
        "prompt": (
            '{"tool": "mcp__stripe__list_balance_transactions", "arguments": {"payout": '
            '"po_1QKn7P2eZvKYlo2C6mJ5hG4f", "type": "charge", "limit": 100, "expand": '
            '["data.source", "data.fee_details"], "created": {"gte": 1794897600, "lte": 1795070400}, '
            '"reconcile_gl_account": "1020_STRIPE_CLEARING_USD"}}'
        ),
    },

    # =========================================================================
    # 3. Crypto/Auth Code (20 prompts)
    # =========================================================================
    {
        "id": "cry_00",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def encrypt_envelope_aes256_gcm(dek_bytes: bytes, plaintext: bytes, aad: bytes) -> bytes:\n"
            "    nonce = os.urandom(12)\n"
            "    cipher = AESGCM(dek_bytes)\n"
            "    ct_and_tag = cipher.encrypt(nonce, plaintext, associated_data=aad)\n"
            "    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=nonce, info=b'envelope-v2')\n"
            "    mac = hmac.new(hkdf.derive(dek_bytes), nonce + ct_and_tag, hashlib.sha256).digest()\n"
            "    return nonce + mac + ct_and_tag"
        ),
    },
    {
        "id": "cry_01",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def verify_ed25519_jws_detatched(pubkey_pem: bytes, jws_header_b64: str, payload: bytes, sig_b64: str) -> bool:\n"
            "    pub = serialization.load_pem_public_key(pubkey_pem)\n"
            "    if not isinstance(pub, ed25519.Ed25519PublicKey):\n"
            "        raise TypeError('Expected Ed25519 verifying key')\n"
            "    signing_input = jws_header_b64.encode('ascii') + b'.' + base64.urlsafe_b64encode(payload).rstrip(b'=')\n"
            "    raw_sig = base64.urlsafe_b64decode(sig_b64 + '=' * (-len(sig_b64) % 4))\n"
            "    pub.verify(raw_sig, signing_input)\n"
            "    return True"
        ),
    },
    {
        "id": "cry_02",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def verify_argon2id_password_constant_time(stored_phc: str, candidate_pw: str, pepper_key: bytes) -> bool:\n"
            "    peppered = hmac.new(pepper_key, candidate_pw.encode('utf-8'), hashlib.sha512).digest()\n"
            "    ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4, hash_len=32, type=Type.ID)\n"
            "    try:\n"
            "        return ph.verify(stored_phc, base64.b64encode(peppered).decode('ascii'))\n"
            "    except VerifyMismatchError:\n"
            "        return False"
        ),
    },
    {
        "id": "cry_03",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def validate_oauth2_pkce_s256(code_verifier: str, stored_challenge: str) -> bool:\n"
            "    if not (43 <= len(code_verifier) <= 128):\n"
            "        return False\n"
            "    digest = hashlib.sha256(code_verifier.encode('ascii')).digest()\n"
            "    computed = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')\n"
            "    return hmac.compare_digest(computed.encode('ascii'), stored_challenge.encode('ascii'))"
        ),
    },
    {
        "id": "cry_04",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "pub fn x25519_hpke_derive_session_keys(eph_sk: StaticSecret, peer_pk: PublicKey, ctx: &[u8]) -> ([u8; 32], [u8; 32]) {\n"
            "    let shared_secret = eph_sk.diffie_hellman(&peer_pk);\n"
            "    let hk = Hkdf::<Sha384>::new(Some(b\"hpke-x25519-sha384\"), shared_secret.as_bytes());\n"
            "    let (mut tx_key, mut rx_key) = ([0u8; 32], [0u8; 32]);\n"
            "    hk.expand(&[ctx, b\"client-tx\"].concat(), &mut tx_key).expect(\"hkdf tx\");\n"
            "    hk.expand(&[ctx, b\"server-rx\"].concat(), &mut rx_key).expect(\"hkdf rx\");\n"
            "    (tx_key, rx_key)\n}"
        ),
    },
    {
        "id": "cry_05",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def verify_mtls_spiffe_client_cert(cert_der: bytes, trusted_root: x509.Certificate, expected_spiffe: str) -> None:\n"
            "    leaf = x509.load_der_x509_certificate(cert_der)\n"
            "    trusted_root.public_key().verify(leaf.signature, leaf.tbs_certificate_bytes, ec.ECDSA(leaf.signature_hash_algorithm))\n"
            "    san = leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value\n"
            "    uris = san.get_values_for_type(x509.UniformResourceIdentifier)\n"
            "    if expected_spiffe not in uris or leaf.not_valid_after_utc < datetime.now(timezone.utc):\n"
            "        raise PermissionError('SPIFFE mTLS identity verification failed')"
        ),
    },
    {
        "id": "cry_06",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def validate_rs256_oidc_access_token(jwt_str: str, jwks_by_kid: dict, issuer: str, audience: str) -> dict:\n"
            "    hdr_b64, pay_b64, sig_b64 = jwt_str.split('.')\n"
            "    hdr = json.loads(base64.urlsafe_b64decode(hdr_b64 + '=='))\n"
            "    if hdr.get('alg') != 'RS256' or hdr.get('kid') not in jwks_by_kid:\n"
            "        raise ValueError('Disallowed JWT alg or unknown kid')\n"
            "    rsa_pub = jwks_by_kid[hdr['kid']]\n"
            "    rsa_pub.verify(urlsafe_b64decode_pad(sig_b64), f'{hdr_b64}.{pay_b64}'.encode(), padding.PKCS1v15(), hashes.SHA256())\n"
            "    return verify_claims_exp_iss_aud(pay_b64, issuer, audience)"
        ),
    },
    {
        "id": "cry_07",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def compute_totp_rfc6238(secret_b32: str, unix_ts: int, step: int = 30, digits: int = 6) -> str:\n"
            "    key = base64.b32decode(secret_b32.upper(), casefold=True)\n"
            "    counter = struct.pack('>Q', unix_ts // step)\n"
            "    digest = hmac.new(key, counter, hashlib.sha1).digest()\n"
            "    offset = digest[-1] & 0x0F\n"
            "    code_int = (struct.unpack('>I', digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)\n"
            "    return str(code_int).zfill(digits)"
        ),
    },
    {
        "id": "cry_08",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def verify_webauthn_fido2_assertion(rp_id: str, client_data_json: bytes, auth_data: bytes, sig: bytes, cose_pk: ec.EllipticCurvePublicKey, expected_challenge: bytes) -> int:\n"
            "    cd = json.loads(client_data_json)\n"
            "    assert cd['type'] == 'webauthn.get' and urlsafe_decode(cd['challenge']) == expected_challenge\n"
            "    assert hmac.compare_digest(auth_data[:32], hashlib.sha256(rp_id.encode()).digest())\n"
            "    assert auth_data[32] & 0x01 != 0\n"
            "    cose_pk.verify(sig, auth_data + hashlib.sha256(client_data_json).digest(), ec.ECDSA(hashes.SHA256()))\n"
            "    return struct.unpack('>I', auth_data[33:37])[0]"
        ),
    },
    {
        "id": "cry_09",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "pub fn chacha20_poly1305_ratchet_step(chain_key: &[u8; 32], msg_num: u32, pt: &[u8]) -> ([u8; 32], Vec<u8>) {\n"
            "    let mut mac_msg = Hmac::<Sha256>::new_from_slice(chain_key).unwrap();\n"
            "    mac_msg.update(&[0x01]);\n"
            "    let msg_key: [u8; 32] = mac_msg.finalize().into_bytes().into();\n"
            "    let mut mac_next = Hmac::<Sha256>::new_from_slice(chain_key).unwrap();\n"
            "    mac_next.update(&[0x02]);\n"
            "    let next_ck: [u8; 32] = mac_next.finalize().into_bytes().into();\n"
            "    let cipher = ChaCha20Poly1305::new((&msg_key).into());\n"
            "    (next_ck, cipher.encrypt(& build_nonce(msg_num), pt).unwrap())\n}"
        ),
    },
    {
        "id": "cry_10",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def sign_aws_sigv4_headers(access_key: str, secret_key: str, region: str, service: str, amz_date: str, canonical_req: str) -> str:\n"
            "    datestamp = amz_date[:8]\n"
            "    k_date = hmac.new(('AWS4' + secret_key).encode(), datestamp.encode(), hashlib.sha256).digest()\n"
            "    k_region = hmac.new(k_date, region.encode(), hashlib.sha256).digest()\n"
            "    k_service = hmac.new(k_region, service.encode(), hashlib.sha256).digest()\n"
            "    k_signing = hmac.new(k_service, b'aws4_request', hashlib.sha256).digest()\n"
            "    sts = f'AWS4-HMAC-SHA256\\n{amz_date}\\n{datestamp}/{region}/{service}/aws4_request\\n' + hashlib.sha256(canonical_req.encode()).hexdigest()\n"
            "    return hmac.new(k_signing, sts.encode(), hashlib.sha256).hexdigest()"
        ),
    },
    {
        "id": "cry_11",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def shamir_split_gf256_secret(secret: bytes, threshold_k: int, num_shares_n: int) -> list[tuple[int, bytes]]:\n"
            "    assert 2 <= threshold_k <= num_shares_n <= 255\n"
            "    coeffs = [[b] + [ secrets.randbelow(256) for _ in range(threshold_k - 1) ] for b in secret]\n"
            "    shares = []\n"
            "    for x in range(1, num_shares_n + 1):\n"
            "        y_bytes = bytes(gf256_eval_poly(poly, x) for poly in coeffs)\n"
            "        shares.append((x, y_bytes))\n"
            "    return shares"
        ),
    },
    {
        "id": "cry_12",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def mint_dpop_bound_proof_jwt(ec_privkey: ec.EllipticCurvePrivateKey, htm: str, htu: str, ath_sha256_b64: str) -> str:\n"
            "    jwk_pub = export_ec_p256_public_jwk(ec_privkey.public_key())\n"
            "    header = {'typ': 'dpop+jwt', 'alg': 'ES256', 'jwk': jwk_pub}\n"
            "    claims = {'jti': secrets.token_urlsafe(24), 'htm': htm, 'htu': htu, 'iat': int(time.time()), 'ath': ath_sha256_b64}\n"
            "    signing_input = b64u_json(header) + b'.' + b64u_json(claims)\n"
            "    der_sig = ec_privkey.sign(signing_input, ec.ECDSA(hashes.SHA256()))\n"
            "    return (signing_input + b'.' + b64u(der_to_raw_ecdsa_rs(der_sig, 32))).decode('ascii')"
        ),
    },
    {
        "id": "cry_13",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def merkle_verify_inclusion_proof(leaf_payload: bytes, leaf_index: int, sibling_hashes: list[bytes], expected_root: bytes) -> bool:\n"
            "    acc = hashlib.sha256(b'\\x00' + leaf_payload).digest()\n"
            "    idx = leaf_index\n"
            "    for sib in sibling_hashes:\n"
            "        if idx & 1:\n"
            "            acc = hashlib.sha256(b'\\x01' + sib + acc).digest()\n"
            "        else:\n"
            "            acc = hashlib.sha256(b'\\x01' + acc + sib).digest()\n"
            "        idx >>= 1\n"
            "    return hmac.compare_digest(acc, expected_root)"
        ),
    },
    {
        "id": "cry_14",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def rsa_oaep_unwrap_kms_cek(rsa_priv: rsa.RSAPrivateKey, wrapped_cek: bytes, key_context: bytes) -> bytes:\n"
            "    cek = rsa_priv.decrypt(\n"
            "        wrapped_cek,\n"
            "        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=key_context),\n"
            "    )\n"
            "    if len(cek) != 32:\n"
            "        raise ValueError('Invalid unwrapped AES-256 content encryption key length')\n"
            "    return cek"
        ),
    },
    {
        "id": "cry_15",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def verify_macaroon_caveats(root_key: bytes, identifier: bytes, caveats: list[bytes], presented_tag: bytes, verifier_fn) -> bool:\n"
            "    sig = hmac.new(root_key, identifier, hashlib.sha256).digest()\n"
            "    for caveat in caveats:\n"
            "        if not verifier_fn(caveat.decode('utf-8')):\n"
            "            return False\n"
            "        sig = hmac.new(sig, caveat, hashlib.sha256).digest()\n"
            "    return hmac.compare_digest(sig, presented_tag)"
        ),
    },
    {
        "id": "cry_16",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def scrypt_derive_keystore_cipherkey(passphrase: str, salt: bytes, ciphertext: bytes, mac_expected: bytes) -> bytes:\n"
            "    kdf = Scrypt(salt=salt, length=64, n=2**15, r=8, p=1)\n"
            "    derived = kdf.derive(passphrase.encode('utf-8'))\n"
            "    enc_key, mac_key = derived[:32], derived[32:]\n"
            "    computed_mac = hashlib.sha3_256(mac_key + ciphertext).digest()\n"
            "    if not hmac.compare_digest(computed_mac, mac_expected):\n"
            "        raise InvalidTag('Keystore MAC authentication failed')\n"
            "    return enc_key"
        ),
    },
    {
        "id": "cry_17",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "pub fn verify_webhook_replay_window(secret: &[u8], ts_header: i64, now_unix: i64, raw_body: &[u8], hex_sig: &str) -> Result<(), AuthErr> {\n"
            "    if (now_unix - ts_header).abs() > 300 { return Err(AuthErr::StaleTimestamp); }\n"
            "    let mut mac = Hmac::<Sha256>::new_from_slice(secret).map_err(|_| AuthErr::KeyErr)?;\n"
            "    mac.update(format!(\"v0:{}:\", ts_header).as_bytes());\n"
            "    mac.update(raw_body);\n"
            "    let sig_bytes = hex::decode(hex_sig).map_err(|_| AuthErr::BadEncoding)?;\n"
            "    mac.verify_slice(&sig_bytes).map_err(|_| AuthErr::BadSignature)\n}"
        ),
    },
    {
        "id": "cry_18",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def aes_kw_rfc3394_wrap_dek(kek: bytes, target_dek: bytes) -> bytes:\n"
            "    if len(kek) != 32 or len(target_dek) % 8 != 0:\n"
            "        raise ValueError('RFC 3394 requires 256-bit KEK and 64-bit multiple DEK')\n"
            "    wrapped = keywrap.aes_key_wrap(kek, target_dek)\n"
            "    unwrapped_check = keywrap.aes_key_unwrap(kek, wrapped)\n"
            "    assert hmac.compare_digest(unwrapped_check, target_dek)\n"
            "    return wrapped"
        ),
    },
    {
        "id": "cry_19",
        "category_idx": 3,
        "category": "Crypto/Auth Code",
        "prompt": (
            "def validate_saml2_xml_signature_digest(assertion_c14n_bytes: bytes, signed_info_c14n: bytes, digest_b64: str, sig_b64: str, idp_cert: x509.Certificate) -> bool:\n"
            "    calc_digest = base64.b64encode(hashlib.sha256(assertion_c14n_bytes).digest()).decode('ascii')\n"
            "    if not hmac.compare_digest(calc_digest, digest_b64):\n"
            "        return False\n"
            "    idp_cert.public_key().verify(base64.b64decode(sig_b64), signed_info_c14n, padding.PKCS1v15(), hashes.SHA256())\n"
            "    return True"
        ),
    },

    # =========================================================================
    # 4. Legal Contract (20 prompts)
    # =========================================================================
    {
        "id": "leg_00",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "SECTION 9.2 INDEMNIFICATION BY SELLER AND ESCROW CAP. Subject to the survival "
            "periods set forth in Section 9.1, Seller Stockholders shall severally and not "
            "jointly indemnify, defend, and hold harmless Buyer Indemnified Parties from Losses "
            "arising out of any breach of Fundamental Representations, provided aggregate liability "
            "for general representations shall not exceed the Indemnity Escrow Amount of $14,500,000 "
            "subject to a non-refundable deductible tipping basket equal to 0.75% of Enterprise Value."
        ),
    },
    {
        "id": "leg_01",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "DATA PROCESSING ADDENDUM — INTERNATIONAL TRANSFERS & GDPR SCCs. With respect to "
            "Personal Data of European Economic Area Data Subjects transferred to Processor in "
            "third countries lacking an adequacy decision under Article 45 of Regulation (EU) "
            "2016/679, the Parties hereby incorporate Module Two (Controller-to-Processor) of "
            "the European Commission Standard Contractual Clauses (Decision 2021/914) and the "
            "UK International Data Transfer Addendum, with 48-hour Personal Data Breach notification."
        ),
    },
    {
        "id": "leg_02",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "AMENDED AND RESTATED CERTIFICATE OF INCORPORATION — SERIES B PROTECTIVE PROVISIONS. "
            "So long as at least 2,500,000 shares of Series B Preferred Stock remain outstanding, "
            "this Delaware Corporation shall not, either directly or by amendment, merger, or "
            "consolidation, without the affirmative written consent of the Requisite Series B "
            "Holders: (i) authorize any senior or pari passu security, (ii) declare any dividend, "
            "or (iii) consummate a Deemed Liquidation Event yielding less than 2.0x Original Issue Price."
        ),
    },
    {
        "id": "leg_03",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "INTELLECTUAL PROPERTY INDEMNITY & LIMITATION OF LIABILITY. Licensor shall defend "
            "Customer against any third-party suit alleging that the unmodified Subscription "
            "Software infringes any issued United States patent, copyright, or trade secret, "
            "and shall pay damages finally awarded by a court of competent jurisdiction. "
            "NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, INCIDENTAL, CONSEQUENTIAL, OR PUNITIVE "
            "DAMAGES, AND SUPER-CAP LIABILITY SHALL NOT EXCEED THREE TIMES (3X) PRECEDING 12-MONTH FEES."
        ),
    },
    {
        "id": "leg_04",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "MUTUAL NON-DISCLOSURE AND RESIDUALS EXCLUSION CLAUSE. Receiving Party shall hold "
            "Disclosing Party's Confidential Information in strict confidence for five (5) years "
            "following termination, and perpetually with respect to Trade Secrets under the "
            "Defend Trade Secrets Act, 18 U.S.C. § 1836. Receiving Party is expressly prohibited "
            "from reverse engineering, decompiling, or utilizing Confidential Information to train "
            "foundation machine learning weights, and no license is granted by implication or estoppel."
        ),
    },
    {
        "id": "leg_05",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "CHANGE OF CONTROL DOUBLE-TRIGGER VESTING ACCELERATION. In the event Executive's "
            "employment is terminated by the Company without Cause, or Executive resigns for "
            "Good Reason within twelve (12) months following the consummation of a Change in "
            "Control, one hundred percent (100%) of Executive's then-unvested restricted stock "
            "units and stock options shall immediately accelerate and vest, conditioned upon "
            "execution and non-revocation of a general release of claims under Section 409A."
        ),
    },
    {
        "id": "leg_06",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "PATENT CROSS-LICENSE AND COVENANT NOT TO SUE. Subject to timely payment of running "
            "royalties under Schedule C, Licensor grants Licensee a worldwide, non-exclusive, "
            "non-sublicensable, nontransferable license under the Licensed Patent Portfolio "
            "solely within the Defined Field of Use, subject to defensive termination if Licensee "
            "or any Affiliate initiates any inter partes review, declaratory judgment, or patent "
            "infringement proceeding challenging the validity or enforceability of Licensor's claims."
        ),
    },
    {
        "id": "leg_07",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "GOVERNING LAW, JAMS ARBITRATION, AND EQUITABLE INJUNCTIVE RELIEF. This Agreement "
            "shall be governed by and construed in accordance with the internal laws of the "
            "State of New York without giving effect to conflict-of-laws principles (N.Y. Gen. "
            "Oblig. Law § 5-1401). Any dispute arising hereunder shall be resolved by binding "
            "confidential arbitration in New York County before three neutral arbitrators under "
            "JAMS Comprehensive Rules, preserving provisional injunctive relief in federal court."
        ),
    },
    {
        "id": "leg_08",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "HIPAA BUSINESS ASSOCIATE AGREEMENT — SAFEGUARDS & SUBCONTRACTORS. Business Associate "
            "agrees to use Protected Health Information solely as permitted by the Underlying "
            "Services Agreement or as Required By Law under 45 C.F.R. § 164.504(e), implement "
            "administrative, physical, and technical safeguards compliant with the HIPAA Security "
            "Rule, bind all downstream Subcontractors to written obligations no less restrictive, "
            "and report any Unsecured PHI Breach pursuant to HITECH Act 42 U.S.C. § 17932."
        ),
    },
    {
        "id": "leg_09",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "COMMERCIAL LEASE — FORCE MAJEURE, CASUALTY & RENT ABATEMENT. Neither Landlord nor "
            "Tenant shall be deemed in default with respect to non-monetary covenants delayed "
            "by Force Majeure events beyond reasonable control; provided, however, Force Majeure "
            "shall not excuse Tenant's timely payment of Base Rent or Additional Operating "
            "Expenses unless governmental condemnation or casualty renders more than thirty-five "
            "percent (35%) of the rentable square footage untenantable for ninety consecutive days."
        ),
    },
    {
        "id": "leg_10",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "ASSET PURCHASE AGREEMENT — MATERIAL ADVERSE EFFECT & EARNOUT COVENANTS. Between "
            "the Signing Date and Closing, Target shall operate strictly in the Ordinary Course "
            "of Business and confirm absence of any Material Adverse Effect excluding industry-wide "
            "macroeconomic shifts not disproportionately affecting Target. Buyer shall calculate "
            "FY2027 Adjusted EBITDA in accordance with GAAP consistently applied and refrain from "
            "bad-faith actions whose primary purpose is frustrating achievement of the Earnout Milestone."
        ),
    },
    {
        "id": "leg_11",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "SENIOR SECURED CREDIT AGREEMENT — FINANCIAL COVENANTS & EVENTS OF DEFAULT. Borrower "
            "shall maintain as of the last day of each fiscal quarter a Consolidated Total Net "
            "Leverage Ratio not exceeding 3.75 to 1.00 and a Fixed Charge Coverage Ratio of at "
            "least 1.25 to 1.00. Occurrence of any Cross-Default on Indebtedness exceeding "
            "$5,000,000 or insolvency filing under Title 11 of the Bankruptcy Code shall permit "
            "Administrative Agent to accelerate all unpaid principal and exercise UCC Article 9 remedies."
        ),
    },
    {
        "id": "leg_12",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "SOURCE CODE ESCROW RELEASE CONDITIONS & BANKRUPTCY SECTION 365(n). Licensor shall "
            "deposit annotated human-readable Source Code and build scripts with Iron Mountain "
            "Escrow Agent within fifteen (15) days of each major release. All rights and licenses "
            "granted hereunder are, for purposes of Section 365(n) of the United States Bankruptcy "
            "Code, licenses of rights to 'intellectual property,' entitling Licensee to retain "
            "its rights and demand immediate Escrow Deposit release upon Licensor liquidation."
        ),
    },
    {
        "id": "leg_13",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "NON-SOLICITATION, INVENTION ASSIGNMENT & CALIFORNIA LABOR CODE § 2870 NOTICE. "
            "Consultant hereby irrevocably assigns to Company all right, title, and interest "
            "worldwide in Moral Rights and Company Inventions conceived during the Engagement, "
            "excluding Prior Inventions listed on Exhibit A and inventions developed entirely "
            "on Consultant's own time without Company equipment or trade secrets as protected "
            "by California Labor Code Section 2870. For twelve months post-termination, Consultant shall not solicit employees."
        ),
    },
    {
        "id": "leg_14",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "MOST FAVORED NATION (MFN) PRICING & AUDIT RIGHTS. Vendor warrants that the unit "
            "rates and enterprise discount tiers extended to Purchaser under Order Form #7 are "
            "no less favorable than those granted to any commercial customer purchasing comparable "
            "annual committed volumes. Should Vendor offer lower net effective pricing during "
            "the Term, such reduced rates shall apply retroactively, verifiable once annually "
            "by an independent Big Four accounting firm upon thirty (30) days prior written notice."
        ),
    },
    {
        "id": "leg_15",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "EXPORT CONTROL, OFAC SANCTIONS & FCPA ANTI-CORRUPTION COMPLIANCE. Counterparty "
            "represents and warrants that neither it nor any beneficial owner owning ten percent "
            "(10%) or more is designated on the U.S. Treasury OFAC Specially Designated Nationals "
            "List, nor will it re-export technical deliverables in violation of the Export "
            "Administration Regulations (15 C.F.R. Parts 730-774) or offer anything of value "
            "to any foreign official in violation of the U.S. Foreign Corrupt Practices Act."
        ),
    },
    {
        "id": "leg_16",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "RIGHT OF FIRST REFUSAL AND CO-SALE AGREEMENT — DRAG-ALONG OBLIGATIONS. In the event "
            "that Holders of a majority of the Common Stock and Requisite Preferred Holders, "
            "voting separately, approve a Sale of the Corporation in an arm's-length transaction, "
            "each Key Holder and Investor agrees to vote all Shares in favor of such Sale, waive "
            "any statutory appraisal or dissenters' rights under DGCL Section 262, and execute "
            "ancillary agreements subject to pro rata several liability capped at net proceeds received."
        ),
    },
    {
        "id": "leg_17",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "MASTER PROFESSIONAL SERVICES AGREEMENT — WORK MADE FOR HIRE & BACKGROUND IP. All "
            "Custom Deliverables specially commissioned and paid for under a Statement of Work "
            "shall constitute 'works made for hire' under 17 U.S.C. § 101, and to the extent "
            "not qualifying as such, Vendor assigns all copyrights therein to Client. Vendor "
            "retains sole ownership of Pre-Existing Background Tools and grants Client a perpetual, "
            "irrevocable, paid-up, worldwide license to use and modify Background Tools embedded in Deliverables."
        ),
    },
    {
        "id": "leg_18",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "INSURANCE MINIMUMS, SUBROGATION WAIVER & ADDITIONAL INSURED ENDORSEMENT. Throughout "
            "the Term and for three (3) years thereafter, Supplier shall maintain with A.M. Best "
            "A-VIII rated carriers: (a) Commercial General Liability of $5,000,000 per occurrence, "
            "(b) Technology Errors & Omissions and Cyber Liability covering network security "
            "and privacy regulatory fines of $10,000,000 aggregate, naming Enterprise Customer "
            "as Additional Insured on a primary non-contributory basis with waiver of subrogation."
        ),
    },
    {
        "id": "leg_19",
        "category_idx": 4,
        "category": "Legal Contract",
        "prompt": (
            "ASSIGNMENT, ANTI-DELEGATION & TERMINATION FOR CONVENIENCE OR INSOLVENCY. Neither "
            "Party may assign or transfer this Agreement by operation of law or otherwise without "
            "prior written consent, except to a successor acquiring substantially all assets "
            "that is not a Direct Competitor listed on Schedule D; any purported assignment in "
            "contravention hereof is void ab initio. Customer may terminate any Order Form for "
            "convenience upon sixty (60) days written notice with pro rata refund of unearned prepaid fees."
        ),
    },
]

# General-domain multi-topic text corpus for measuring exact LM Perplexity (PPL) and ΔPPL
PPL_EVAL_CORPUS: list[str] = [
    (
        "Distributed consensus algorithms such as Raft and Paxos guarantee linearizable state "
        "machine replication across unreliable networks by electing a distinguished leader and "
        "committing log entries only after a majority quorum persists them to stable storage."
    ),
    (
        "Photosynthesis in vascular plants couples light-driven electron transport across the "
        "thylakoid membrane to generate ATP and NADPH with the Calvin-Benson cycle in the stroma, "
        "fixing atmospheric carbon dioxide into triose phosphate sugars via RuBisCO catalysis."
    ),
    (
        "During the Renaissance, Florentine workshops transformed European visual art through "
        "linear mathematical perspective, anatomical dissection studies, and oil glaze layering "
        "that enabled subtle chiaroscuro transitions between illuminated surfaces and shadow."
    ),
    (
        "Modern compiler backends lower static single assignment intermediate representations "
        "through instruction selection, graph-coloring register allocation, and modulo loop "
        "pipelining to saturate superscalar out-of-order execution units and vector pipelines."
    ),
    (
        "Oceanic thermohaline circulation transports warm surface waters poleward into the North "
        "Atlantic, where evaporative cooling and sea-ice brine rejection increase seawater density, "
        "driving deep water formation that ventilates abyssal ocean basins over centuries."
    ),
    (
        "Central banks calibrate monetary policy transmission by adjusting overnight policy interest "
        "rates and reserve balances, influencing commercial credit creation, yield curve expectations, "
        "exchange rate dynamics, and medium-term consumer price inflation trajectories."
    ),
    (
        "Mixture-of-Experts transformer architectures decouple parameter capacity from per-token "
        "floating-point compute by routing each hidden token representation through a sparse top-k "
        "gating network to a small subset of specialized feed-forward expert subnetworks."
    ),
    (
        "High-speed railway viaducts in seismically active corridors incorporate base-isolated "
        "elastomeric bearings, ductile reinforced concrete piers, and tuned mass dampers to "
        "dissipate ground motion energy while preserving millimeter-level track geometry."
    ),
]
