-- ══════════════════════════════════════════════════════════════════════════════
-- Supabase setup — run this once in the SQL Editor
-- ══════════════════════════════════════════════════════════════════════════════

-- 1. Create predictions table
-- ──────────────────────────────────────────────────────────────────────────────
CREATE TABLE predictions (
    id           uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at   timestamptz DEFAULT now(),
    updated_at   timestamptz DEFAULT now(),
    update_count int DEFAULT 1,
    ticker       text NOT NULL,
    trade_date   date NOT NULL,
    predicted    int,          -- +1 (up) or -1 (down)
    actual       int,          -- filled next day: +1, -1
    correct      boolean,      -- filled next day: true/false
    confidence   float,        -- model probability 0-1
    sentiment    float,        -- FinBERT mean score
    retrained    boolean DEFAULT false
);

-- 2. Indexes
-- ──────────────────────────────────────────────────────────────────────────────
-- Unique constraint: one row per ticker per trading day
CREATE UNIQUE INDEX idx_predictions_unique    ON predictions(ticker, trade_date);
CREATE INDEX         idx_predictions_retrained ON predictions(retrained);
CREATE INDEX         idx_predictions_date      ON predictions(trade_date);
CREATE INDEX         idx_predictions_ticker    ON predictions(ticker);

-- 3. Row Level Security
-- ──────────────────────────────────────────────────────────────────────────────
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- App can read all rows
CREATE POLICY "app_select" ON predictions
    FOR SELECT TO anon USING (true);

-- App can insert new predictions
CREATE POLICY "app_insert" ON predictions
    FOR INSERT TO anon WITH CHECK (true);

-- App can update predictions (fill in actuals, bump update_count)
CREATE POLICY "app_update" ON predictions
    FOR UPDATE TO anon USING (true);

-- App can ONLY delete retrained rows (protects unprocessed predictions)
CREATE POLICY "app_delete" ON predictions
    FOR DELETE TO anon USING (retrained = true);

-- 4. Automated cleanup via pg_cron
-- ──────────────────────────────────────────────────────────────────────────────
-- Runs at 2am UTC daily (before any market opens globally).
-- Deletes retrained rows older than 30 days.
-- This runs even when nobody is using the app.

CREATE EXTENSION IF NOT EXISTS pg_cron;

SELECT cron.schedule(
    'daily-predictions-cleanup',          -- job name
    '0 2 * * *',                          -- 2:00 AM UTC every day
    $$
        DELETE FROM predictions
        WHERE retrained = true
        AND created_at < NOW() - INTERVAL '7 days';
    $$
);

-- To verify the job was created:
-- SELECT * FROM cron.job;

-- To remove the job if needed:
-- SELECT cron.unschedule('daily-predictions-cleanup');

-- 5. Optional: check row count at any time
-- ──────────────────────────────────────────────────────────────────────────────
-- SELECT
--     COUNT(*)                                          AS total_rows,
--     COUNT(*) FILTER (WHERE retrained = true)          AS retrained_rows,
--     COUNT(*) FILTER (WHERE actual IS NULL)            AS pending_actuals,
--     COUNT(*) FILTER (WHERE actual IS NOT NULL
--                      AND retrained = false)           AS ready_to_retrain
-- FROM predictions;
