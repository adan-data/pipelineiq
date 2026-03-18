-- ============================================================
-- Query 07: CAC Breakeven Risk by Tier
-- Purpose : Shows which tiers are destroying capital by failing
--           to survive to their CAC payback month.
--           The most financially consequential query in the project.
-- ============================================================
SELECT
    b.tier,
    b.cac,
    b.monthly_gp,
    b.breakeven_mo,
    b.pct_survive_to_breakeven,
    b.expected_net_value,
    CASE
        WHEN b.expected_net_value < 0    THEN 'DESTROYING CAPITAL'
        WHEN b.expected_net_value < 50   THEN 'MARGINALLY PROFITABLE'
        ELSE                                  'PROFITABLE'
    END AS capital_status,
    k.churn_rate,
    k.S_6mo,
    k.S_12mo
FROM cac_breakeven b
LEFT JOIN km_survival k ON b.tier = k.tier
ORDER BY b.expected_net_value ASC;
