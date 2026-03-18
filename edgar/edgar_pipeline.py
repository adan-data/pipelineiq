"""
EDGAR Pipeline — pulls real quarterly revenue for public SaaS comps.
No API key required. Uses SEC EDGAR public JSON endpoints.
"""
import requests, pandas as pd, time

COMPANIES = {
    "HubSpot":   "0001404655",
    "Datadog":   "0001677250",
    "Snowflake": "0001640147",
}

def get_revenue(cik, company_name, concept="Revenues"):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    headers = {"User-Agent": "pulsemetrics analytics@pulsemetrics.io"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        facts = data.get("facts",{}).get("us-gaap",{})
        for tag in [concept, "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]:
            if tag in facts:
                entries = facts[tag].get("units",{}).get("USD",[])
                qtrs = [e for e in entries if e.get("form") in ["10-Q","10-K"] and e.get("fp","")!="FY"]
                df = pd.DataFrame(qtrs)[["end","val","form","fp"]].rename(columns={"end":"date","val":"revenue"})
                df["company"] = company_name
                df["revenue_m"] = df["revenue"] / 1e6
                return df.sort_values("date").tail(12)
    except Exception as e:
        print(f"  Error fetching {company_name}: {e}")
    return pd.DataFrame()

if __name__ == "__main__":
    all_data = []
    for name, cik in COMPANIES.items():
        print(f"Fetching {name}...")
        df = get_revenue(cik, name)
        if len(df): all_data.append(df)
        time.sleep(0.5)  # be polite to SEC servers
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv("data/processed/edgar_benchmarks.csv", index=False)
        print(combined.groupby("company")["revenue_m"].describe().round(1))
    else:
        print("No data retrieved — check network connectivity")
