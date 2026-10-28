"""Real series for post-training, from the GIFT-Eval pretraining split, minus anything a benchmark draws on.

GIFT-Eval built `Salesforce/GiftEvalPretrain` without its own test series. fev-bench
and TIME were not part of that, so a dataset is kept only when its source does not
appear in either of them. Left out for that reason:

- fev-bench: favorita_sales and favorita_transactions (favorita), m5 (m5), gfc12_load,
  gfc14_load and gfc17_load (proenfo_gfc), kdd2022 (kdd_cup_2022), fred_md (fred_md_2025),
  spain (entsoe), tourism_monthly, tourism_quarterly and tourism_yearly (australian_tourism).
- TIME: australian_electricity_demand, elecdemand, solar_power, wind_power and
  wind_farms_with_missing (OpenElectricity_NEM, Australia_Solar), azure_vm_traces_2017
  (azure2019), bitcoin_with_missing (Crypto), cdc_fluview_ilinet and cdc_fluview_who_nrevss
  (Global_Influenza), weather and oikolab_weather (Australian weather, beside GIFT-Eval's
  temperature_rain).

The climate grids (era5, cmip6) and the multi-gigabyte traces are left out for size.
"""

REAL_DATASETS = (
    "BEIJING_SUBWAY_30MIN",
    "HZMETRO",
    "SHMETRO",
    "LOS_LOOP",
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
    "PEMS_BAY",
    "traffic_hourly",
    "traffic_weekly",
    "taxi_30min",
    "uber_tlc_daily",
    "uber_tlc_hourly",
    "rideshare_with_missing",
    "vehicle_trips_with_missing",
    "pedestrian_counts",
    "bdg-2_bear",
    "bdg-2_fox",
    "bdg-2_panther",
    "bdg-2_rat",
    "bull",
    "cockatoo",
    "hog",
    "ideal",
    "lcl",
    "sceaux",
    "smart",
    "pdb",
    "borealis",
    "elf",
    "covid19_energy",
    "london_smart_meters_with_missing",
    "beijing_air_quality",
    "china_air_quality",
    "subseasonal",
    "subseasonal_precip",
    "sunspot_with_missing",
    "project_tycho",
    "covid_mobility",
    "godaddy",
    "kaggle_web_traffic_weekly",
    "wiki-rolling_nips",
    "nn5_daily_with_missing",
    "nn5_weekly",
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    "monash_m3_monthly",
    "monash_m3_quarterly",
    "monash_m3_yearly",
    "monash_m3_other",
    "cif_2016_6",
    "cif_2016_12",
)

# The sources above whose series have several related variates, long enough to forecast.
REAL_MULTIVARIATE = (
    "BEIJING_SUBWAY_30MIN",
    "HZMETRO",
    "SHMETRO",
    "PEMS04",
    "PEMS08",
    "beijing_air_quality",
    "china_air_quality",
    "subseasonal",
)
