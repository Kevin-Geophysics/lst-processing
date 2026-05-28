def report(result):

    bands_processed = result["bands_processed"]

    has_band_m = "M15" in bands_processed   # Report untuk band M
    has_band_i = "I5" in bands_processed    # Report untuk band I

    if has_band_m:

        s = result["anomaly"]["stats"]

        print(f"  LST M15+M16 : {s['lst_min_celsius']:.1f} – {s['lst_max_celsius']:.1f} °C")
        print(f"  Mean        : {s['lst_mean_celsius']:.1f} °C")
        print(f"  Hot pixels  : {s['hot_pixels']} piksel (Z > 2)")

    if has_band_i:

        a = result.get("anomaly_i5") or result.get("anomaly")
        s = a["stats"]

        print(f"  LST I5      : {s['lst_min_celsius']:.1f} – {s['lst_max_celsius']:.1f} °C")
        print(f"  Mean I5     : {s['lst_mean_celsius']:.1f} °C")