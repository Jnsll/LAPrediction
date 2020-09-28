import pandas as pd
import csv

df_geomorph = pd.DataFrame(columns=["Site", "Slope", "Elevation", "LC", "CW", "Area"])
folder = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results"

for site in range(1, 31):

    with open(
        "/DATA/These/OSUR/Extract_BV_june/" + str(site) + "_slope_elevation_Lc_Cw_A",
        newline="",
    ) as f:
        reader = csv.reader(f)
        geomorph = list(reader)

    slope = float(geomorph[0][0])
    elev = float(geomorph[0][1])
    Lc = float(geomorph[0][2])
    Cw = float(geomorph[0][3])
    A = float(geomorph[0][4])

    df_geomorph = df_geomorph.append(
        {
            "Site": int(site),
            "Slope": slope,
            "Elevation": elev,
            "LC": Lc,
            "CW": Cw,
            "Area": A,
        },
        ignore_index=True,
    )
print(df_geomorph)

df_geomorph.to_csv(
    folder + "/Geomorph_Features_All_Sites_Saturation.csv", sep=";", index=False
)
