import pandas as pd
import csv


def Geomorph_crits_file():
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


def Geomoph_crits_sub_file():
    df_geomorph = pd.DataFrame(columns=["Site", "SubCatch", "Slope", "Elevation", "LC", "SAR", "Area", "CV", "HV"])
    folder = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/"

    for site in range(1, 41):
        try:
            with open("/DATA/These/Projects/AWDE/output/" + str(site) + "_slope_elevation_Lc_SAR_A_CV_HV",
                newline="",) as f:
                reader = csv.reader(f)
                lines = list(reader)
        except:
            print("Not file for site number: ", site)
            continue
        sub = 1
        for line in lines:
            crits = line[0].split()
            df_geomorph = df_geomorph.append(
            {
                "Site": int(site),
                "SubCatch": sub,
                "Slope": crits[0],
                "Elevation": crits[1],
                "LC": crits[2],
                "SAR": crits[3],
                "Area": crits[4],
                "CV": crits[5],
                "HV": crits[6],
            },
            ignore_index=True,)
            sub += 1


    df_geomorph.to_csv(folder + "Geomorph_Features_All_Sites_Saturations_SubCatchbis.csv", index=False)


if __name__ == "__main__":
    Geomoph_crits_sub_file()