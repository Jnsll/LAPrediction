
import helpers
import flopy.utils.binaryfile as fpu
import argparse


def get_evolution_of_head_level_for_a_cell(folder, site_number, chronicle, approx, rate, row, col, ref):
    simu_name = helpers.get_model_name(site_number, chronicle, approx, rate, ref=ref, perm=False)
    site_name = helpers.get_site_name_from_site_number(site_number)
    repo_simu = folder + site_name + simu_name
    topoSimu = helpers.get_soil_surface_values_for_a_simulation(repo_simu, simu_name)
    simu_hds = fpu.HeadFile(repo_simu + '/' + simu_name + '.hds')

    endTime = int(simu_hds.get_times()[-1])#15341
    print(endTime)

    #for day in range(endTime):




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-site", "--sitenumber", type=int, required=True)
    parser.add_argument("-chr", "--chronicle", type=int, required=True)
    parser.add_argument("-approx", "--approx", type=int, required=False)
    parser.add_argument("-rate", "--rate", type=int, required=False)
    parser.add_argument("-f", "--folder", type=str, required=True)
    parser.add_argument("-r", "--row", type=int, required=True)
    parser.add_argument("-c", "--col", type=int, required=True)
    parser.add_argument("-ref", "-ref", action='store_true')
    args = parser.parse_args()

    site_number = args.sitenumber
    chronicle = args.chronicle
    approx = args.approx
    rate = args.rate
    folder = args.folder
    row = args.row
    col = args.col
    ref = args.ref

    get_evolution_of_head_level_for_a_cell(folder, site_number, chronicle, approx, rate, row, col, ref)