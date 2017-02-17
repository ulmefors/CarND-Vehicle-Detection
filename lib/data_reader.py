import matplotlib.image as mpimg
import glob

DATA_DIR = 'data/'
CAR_SMALL = 'vehicles_smallset/'
NON_CAR_SMALL = 'non-vehicles_smallset/'
CAR = 'vehicles/'
NON_CAR = 'non-vehicles/'


def read_data(small_sample=True, nb_data=0):
    if small_sample:
        non_car_files = glob.glob(DATA_DIR + NON_CAR_SMALL + 'notcars*/*.jpeg')
        car_files = glob.glob(DATA_DIR + CAR_SMALL + 'cars*/*.jpeg')
    else:
        non_car_files = glob.glob(DATA_DIR + NON_CAR + '*/*.png')
        car_files = glob.glob(DATA_DIR + CAR + '*/*.png')

    if nb_data > 0:
        if nb_data < len(car_files):
            car_files = car_files[0:nb_data]
        if nb_data < len(non_car_files):
            non_car_files = non_car_files[0:nb_data]
    return car_files, non_car_files


def main():
    pass


if __name__ == "__main__":
    main()
