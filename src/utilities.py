import numpy as np

def generate_random_data_point(n, range_x=(0.0, 1.0), range_y=(0.0, 1.0), range_z=(0.0, 1.0)):
    """ Generate n number of random data points
    """
    data = np.random.rand(n, 3)
    data[:,0] = range_x[0] + abs(range_x[1]-range_x[0])*data[:,0]  # x values
    data[:,1] = range_y[0] + abs(range_y[1]-range_y[0])*data[:,1]  # y values
    data[:,2] = range_z[0] + abs(range_z[1]-range_z[0])*data[:,2]  # z values

    return data


def save_data_as_csv(data, filename='sample_data.csv', header="X, Y, Z"):
    """ Saves data as csv data file"""
    np.savetxt(filename, data, delimiter=',', fmt='%.2f', header=header)
    #np.savetxt(filename, data, delimiter=',', fmt='%.2f')



if __name__ == '__main__':
    # known data
    data = generate_random_data_point(10, range_x=(0, 20.0), range_y=(0, 30,0), range_z=(8.0, 19.0))
    save_data_as_csv(data, 'sample_known_data.csv')

    # unknown data
    data_unknown = generate_random_data_point(5, range_x=(0, 20.0), range_y=(0, 30,0), range_z=(8.0, 19.0))
    save_data_as_csv(data_unknown[:,:-1], 'sample_unknown_data.csv', header='X, Y')