import re
from typing import List, Dict

from matplotlib import pyplot as plt


def plot_cities(cities_dict: Dict, test:bool = False) -> None:
    """
    plot with matplotlib the parsed TSP file
    :param test: if testing is True
    :param cities_dict: {'1': (38.24, 20.42), '2': (39.57, 26.15),...}
    :return: NADA plot cities coordinates
    """
    plt.clf()
    plt.axis = (0.0, 1.0, 0.0, 1.0)
    sorted_tuples = sorted(cities_dict.values(), key=lambda tup: tup[1])
    plt.scatter(*zip(*sorted_tuples), s=40, zorder=1)
    plt.plot(*zip(*cities_dict.values()), 'm--', zorder=0)
    if test is True:
        return plt.axis  # for the test
    plt.show()


def check_filename_tsp(filename: str) -> bool:
    """
        Check if the file provided is a valid TSP file
        ...ends with .tsp
    """
    print(filename)
    if filename.endswith(".tsp"):
        return True
    else:
        return False


def read_tsp_file_contents(filename: str) -> List:
    """
    gets the contents of the file into a list
    :param filename: the filename to read
    :return: a list like ['NAME: ulysses16.tsp',
                        'TYPE: TSP COMMENT: Odyssey of Ulysses (Groetschel/Padberg)',
                        'DIMENSION: 16',
                        'EDGE_WEIGHT_TYPE: GEO',
                        'DISPLAY_DATA_TYPE: COORD_DISPLAY',
                        'NODE_COORD_SECTION',
                        '1 38.24 20.42',
                        '2 39.57 26.15',..., 'EOF']
    """
    with open(filename) as f:
        content = [line.strip() for line in f.read().splitlines()]
        return content


class TSPParser:
    """
    TSP file parser to turn the TSP problem file into a dictionary of type
    [[38.24, 20.42], [39.57, 26.15]]

    Usage like
    TSPParser(filename=file_name)
    print(TSPParser.tsp_cities_list)
    """
    should_plot: bool = False
    filename: str = None
    subscribed: bool = False
    dimension: int = None
    tsp_file_contents: List = []
    tsp_cities_list: List = []

    @classmethod
    def __init__(cls, filename: str, plot_tsp: bool) -> None:
        cls.clear_data()
        cls.filename = filename
        cls.should_plot = plot_tsp
        cls.on_file_selected()

    @classmethod
    def on_file_selected(cls) -> None:
        """
        internal use when instantiating class object
        :return: NADA goes to open the file
        """
        cls.open_tsp_file()

    @classmethod
    def open_tsp_file(cls) -> None:
        """
        if file is tsp will read the contents
        :return: NADA assign to tsp_file_contents
        """
        if not check_filename_tsp(cls.filename):
            # TODO raise a custom exception
            print("File is not TSP file")
        else:
            cls.tsp_file_contents = read_tsp_file_contents(cls.filename)
            cls.detect_dimension()

    @classmethod
    def detect_dimension(cls) -> None:
        """
        finds the list element that starts with DIMENSION and gets the int
        :return: NADA goes to get the dict
        """
        for record in cls.tsp_file_contents:
            if record.startswith("DIMENSION"):
                parts = record.split(":")
                cls.dimension = int(parts[1])
                print(cls.dimension)
        cls.get_cities_dict()

    @classmethod
    def get_cities_dict(cls) -> None:
        """
        zero index is the index in the contents list where city coordinates starts
        last index of parser is the zero index + dimension of the file
        at the end plots if asked
        :return: NADA assign to tsp_cities_list list like [[38.24, 20.42], [39.57, 26.15]]
        """
        zero_index = cls.tsp_file_contents.index("NODE_COORD_SECTION") + 1
        for index in range(zero_index, zero_index + cls.dimension):
            parts = cls.tsp_file_contents[index].strip()
            city_coords_parts = re.findall(r"[+-]?\d+(?:\.\d+)?", parts)
            # print(city_coords_parts)
            cls.tsp_cities_list.append([float(city_coords_parts[1]), float(city_coords_parts[2])])
        if cls.should_plot:
            plot_cities(cls.tsp_cities_list)

    @classmethod
    def clear_data(cls) -> None:
        """
        re-use the class
        :return: NADA
        """
        cls.filename = ""
        cls.tsp_cities_list = []
        cls.tsp_file_contents = []
        cls.dimension = 0
