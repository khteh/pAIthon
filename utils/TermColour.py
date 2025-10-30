class bcolors:
    """
    https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    https://stackoverflow.com/questions/67241111/python-colored-text-to-the-terminal
    | Color                                  | Font code        | Background code  |
    |----------------------------------------|------------------|------------------|
    | Black                                  | \x1B[30m         | \x1B[40m         |
    | Red                                    | \x1B[31m         | \x1B[41m         |
    | Green                                  | \x1B[32m         | \x1B[42m         |
    | Yellow                                 | \x1B[33m         | \x1B[43m         |
    | Blue                                   | \x1B[34m         | \x1B[44m         |
    | Magenta                                | \x1B[35m         | \x1B[45m         |
    | Cyan                                   | \x1B[36m         | \x1B[46m         |
    | White                                  | \x1B[37m         | \x1B[47m         |
    | Any palette color (with V in [0-255])  | \x1B[38;5;Vm     | \x1B[48;5;Vm     |
    | Any RGB color (with values in [0-255]) | \x1B[38;2;R;G;Bm | \x1B[48;2;R;G;Bm |    
    """
    DEFAULT = '\x1b[0m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'    
    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKCYAN = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''
        self.BOLD = ''
        self.UNDERLINE = ''