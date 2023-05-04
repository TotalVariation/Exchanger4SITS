import os
import tarfile
from glob import glob


def unzipper(rootdir):
    '''
    THIS FUNCTION DEFINE AN UNZIPPER FOR TAR.GZ FILES IN A DIRECTORY.
    :param rootdir: where  the compressed files are located
    :return: None
    '''

    inputs = glob(rootdir + '/*.tar.gz', recursive=True)
    for input in inputs:
        datadir = os.path.dirname(input)
        opener, mode = tarfile.open, 'r:gz'
        rootpath = input.replace(".tar.gz", "")

        if not (os.path.exists(rootpath) and os.path.isdir(rootpath)):
            print(f"INFO: Unzipping {input} to {datadir}")
            file = opener(input, mode)
            try:
                file.extractall(datadir)
            finally:
                file.close()
        else:
            print(f"INFO: Found folder in {rootpath}, no need to unzip")


if __name__ == '__main__':
    """
    EXAMPLE USAGE OF UNZIPPER
    """

    ziproot = "../data/"
    unzipper(ziproot)
