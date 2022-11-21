import os
from pathlib import Path
import zipfile
import wget


class PadUfes20Data:
    def __init__(self, root, download:bool=False):
        self.root = Path(root)
        if download:
            self.download()

    def _get_data_folder(self):
        # todo create checker for data path uniqueness
        return self.root / 'PadUfes20'

    def _check_exists(self):
        # todo: implement more thorough checks on files
        return self._get_data_folder().exists()
    
    def download(self):
        if self._check_exists():
            return

        fldr = self._get_data_folder()
        fldr.mkdir(parents=True, exist_ok=False)

        zip_pt = fldr / 'tmp.zip'
        zip_pt = str(zip_pt.absolute())
        link = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip'
        wget.download(url=link, out=zip_pt)

        files_fldr = fldr / 'files'
        with zipfile.ZipFile(zip_pt, 'r') as zip_ref:
            zip_ref.extractall(files_fldr)

        for pt in list((files_fldr / 'images').iterdir()):
            with zipfile.ZipFile(pt, 'r') as zip_ref:
                zip_ref.extractall(pt.parent)
            os.remove(pt)

        os.remove(zip_pt)
        assert self._check_exists(), "Problems with downloaded files"

        
