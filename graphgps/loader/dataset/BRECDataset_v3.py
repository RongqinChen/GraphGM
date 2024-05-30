import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.utils.convert import from_networkx
import os
import shutil
from tqdm import tqdm

torch_geometric.seed_everything(2022)

part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.url = "https://raw.githubusercontent.com/GraphPKU/BREC/Release/BREC_data_all.zip"
        self.root = root
        self.name = "BRECv3"
        super().__init__(root, transform, None, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if pre_transform is not None:
            data_list = [
                pre_transform(data) for data in self
            ]
            self._data_list = data_list

        print('')

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3.pt"]

    def process(self):

        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data_list = [graph6_to_pyg(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    dataset = BRECDataset()
    print(len(dataset))


if __name__ == "__main__":
    main()
