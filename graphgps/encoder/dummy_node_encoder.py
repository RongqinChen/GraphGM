import torch
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('DummyNode')
class DummyNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        dummy_attr = batch.edge_index.new_zeros((batch.num_nodes,))
        batch.x = self.encoder(dummy_attr)
        return batch
