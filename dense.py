import faiss
import torch


GRAPH_EMBEDDING_PATH = "graph_id_to_embedding.pt"
TRAIN_SET_SIZE = 18889


graph_id_to_embedding = torch.load(GRAPH_EMBEDDING_PATH)


graph_embeddings = [graph_id_to_embedding[i+1] for i in range(TRAIN_SET_SIZE)]

graph_embeddings = torch.stack(graph_embeddings, dim=0)

print(graph_embeddings.shape)


d = graph_embeddings.shape[1]

index = faiss.IndexFlatL2(d)

print(index.is_train)

index.add(graph_embeddings)

index.ntotal

k = 5

D, I = index.search()
