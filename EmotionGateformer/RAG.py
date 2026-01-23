import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def build_split_isolated_rag(master_json, arc_dir, output_dir, top_k=100):
    os.makedirs(output_dir, exist_ok=True)

    with open(master_json, 'r') as f:
        master_cfg = json.load(f)
    
    train_scene_ids = set()
    for movie in master_cfg['movies']:
        train_scene_ids.update(movie.get('train', []))
    
    print(f" Identified training scenes count: {len(train_scene_ids)}")

    all_files = sorted([f for f in os.listdir(arc_dir) if f.endswith('.npy')])
    
    gallery_files = []
    query_files = all_files

    for f in all_files:
        scene_id = f.split('@')[0] 
        if scene_id in train_scene_ids:
            gallery_files.append(f)

    print(f" Gallery scale (Training set only): {len(gallery_files)}")
    print(f" Total query count: {len(query_files)}")

    def load_vectors(file_list):
        vecs = []
        for f in tqdm(file_list, desc="Loading features"):
            vec = np.load(os.path.join(arc_dir, f))
            vecs.append(vec.flatten())
        return np.array(vecs)

    gallery_vecs = load_vectors(gallery_files)
    query_vecs = load_vectors(query_files)

    print("Calculating similarity matrix (Query x Gallery)...")
    sim_matrix = cosine_similarity(query_vecs, gallery_vecs)

    for i, q_file in enumerate(tqdm(query_files, desc="Generating indices")):
        q_basename = q_file.split("-arc-")[1].replace('.npy', '')
        
        scores = sim_matrix[i].copy()
        
        if q_file in gallery_files:
            self_idx_in_gallery = gallery_files.index(q_file)
            scores[self_idx_in_gallery] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        
        result_list = []
        for idx in top_indices:
            ref_emotion_name = gallery_files[idx].replace("-arc-", "-emotion-")
            
            result_list.append({
                "file_name": ref_emotion_name,
                "score": float(scores[idx]),
                "ref_arc_name": gallery_files[idx]
            })
            
        output_name = f"{q_basename}-arc_top_100.json"
        with open(os.path.join(output_dir, output_name), 'w') as jf:
            json.dump(result_list, jf, indent=2)

    print(f"Task completed! Indices saved in: {output_dir}")

if __name__ == "__main__":
    build_split_isolated_rag(
        master_json = "Dataset/data/dataset.json",
        arc_dir = "Dataset/preprocessed_data/features/arc",
        output_dir = "Dataset/preprocessed_data/features/rag_indices",
        top_k = 100
    )