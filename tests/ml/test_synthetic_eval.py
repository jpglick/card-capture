"""Tests for src/card_capture/ml/synthetic_eval.py"""
from card_capture.ml.synthetic_eval import generate_dedup_dataset, generate_fb_dataset


def test_fb_synthetic_balanced(tmp_path):
    ds = generate_fb_dataset(out_dir=tmp_path, n_per_class=20, seed=7)
    assert len(ds) == 40
    fronts = [item for item in ds if item.label == "front"]
    backs = [item for item in ds if item.label == "back"]
    assert len(fronts) == 20 and len(backs) == 20


def test_fb_images_exist(tmp_path):
    ds = generate_fb_dataset(out_dir=tmp_path, n_per_class=5, seed=0)
    for item in ds:
        assert item.image_path.exists()


def test_dedup_synthetic_clusters(tmp_path):
    ds = generate_dedup_dataset(out_dir=tmp_path, n_clusters=5, samples_per_cluster=4, seed=7)
    assert len(ds.items) == 20
    cluster_sizes = {cid: 0 for cid in ds.cluster_ids}
    for item in ds.items:
        cluster_sizes[item.cluster_id] += 1
    assert all(v == 4 for v in cluster_sizes.values())


def test_dedup_images_exist(tmp_path):
    ds = generate_dedup_dataset(out_dir=tmp_path, n_clusters=3, samples_per_cluster=2, seed=1)
    for item in ds.items:
        assert item.image_path.exists()
