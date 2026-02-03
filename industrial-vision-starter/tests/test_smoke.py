from ivs.utils import TrainConfig
from ivs.train import train

def test_train_smoke(tmp_path):
    cfg = TrainConfig(epochs=1, train_steps=5, val_steps=2, batch_size=2, out_dir=str(tmp_path))
    ckpt = train(cfg)
    assert ckpt.exists()
