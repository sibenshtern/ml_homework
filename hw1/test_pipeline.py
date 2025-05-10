import os
import json
from pathlib import Path

import pytest
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from train import main as train_main
from compute_metrics import main as metrics_main
from prepare_data import prepare_train, prepare_test
from hparams import config


@pytest.fixture()
def train_dataset():
    # note: реализуйте и протестируйте подготовку данных (скачиание и препроцессинг)
    train_directory = Path("CIFAR10/train")
    test_directory = Path("CIFAR10/test")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.Resize((224, 224))
    ])

    if not train_directory.exists():
        train_dataset = prepare_train()
    else:
        train_dataset = CIFAR10(root='CIFAR10/train', download=False,
                                transform=transform)
    
    if not test_directory.exists():
        prepare_test()

    assert CIFAR10(root='CIFAR10/train', download=False, transform=transform)
    assert CIFAR10(root='CIFAR10/test', train=False, download=False,
                   transform=transform)

    return train_dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"], ["mps"]])
def test_train_on_one_batch(device, train_dataset):
    # note: реализуйте и протестируйте один шаг обучения вместе с метрикой
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    default_epochs = config['epochs']

    config['epochs'] = 1
    config['one_batch'] = True
    try:
        train_main(device, train_dataset)
    finally:
        config['epochs'] = default_epochs
        config['one_batch'] = False


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"], ["mps"]])
def test_training(device, train_dataset):
    # note: реализуйте и протестируйте полный цикл обучения модели (обучение, валидацию, логирование, сохранение артефактов)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    model_file = Path("model.pt")
    run_id_file = Path("run_id.txt")
    metrics_file = Path("metrics.json")

    if model_file.exists():
        os.remove(model_file)
    if run_id_file.exists():
        os.remove(run_id_file)

    train_main(device, train_dataset)

    assert model_file.exists(), "Модель не сохранилась в model.pt"

    state = torch.load(model_file, map_location=device)
    assert isinstance(state, dict), "Содержимое model.pt не является dict"

    with open(run_id_file) as file:
        run_id = file.readline().strip()
        assert len(run_id) > 0, "Файл run_id.txt пустой"

    metrics_main(device)

    with open(metrics_file) as file:
        metrics = json.load(file)
        assert "accuracy" in metrics, "Accuracy отсутствует в файле"
        assert 0 <= metrics["accuracy"] <= 1, "Accuracy некорректна"
