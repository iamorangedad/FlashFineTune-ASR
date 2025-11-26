import torch
import torch.nn as nn
from model.asr import ASRModel


def test_model_train():
    N_MELS = 20
    VOCAB_SIZE = 100
    model = ASRModel(
        n_mels=N_MELS,
        d_model=64,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
    )
    batch_size = 2
    audio_len = 20
    text_len = 10
    audio_features = torch.randn(batch_size, audio_len, N_MELS)
    text = torch.randint(0, VOCAB_SIZE, (batch_size, text_len + 1))
    tgt_input = text[:, :-1]
    tgt_output = text[:, 1:]
    tgt_mask = model.generate_square_subsequent_mask(text_len).unsqueeze(0).unsqueeze(0)
    output = model(audio_features, tgt_input, tgt_mask=tgt_mask)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
    loss.backward()

    assert output.shape == (2, 10, 100)
    assert loss.shape == ()
